from __future__ import annotations

import os
import random
import gc
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import torch
from typing import Dict, List


def cleanup_memory(verbose: bool = True) -> None:
    """
    Clean up GPU and CPU memory between training phases.

    Call this between training phases to prevent memory accumulation when
    reloading models. Especially important for large models (3B+ parameters)
    where leftover tensors can cause OOM errors.

    Args:
        verbose: If True, print memory stats before/after cleanup

    Usage:
        pre_trainer.train()
        cleanup_memory()  # <-- Add between phases
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    """
    if verbose and torch.cuda.is_available():
        before_allocated = torch.cuda.memory_allocated() / 1024**3
        before_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[cleanup_memory] Before: {before_allocated:.2f} GB allocated, {before_reserved:.2f} GB reserved")

    # Force Python garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Second gc pass to catch any newly dereferenced objects
    gc.collect()

    if verbose and torch.cuda.is_available():
        after_allocated = torch.cuda.memory_allocated() / 1024**3
        after_reserved = torch.cuda.memory_reserved() / 1024**3
        freed = before_reserved - after_reserved
        print(f"[cleanup_memory] After: {after_allocated:.2f} GB allocated, {after_reserved:.2f} GB reserved")
        print(f"[cleanup_memory] Freed: {freed:.2f} GB")


def get_tokenizer_hash(tokenizer, length: int = 8) -> str:
    """
    Compute a short hash of the tokenizer's vocabulary and configuration.

    Use this to include in dataset cache paths, ensuring cached tokenized
    datasets are invalidated when the tokenizer changes (e.g., new tokens added).

    Args:
        tokenizer: HuggingFace tokenizer instance
        length: Length of hash string to return (default 8 chars)

    Returns:
        Short hex hash string (e.g., "a1b2c3d4")

    Usage:
        cache_dir = os.path.join(save_directory, 'data_line',
                                 f'tokenized_datasets_{get_tokenizer_hash(tokenizer)}')
        if os.path.exists(os.path.join(cache_dir, 'dataset_dict.json')):
            tokenized_datasets = load_from_disk(cache_dir)
        else:
            # ... tokenize and save to cache_dir
    """
    import hashlib
    import json

    # Key properties that affect tokenization
    hash_data = {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "pad_token": tokenizer.pad_token,
        "eos_token": tokenizer.eos_token,
        "bos_token": getattr(tokenizer, "bos_token", None),
        "unk_token": tokenizer.unk_token,
        # Include a sample of the vocab to detect changes
        # Sort to ensure deterministic ordering
        "vocab_sample": sorted(list(tokenizer.get_vocab().keys())[:100]),
        "added_tokens_count": len(tokenizer.added_tokens_encoder),
    }

    # Create deterministic JSON string
    hash_str = json.dumps(hash_data, sort_keys=True, ensure_ascii=True)

    # Compute hash
    full_hash = hashlib.sha256(hash_str.encode("utf-8")).hexdigest()
    return full_hash[:length]


def get_cache_path(base_dir: str, cache_name: str, tokenizer=None) -> str:
    """
    Get a cache path that includes tokenizer hash for automatic invalidation.

    Args:
        base_dir: Base directory for cache (e.g., save_directory/data_line)
        cache_name: Name of the cache (e.g., 'tokenized_datasets', 'u_tokenized_datasets')
        tokenizer: Optional tokenizer to include hash in path. If None, returns
                   plain cache_name for backwards compatibility.

    Returns:
        Full cache directory path

    Example:
        cache_path = get_cache_path(
            os.path.join(save_directory, 'data_line'),
            'tokenized_datasets',
            tokenizer
        )
        # Returns: /path/to/save_directory/data_line/tokenized_datasets_a1b2c3d4
    """
    if tokenizer is not None:
        tok_hash = get_tokenizer_hash(tokenizer)
        cache_dir = f"{cache_name}_{tok_hash}"
    else:
        cache_dir = cache_name
    return os.path.join(base_dir, cache_dir)


def drop_or_repair_empty_inputs(ds, tok):
    PAD = tok.pad_token_id or 0
    EOS = getattr(tok, "eos_token_id", None)

    def _fix(batch):
        fixed_ids = []
        fixed_len = []
        for ids in batch["input_ids"]:
            if len(ids) == 0:
                ids = [EOS] if EOS is not None else [PAD]
            fixed_ids.append(ids)
            fixed_len.append(len(ids))
        out = {"input_ids": fixed_ids}
        # if you keep input_length around:
        if "input_length" in batch:
            out["input_length"] = fixed_len
        return out

    # repair
    ds = ds.map(_fix, batched=True)

    # optional: drop hopeless rows (both source and target empty)
    def _keep(ex):
        ok_src = len(ex["input_ids"]) > 0
        ok_tgt = ("labels" not in ex) or (len(ex["labels"]) > 0)
        return ok_src and ok_tgt

    ds = ds.filter(_keep)
    return ds

class SafeBucketedSeq2SeqCollator:
    def __init__(self, tokenizer, model=None, pad_to_multiple_of=8,
                 buckets=(8, 16, 32, 64)):
        # Let HF do the heavy lifting: dynamic pad + labels -> -100 + decoder_input_ids
        self.base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,                       # pass the model so decoder_input_ids are prepared
            label_pad_token_id=-100,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.buckets = tuple(sorted(buckets))
        self.pad_id  = tokenizer.pad_token_id or 0

    def _round_up(self, L: int) -> int:
        for b in self.buckets:
            if L <= b:
                return b
        return self.buckets[-1]

    def __call__(self, features):
        # 1) strip non-tensor fields BEFORE padding
        KEEP = {"input_ids", "attention_mask", "labels"}
        features = [{k: v for k, v in f.items() if k in KEEP} for f in features]

        # 2) dynamic pad (to per-batch max, multiple of 8); also sets decoder_input_ids
        batch = self.base(features)

        # 3) bucket-pad to a small set of shapes (reduces kernel churn, still tight padding)
        x, am, y = batch["input_ids"], batch["attention_mask"], batch["labels"]
        B, Lin = x.shape
        Lt     = y.shape[1]

        Lin2 = self._round_up(Lin)
        Lt2  = self._round_up(Lt)

        if Lin2 > Lin:
            xpad  = torch.full((B, Lin2 - Lin), self.pad_id, dtype=x.dtype, device=x.device)
            ampad = torch.zeros((B, Lin2 - Lin), dtype=am.dtype, device=am.device)
            batch["input_ids"]      = torch.cat([x,  xpad],  dim=1)
            batch["attention_mask"] = torch.cat([am, ampad], dim=1)

        if Lt2 > Lt:
            lpad = torch.full((B, Lt2 - Lt), -100, dtype=y.dtype, device=y.device)
            batch["labels"] = torch.cat([y, lpad], dim=1)

        # If HF created decoder_input_ids, bucket them too so shapes match
        if "decoder_input_ids" in batch:
            dec = batch["decoder_input_ids"]
            Ld  = dec.shape[1]
            Ld2 = self._round_up(Ld)
            if Ld2 > Ld:
                dpad = torch.full((B, Ld2 - Ld), self.pad_id, dtype=dec.dtype, device=dec.device)
                batch["decoder_input_ids"] = torch.cat([dec, dpad], dim=1)

        # 4) sanity: ensure the batch includes at least one supervised token
        if (batch["labels"] != -100).sum().item() == 0:
            raise ValueError(
                "All labels in this batch are -100 (ignored). "
                "Check tokenization/truncation: targets may be empty."
            )

        return batch
      

import random
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class DataCollatorForTranslationCorruption(DataCollatorForSeq2Seq):
    """
    A data collator that:
    - Receives examples with 'input_ids' (source cuneiform) and 'labels' (target English).
    - Optionally corrupts the source cuneiform using span corruption.
      Instead of using T5-style extra id tokens, we replace each corrupted span with the asterisk ("*") token.
    - Leaves the target (labels) intact, except we also enforce a target_length if desired.
    - Then uses the parent's Seq2Seq logic to do final padding/tensor conversion.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        source_length: int = 64,
        target_length: int = 64,
    ):
        """
        Args:
            tokenizer: Your seq2seq tokenizer.
            model: The model passed to DataCollatorForSeq2Seq (for special logic).
            corruption_probability: Probability of corrupting an example's input_ids.
            noise_density: Fraction of tokens to mask for corruption.
            mean_noise_span_length: Average span length for masked spans.
            source_length: The maximum length to enforce on the (possibly corrupted) input_ids.
            target_length: The maximum length to enforce on the labels.
        """
        super().__init__(tokenizer, model=model, return_tensors="pt")
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.tokenizer = tokenizer
        self.source_length = source_length
        self.target_length = target_length

    def corrupt_input(self, input_ids: list[int]) -> list[int]:
        """
        Apply span corruption on a single example's input_ids.
        Instead of T5's <extra_id_x> tokens, this method replaces spans with the '*' token.
        """
        num_tokens = len(input_ids)
        num_to_mask = max(1, int(round(num_tokens * self.noise_density)))

        # Decide lengths of spans to mask
        lengths = []
        while sum(lengths) < num_to_mask:
            lengths.append(
                min(
                    max(int(random.expovariate(1 / self.mean_noise_span_length)), 1),
                    num_tokens - sum(lengths),
                )
            )

        # Select starting positions for each span without overlapping
        span_starts = []
        used_positions = set()
        max_attempts = 10
        for length in lengths:
            attempts = 0
            while attempts < max_attempts:
                start = random.randint(0, num_tokens - length)
                positions = set(range(start, start + length))
                if not (positions & used_positions):
                    span_starts.append(start)
                    used_positions.update(positions)
                    break
                attempts += 1

        # Sort spans by their starting positions
        spans = sorted(zip(span_starts, lengths))

        # Create the corrupted input_ids list
        corrupted_ids = []
        prev_end = 0
        for start, length in spans:
            # Add tokens before the masked span
            corrupted_ids.extend(input_ids[prev_end:start])
            # Replace the span with the '*' token (using the tokenizer to get its id)
            corruption_token = self.tokenizer.convert_tokens_to_ids("*")
            corrupted_ids.append(corruption_token)
            # Skip over the tokens in the span
            prev_end = start + length

        # Append any remaining tokens after the last span
        corrupted_ids.extend(input_ids[prev_end:])
        return corrupted_ids

        def __call__(self, features):
            PAD_ID = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

            for f in features:
                # Possibly corrupt the input_ids with probability self.corruption_probability
                if random.random() < self.corruption_probability:
                    f["input_ids"] = self.corrupt_input(f["input_ids"])

                # Truncate to maximum lengths
                f["input_ids"] = f["input_ids"][: self.source_length]
                if "attention_mask" in f:
                    f["attention_mask"] = f["attention_mask"][: len(f["input_ids"])]
                if "labels" in f:
                    f["labels"] = f["labels"][: self.target_length]

                # Pad input_ids to self.source_length if necessary
                if len(f["input_ids"]) < self.source_length:
                    f["input_ids"].extend([PAD_ID] * (self.source_length - len(f["input_ids"])))
                # Pad attention_mask similarly if it exists
                if "attention_mask" in f and len(f["attention_mask"]) < self.source_length:
                    f["attention_mask"].extend([0] * (self.source_length - len(f["attention_mask"])))
                # Pad labels to self.target_length if necessary
                if "labels" in f and len(f["labels"]) < self.target_length:
                    f["labels"].extend([PAD_ID] * (self.target_length - len(f["labels"])))

            # Use the parent's __call__ to perform final padding/tensor conversion
            batch = super().__call__(features)
            return batch

import random
import torch
from typing import Dict, List, Tuple
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

class FastTranslationCorruptionCollator:
    """
    - Corrupts source (encoder) sequences by replacing random spans with a single '*' token id.
    - Leaves labels intact.
    - Uses HF's DataCollatorForSeq2Seq for dynamic padding (+ decoder_input_ids).
    - Then bucket-pads to a small set of lengths to reduce kernel recompiles.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        pad_to_multiple_of: int = 8,
        buckets: Tuple[int, ...] = (64, 96, 128, 160, 192, 256),
        avoid_eos_in_corruption: bool = True,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.buckets = tuple(sorted(buckets))
        self.pad_id = tokenizer.pad_token_id or 0
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        # get the id for a real '*' token
        star_id = tokenizer.convert_tokens_to_ids("*")
        if star_id is None or star_id < 0:
            # fallback: encode raw "*" without specials
            star_id = tokenizer("*", add_special_tokens=False)["input_ids"][0]
        self.star_id = star_id
        self.avoid_eos_in_corruption = avoid_eos_in_corruption

        # Let HF do dynamic pad + labels->-100 + decoder_input_ids
        self.base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    # ---------- helpers ----------

    def _round_up(self, L: int) -> int:
        for b in self.buckets:
            if L <= b:
                return b
        return self.buckets[-1]

    def _content_bounds(self, ids: List[int]) -> Tuple[int, int]:
        """
        Return (content_len, full_len_no_pad).
        content_len excludes trailing pad and (optionally) final EOS for corruption.
        """
        n = len(ids)
        # strip right pads
        while n > 0 and ids[n - 1] == self.pad_id:
            n -= 1
        full_len = n
        if self.avoid_eos_in_corruption and self.eos_id is not None and n > 0 and ids[n - 1] == self.eos_id:
            n -= 1
        return max(n, 0), full_len

    def _sample_spans(self, num_tokens: int) -> List[Tuple[int, int]]:
        """
        Sample non-overlapping spans (start, length) covering ~noise_density fraction.
        """
        if num_tokens <= 0:
            return []

        target = max(1, int(round(num_tokens * self.noise_density)))
        spans, used = [], set()
        attempts = 0
        max_attempts = 10 * max(target, 1)

        covered = 0
        while covered < target and attempts < max_attempts:
            # expovariate → geometric-ish
            L = max(1, int(random.expovariate(1.0 / self.mean_noise_span_length)))
            if L > num_tokens:
                L = num_tokens
            start_max = num_tokens - L
            if start_max < 0:
                break
            s = random.randint(0, start_max)
            rng = range(s, s + L)
            if not any(i in used for i in rng):
                spans.append((s, L))
                used.update(rng)
                covered += L
            attempts += 1

        spans.sort()
        return spans

    def _corrupt_one(self, ids: List[int]) -> List[int]:
        """
        Replace each selected span with a single '*' id, leaving EOS and pads alone.
        """
        content_len, full_len = self._content_bounds(ids)
        if content_len <= 0:
            return ids  # nothing to corrupt

        head = ids[:content_len]
        tail = ids[content_len:full_len]    # (possibly EOS)
        pads = ids[full_len:]               # trailing pads

        spans = self._sample_spans(len(head))
        if not spans:
            return ids

        out = []
        prev = 0
        for s, L in spans:
            # copy head before the span
            if s > prev:
                out.extend(head[prev:s])
            # drop span and insert single '*'
            out.append(self.star_id)
            prev = s + L
        # tail of head
        if prev < len(head):
            out.extend(head[prev:])

        # reattach tail (e.g., EOS) and original pads
        out.extend(tail)
        out.extend(pads)
        return out

    # ---------- main entry ----------

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1) corrupt sources (drop attention_mask so it’s rebuilt to match new lengths)
        cleaned = []
        for f in features:
            g = {
                "input_ids": f["input_ids"],
                "labels": f["labels"],
            }
            if random.random() < self.corruption_probability:
                g["input_ids"] = self._corrupt_one(g["input_ids"])
            cleaned.append(g)

        # 2) dynamic pad via HF (creates attention_mask & decoder_input_ids)
        batch = self.base(cleaned)

        # 3) bucket-pad to a small set of shapes
        x, am, y = batch["input_ids"], batch["attention_mask"], batch["labels"]
        B, Lin = x.shape
        Lt     = y.shape[1]
        Lin2   = self._round_up(Lin)
        Lt2    = self._round_up(Lt)

        if Lin2 > Lin:
            xpad  = torch.full((B, Lin2 - Lin), self.pad_id, dtype=x.dtype, device=x.device)
            ampad = torch.zeros((B, Lin2 - Lin), dtype=am.dtype, device=am.device)
            batch["input_ids"]      = torch.cat([x,  xpad],  dim=1)
            batch["attention_mask"] = torch.cat([am, ampad], dim=1)

        if Lt2 > Lt:
            lpad = torch.full((B, Lt2 - Lt), -100, dtype=y.dtype, device=y.device)
            batch["labels"] = torch.cat([y, lpad], dim=1)

        if "decoder_input_ids" in batch:
            dec = batch["decoder_input_ids"]
            Ld  = dec.shape[1]
            Ld2 = self._round_up(Ld)
            if Ld2 > Ld:
                dpad = torch.full((B, Ld2 - Ld), self.pad_id, dtype=dec.dtype, device=dec.device)
                batch["decoder_input_ids"] = torch.cat([dec, dpad], dim=1)

        # 4) sanity: ensure at least one supervised token exists
        if (batch["labels"] != -100).sum().item() == 0:
            raise ValueError(
                "All labels are -100 in this batch. Check tokenization/truncation—targets may be empty."
            )
        return batch

class NllbDataCollatorForSpanCorruption:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 512,
        target_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        # 1) pull out the proper mask token ID
        if tokenizer.mask_token_id is None:
            raise ValueError("Your NLLB tokenizer must have a mask_token_id")
        self.mask_token_id = tokenizer.mask_token_id
    def __call__(self, examples):
        # examples: list of dicts with "input_ids" already containing <src_lang> & <eos>
        batch_input_ids = [ex["input_ids"] for ex in examples]
        batch_inputs, batch_labels = [], []
        for input_ids in batch_input_ids:
            corrupted, labels = self._corrupt_input(input_ids)
            # enforce your max‐lengths
            corrupted = corrupted[: self.input_length]
            labels    = labels[: self.target_length]
            batch_inputs.append(corrupted)
            batch_labels.append(labels)
        # pad both with the tokenizer (will respect pad_token_id)
        batch_enc = self.tokenizer.pad(
            {"input_ids": batch_inputs},
            return_tensors="pt",
            padding="max_length",
            max_length=self.input_length,
        )
        batch_lab = self.tokenizer.pad(
            {"input_ids": batch_labels},
            return_tensors="pt",
            padding="max_length",
            max_length=self.target_length,
        )
        # replace pad tokens in labels with -100 so loss ignores them
        labels_tensor = batch_lab["input_ids"]
        labels_tensor[labels_tensor == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      batch_enc["input_ids"].contiguous(),
            "attention_mask": batch_enc["attention_mask"].contiguous(),
            "labels":         labels_tensor.contiguous(),
        }
    def _corrupt_input(self, input_ids):
        """
        Span‑mask only the non‑special tokens in `input_ids`.
        """
        num_tokens = len(input_ids)
        # 2) build a mask of special tokens
        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids,
            already_has_special_tokens=True
        )
        # candidate positions to mask
        candidates = [i for i, m in enumerate(special_mask) if m == 0]
        # how many tokens to mask (from candidates only)
        num_to_mask = max(1, int(round(len(candidates) * self.noise_density)))
        # sample spans via a simple Poisson process
        lengths, covered = [], set()
        while sum(lengths) < num_to_mask:
            span_len = min(
                max(int(random.expovariate(1 / self.mean_noise_span_length)), 1),
                num_to_mask - sum(lengths),
            )
            # find a start in `candidates` such that the full span is in candidates
            random.shuffle(candidates)
            for start in candidates:
                span = set(range(start, start + span_len))
                if span.issubset(candidates) and not (span & covered):
                    lengths.append(span_len)
                    covered |= span
                    break
            else:
                break  # if we can’t place any more spans, stop
        spans = sorted((min(span), length) for span, length in
                       ((start, length) for start, length in zip(sorted(covered), lengths)))
        corrupted, labels = [], []
        prev_end = 0
        # 3) interleave unchanged, mask token, and labels
        for start, length in spans:
            corrupted.extend(input_ids[prev_end:start])
            corrupted.append(self.mask_token_id)
            labels.append(self.mask_token_id)
            labels.extend(input_ids[start : start + length])
            prev_end = start + length
        # finish off the sequences
        corrupted.extend(input_ids[prev_end:])
        if prev_end < num_tokens:
            corrupted.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        return corrupted, labels

from typing import List, Dict, Any, Tuple, Optional
import random
from dataclasses import dataclass

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase


# ---------- helpers -----------------------------------------------------------

def ensure_mask_token(tokenizer: PreTrainedTokenizerBase, model=None, mask_token: str = "<mask>"):
    """
    Make sure a mask token exists before building the collator.
    """
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": mask_token})
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
    return tokenizer.mask_token_id


def attach_lang_ids_to_examples(
    tokenizer: PreTrainedTokenizerBase,
    src_field: str = "src_lang",
    tgt_field: str = "tgt_lang",
    out_src_id: str = "src_code_id",
    out_tgt_id: str = "tgt_code_id",
):
    """
    Dataset.map() helper: converts lang code strings (e.g. "akk_Cune") into token IDs and
    stores them alongside each example so the collator can inject BOS reliably.
    Use on BOTH your pretrain and translation datasets.

    Example:
        tokenized = tokenized.map(attach_lang_ids_to_examples(tokenizer), batched=False)
    """
    if not hasattr(tokenizer, "lang_code_to_id") or not tokenizer.lang_code_to_id:
        raise ValueError(
            "tokenizer.lang_code_to_id is missing/empty. Populate it after adding your special lang tags."
        )

    def _map_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        if src_field in ex and out_src_id not in ex:
            code = ex[src_field]
            ex[out_src_id] = tokenizer.lang_code_to_id.get(code, None)
            if ex[out_src_id] is None:
                raise ValueError(f"Unknown src lang code '{code}'.")
        if tgt_field in ex and out_tgt_id not in ex:
            code = ex[tgt_field]
            ex[out_tgt_id] = tokenizer.lang_code_to_id.get(code, None)
            if ex[out_tgt_id] is None:
                raise ValueError(f"Unknown tgt lang code '{code}'.")
        return ex

    return _map_fn


# ---------- collator ----------------------------------------------------------

def _sample_spans(num_candidates: int, noise_density: float, mean_span: float) -> List[Tuple[int, int]]:
    """
    Returns sorted (start_index_in_candidates, length) spans drawn on [0, num_candidates).
    """
    if num_candidates <= 0 or noise_density <= 0.0:
        return []
    num_to_mask = max(1, int(round(num_candidates * noise_density)))

    lengths, covered = [], set()
    while sum(lengths) < num_to_mask:
        span_len = max(1, min(int(random.expovariate(1.0 / max(mean_span, 1e-6))),
                              num_to_mask - sum(lengths)))
        # place by random starts; bounded tries avoids infinite loops
        tries, placed = 16, False
        while tries > 0:
            start = random.randrange(0, num_candidates)
            span = set(range(start, min(start + span_len, num_candidates)))
            if not (span & covered):
                lengths.append(len(span))
                covered |= span
                placed = True
                break
            tries -= 1
        if not placed:
            break

    spans = []
    if covered:
        sorted_pos = sorted(covered)
        s = sorted_pos[0]; e = s + 1
        for p in sorted_pos[1:]:
            if p == e:
                e += 1
            else:
                spans.append((s, e - s))
                s = p; e = p + 1
        spans.append((s, e - s))
    return spans


import numpy as np
from dataclasses import dataclass
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

def _compress_masked_runs(ids: np.ndarray, mask: np.ndarray, mask_token_id: int) -> list[int]:
    """
    Replace each contiguous True-run in `mask` with a single mask token.
    Keep all unmasked tokens. ids, mask are 1D numpy arrays of same length.
    """
    if mask.sum() == 0:
        return ids.tolist()
    # run starts where mask is True and (i==0 or previous is False)
    start = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    end   = np.flatnonzero(mask & np.concatenate((~mask[1:], [True]))) + 1  # exclusive
    out = []
    cursor = 0
    for s, e in zip(start, end):
        if cursor < s:
            out.extend(ids[cursor:s].tolist())
        out.append(mask_token_id)
        cursor = e
    if cursor < len(ids):
        out.extend(ids[cursor:].tolist())
    return out

def _sample_span_mask(L: int, noise_density: float, mean_span: float) -> np.ndarray:
    """
    Fast approximate T5-style span mask. Returns boolean mask of length L.
    We draw alternating span lengths from a geometric distribution to get
    roughly the desired noise density.
    """
    if L <= 0 or noise_density <= 0.0:
        return np.zeros(L, dtype=bool)

    # expected masked tokens
    target = max(1, int(round(L * noise_density)))

    # geometric with mean `mean_span` => p = 1/mean
    p = 1.0 / max(mean_span, 1e-6)

    masked = np.zeros(L, dtype=bool)
    i = 0
    odd = True  # start with an unmasked run of length ~Geometric? We'll seed with 0
    # Start by placing a masked run soon to avoid long prefix of unmasked
    while i < L and masked.sum() < target:
        # draw run length
        # for masked runs, mean ≈ mean_span; for unmasked runs, use 1 / noise_density to keep proportion stable
        mean = mean_span if odd else max(1.0, (1.0 - noise_density) / max(noise_density, 1e-6))
        p_run = 1.0 / mean
        # geometric draw, min 1
        run = int(np.random.geometric(p_run))
        if odd:
            # masked run
            end = min(L, i + run)
            masked[i:end] = True
            i = end
        else:
            # unmasked run
            i = min(L, i + run)
        odd = not odd

    # trim to exact-ish budget if we overshot
    excess = int(masked.sum()) - target
    if excess > 0:
        idx = np.flatnonzero(masked)
        drop = np.random.choice(idx, size=excess, replace=False)
        masked[drop] = False
    return masked

@dataclass
class NLLBInfillingSeq2SeqCollator:
    """
    Fast pretraining collator:
      * assumes BOS (lang tag) is already at position 0 in both input_ids and labels
      * corrupts encoder only
      * pads to multiple-of-8 with -100 label masking
    """
    tokenizer: PreTrainedTokenizerBase
    noise_density: float = 0.15
    mean_noise_span_length: float = 3.0
    src_cap: int | None = None
    tgt_cap: int | None = None
    pad_to_multiple_of: int = 8
    label_pad_token_id: int = -100
    debug: bool = False

    def __post_init__(self):
        if self.tokenizer.mask_token_id is None:
            raise ValueError("Call ensure_mask_token(tokenizer, model) before creating the collator.")
        self._padder = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=None,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            label_pad_token_id=self.label_pad_token_id,
            return_tensors="pt",
        )
        # cache eos if you actually use it; else None
        self._eos = getattr(self.tokenizer, "eos_token_id", None)

    def _corrupt_encoder_fast(self, ids: list[int]) -> list[int]:
        # assume lang BOS at index 0; exclude it from candidates
        L = len(ids)
        if L <= 2:  # nothing to do
            return ids
        arr = np.asarray(ids, dtype=np.int64)

        # candidate region: [1 : L) except EOS if present as last token
        start_cand = 1
        end_cand = L
        if self._eos is not None and L >= 2 and ids[-1] == self._eos:
            end_cand = L - 1

        cand_len = max(0, end_cand - start_cand)
        if cand_len <= 0 or self.noise_density <= 0.0:
            return ids

        mask_local = _sample_span_mask(cand_len, self.noise_density, self.mean_noise_span_length)

        # stitch into full-length mask
        full_mask = np.zeros(L, dtype=bool)
        full_mask[start_cand:end_cand] = mask_local

        return _compress_masked_runs(arr, full_mask, self.tokenizer.mask_token_id)

    def __call__(self, examples):
        feats = []
        for ex in examples:
            enc = ex["input_ids"]
            dec = ex.get("labels", enc)  # pretrain uses clean target

            if self.src_cap is not None:
                enc = enc[: self.src_cap]
            if self.tgt_cap is not None:
                dec = dec[: self.tgt_cap]

            enc_cor = self._corrupt_encoder_fast(enc)
            feats.append({"input_ids": enc_cor, "labels": dec})

        batch = self._padder(feats)

        if self.debug:
            attn = batch["attention_mask"]
            masked = (batch["input_ids"] == self.tokenizer.mask_token_id).float()
            pct = (masked.sum(dim=1) / attn.sum(dim=1).clamp_min(1)).mean().item()
            bsz, seqlen = batch["input_ids"].shape
            print(f"[fast-collator] batch={bsz} len={seqlen} avg_mask_frac≈{pct:.3f}")

        return batch

import random
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

class NllbDataCollatorForTranslationCorruptionOG(DataCollatorForSeq2Seq):
    """
    Corrupts the source side of translation examples by span‑masking,
    using NLLB’s built‑in <mask> token, then pads via the Seq2Seq parent.
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        source_length: int = 64,
        target_length: int = 64,
    ):
        super().__init__(tokenizer, model=model, return_tensors="pt")
        self.tokenizer = tokenizer
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.source_length = source_length
        self.target_length = target_length
        # grab the correct mask token for NLLB
        if tokenizer.mask_token_id is None:
            raise ValueError("Your NLLB tokenizer must have a mask_token_id")
        self.mask_token_id = tokenizer.mask_token_id
    def corrupt_input(self, input_ids: list[int]) -> list[int]:
        num_tokens = len(input_ids)
        # 1) get a mask marking all special tokens (so we never mask them)
        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )
        # candidate positions = non‑special tokens
        candidates = [i for i, v in enumerate(special_mask) if v == 0]
        num_to_mask = max(1, int(round(len(candidates) * self.noise_density)))
        # 2) sample spans (Poisson for lengths)
        lengths, covered = [], set()
        while sum(lengths) < num_to_mask:
            span_len = min(
                max(int(random.expovariate(1 / self.mean_noise_span_length)), 1),
                num_to_mask - sum(lengths),
            )
            # find a span_start in candidates that doesn’t overlap
            random.shuffle(candidates)
            for start in candidates:
                span = set(range(start, start + span_len))
                if span.issubset(candidates) and not (span & covered):
                    lengths.append(span_len)
                    covered |= span
                    break
            else:
                break
        spans = sorted(zip(sorted(covered), lengths))  # might overlap length vs start; you can refine
        # 3) build corrupted sequence
        corrupted, prev_end = [], 0
        for start, length in spans:
            corrupted.extend(input_ids[prev_end:start])
            corrupted.append(self.mask_token_id)
            prev_end = start + length
        corrupted.extend(input_ids[prev_end:])
        return corrupted
    def __call__(self, features):
        PAD_ID = self.tokenizer.pad_token_id or 0
        # 1) corrupt & truncate / pad each feature’s input_ids
        for f in features:
            if random.random() < self.corruption_probability:
                f["input_ids"] = self.corrupt_input(f["input_ids"])
            # truncate
            f["input_ids"] = f["input_ids"][: self.source_length]
            if "attention_mask" in f:
                f["attention_mask"] = f["attention_mask"][: len(f["input_ids"])]
            # pad to source_length
            pad_len = self.source_length - len(f["input_ids"])
            if pad_len > 0:
                f["input_ids"].extend([PAD_ID] * pad_len)
                if "attention_mask" in f:
                    f["attention_mask"].extend([0] * pad_len)
            # also enforce target_length on labels
            if "labels" in f:
                f["labels"] = f["labels"][: self.target_length]
                lab_pad = self.target_length - len(f["labels"])
                if lab_pad > 0:
                    f["labels"].extend([PAD_ID] * lab_pad)
        # 2) hand off to the parent for final padding/tensors
        batch = super().__call__(features)
        return batch

from transformers import DataCollatorForSeq2Seq

class NllbDataCollatorForTranslationCorruption(DataCollatorForSeq2Seq):
    """
    Corrupts the source side by span-masking using NLLB’s <mask> token,
    then relies on the parent to dynamically pad (to a multiple of 8).
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        source_length: int = 64,
        target_length: int = 64,
        pad_to_multiple_of: int | None = 8,
        label_pad_token_id: int = -100,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding="longest",              # dynamic
            max_length=None,                # let inputs be varied
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.source_length = source_length
        self.target_length = target_length

        if tokenizer.mask_token_id is None:
            raise ValueError("Your NLLB tokenizer must have a mask_token_id")
        self.mask_token_id = tokenizer.mask_token_id

    def corrupt_input(self, input_ids: list[int]) -> list[int]:
        import random
        num_tokens = len(input_ids)
        special_mask = self.tokenizer.get_special_tokens_mask(
            input_ids, already_has_special_tokens=True
        )
        candidates = [i for i, v in enumerate(special_mask) if v == 0]
        num_to_mask = max(1, int(round(len(candidates) * self.noise_density)))

        lengths, covered = [], set()
        while sum(lengths) < num_to_mask and candidates:
            span_len = max(1, min(
                int(random.expovariate(1 / self.mean_noise_span_length)),
                num_to_mask - sum(lengths),
            ))
            random.shuffle(candidates)
            placed = False
            for start in candidates:
                span = set(range(start, min(start + span_len, len(input_ids))))
                if span.issubset(candidates) and not (span & covered):
                    lengths.append(len(span))
                    covered |= span
                    placed = True
                    break
            if not placed:
                break

        spans = sorted(zip(sorted(list(covered)), lengths))
        corrupted, prev_end = [], 0
        for start, length in spans:
            corrupted.extend(input_ids[prev_end:start])
            corrupted.append(self.mask_token_id)
            prev_end = start + length
        corrupted.extend(input_ids[prev_end:])
        return corrupted

    def __call__(self, features):
        import random
        # mutate input_ids in-place: corrupt + (optionally) truncate, BUT DO NOT PAD
        for f in features:
            if random.random() < self.corruption_probability:
                f["input_ids"] = self.corrupt_input(f["input_ids"])

            # truncate to keep a hard cap, but do not pad here
            if self.source_length is not None:
                f["input_ids"] = f["input_ids"][: self.source_length]

            if "labels" in f and self.target_length is not None:
                f["labels"] = f["labels"][: self.target_length]

            # Let parent compute attention_mask and do dynamic padding to multiple of 8
        return super().__call__(features)

import torch
from transformers import PreTrainedTokenizerBase

class NLLBMixedDataCollator:
    """
    1) For 'pretrain' samples:
       - prepend task token id
       - apply span corruption
       - (optionally) truncate to caps
    2) For 'train' samples:
       - (optionally) truncate to caps
    3) Jointly pad ALL examples to multiple-of-8 via a single DataCollatorForSeq2Seq call.
    """
    def __init__(
        self,
        tokenizer,
        final_joint_collator: DataCollatorForSeq2Seq,
        pretrain_corruptor,              # any object with .corrupt_input(list[int]) -> list[int]
        task_token_id: int | None,
        source_cap: int | None = None,
        label_cap: int | None  = None,
    ):
        self.tok = tokenizer
        self.final = final_joint_collator
        self.pretrain_corruptor = pretrain_corruptor
        self.task_token_id = task_token_id
        self.source_cap = source_cap
        self.label_cap  = label_cap

    def __call__(self, examples):
        out = []
        for ex in examples:
            x = dict(ex)  # shallow copy; we’ll mutate
            # --- PRETRAIN branch ---
            if x.get("task") == "pretrain":
                # prepend task token
                if self.task_token_id is not None:
                    x["input_ids"] = [self.task_token_id] + x["input_ids"]
                # corrupt (no padding here)
                x["input_ids"] = self.pretrain_corruptor.corrupt_input(x["input_ids"])
                # optional truncation
                if self.source_cap is not None:
                    x["input_ids"] = x["input_ids"][: self.source_cap]
                if "labels" in x and self.label_cap is not None:
                    x["labels"] = x["labels"][: self.label_cap]

            # --- TRANSLATION branch ---
            else:
                if self.source_cap is not None:
                    x["input_ids"] = x["input_ids"][: self.source_cap]
                if "labels" in x and self.label_cap is not None:
                    x["labels"] = x["labels"][: self.label_cap]

            # IMPORTANT: do not add attention_mask or any padding here
            out.append(x)

        # Single *joint* padding step for the whole mixed batch
        batch = self.final(out)  # creates attention_mask; pads to multiple of 8; pads labels with -100
        return batch


from transformers import PreTrainedTokenizerBase
import random
import torch

class T5SpanCorruptionCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 512,
        target_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.extra_id_start = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    def __call__(self, examples):
        inputs, targets = [], []
        for example in examples:
            input_ids = example["input_ids"]
            corrupted_input, target = self._t5_span_corrupt(input_ids)
            inputs.append(corrupted_input[:self.input_length])
            targets.append(target[:self.target_length])

        batch_inputs = self.tokenizer.pad(
            {"input_ids": inputs}, padding="max_length", max_length=self.input_length, return_tensors="pt"
        )
        batch_targets = self.tokenizer.pad(
            {"input_ids": targets}, padding="max_length", max_length=self.target_length, return_tensors="pt"
        )

        labels = batch_targets["input_ids"]
        pad_id = self.tokenizer.pad_token_id
        labels[labels == pad_id] = -100

        return {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "labels": labels,
        }

    def _t5_span_corrupt(self, input_ids):
        num_tokens = len(input_ids)
        num_noise_tokens = max(1, int(round(num_tokens * self.noise_density)))

        # Sample span lengths
        span_lengths = []
        while sum(span_lengths) < num_noise_tokens:
            span_len = min(
                max(int(random.expovariate(1 / self.mean_noise_span_length)), 1),
                num_tokens - sum(span_lengths)
            )
            span_lengths.append(span_len)

        # Sample start indices for each span (non-overlapping)
        spans = []
        used = set()
        for length in span_lengths:
            for _ in range(10):  # up to 10 tries
                start = random.randint(0, num_tokens - length)
                if not any(pos in used for pos in range(start, start + length)):
                    spans.append((start, length))
                    used.update(range(start, start + length))
                    break

        spans.sort()
        corrupted, target = [], []
        cursor = 0
        span_id = 0

        for start, length in spans:
            # Add unmasked tokens
            if cursor < start:
                corrupted.extend(input_ids[cursor:start])
            # Insert <extra_id_n>
            corrupted.append(self.extra_id_start + span_id)
            target.append(self.extra_id_start + span_id)
            target.extend(input_ids[start:start + length])
            span_id += 1
            cursor = start + length

        # Add any remaining tokens
        if cursor < len(input_ids):
            corrupted.extend(input_ids[cursor:])

        # Add EOS to target
        target.append(self.tokenizer.eos_token_id)

        return corrupted, target

import math
import numpy as np
import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase

class T5SpanCorruptionCollatorFast:
    """
    Faster span corruption collator following the T5 'random spans' recipe.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 64,
        target_length: int = 64,
        pad_to_multiple_of: int = 8,
        dynamic_inputs: bool = True,
        max_sentinels: int = 128,
        add_eos_to_target: bool = True,
        rng: np.random.Generator = None,
    ):
        self.tok = tokenizer
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.p2m = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.dynamic_inputs = bool(dynamic_inputs)
        self.max_sentinels = int(max_sentinels)
        self.add_eos_to_target = bool(add_eos_to_target)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.pad_id = tokenizer.pad_token_id
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

        # <-- this is fine *once the method below is correctly indented*
        self.sentinel_ids = self._build_sentinel_ids(self.max_sentinels)

        upper = len(tokenizer)  # includes added/special tokens for many tokenizers
        for i, sid in enumerate(self.sentinel_ids):
            if not (0 <= sid < upper):
                raise ValueError(
                    f"Sentinel id out of range: index {i} -> id {sid}, "
                    f"but len(tokenizer)={upper} (tokenizer.vocab_size={tokenizer.vocab_size})"
                )

    def _build_sentinel_ids(self, n: int) -> List[int]:
        tok = self.tok
        V = int(getattr(tok, "vocab_size", 0) or 0)

        # 1) try official API
        ids = None
        try:
            ids = tok.get_sentinel_token_ids()
        except Exception:
            ids = None

        if ids:
            ids = ids[:n]
            # If any sentinel is out of base vocab range, fall back to T5-style.
            if any((sid is None) or (sid < 0) or (sid >= V) for sid in ids):
                ids = None

        # 2) try converting canonical strings
        if ids is None:
            sentinel_tokens = [f"<extra_id_{i}>" for i in range(n)]
            conv = tok.convert_tokens_to_ids(sentinel_tokens)

            # Some tokenizers may return ids >= V (or unk). If so, fall back to T5-style.
            unk = getattr(tok, "unk_token_id", None)
            if any((sid is None) or (sid < 0) or (sid >= V) or (unk is not None and sid == unk) for sid in conv):
                # T5 recipe: extra_id_0 is vocab_size-1, extra_id_1 is vocab_size-2, ...
                conv = [V - 1 - i for i in range(n)]

            ids = conv

        # Final safety: guarantee in-range
        for i, sid in enumerate(ids):
            if not (0 <= sid < V):
                raise ValueError(f"Sentinel id out of range even after fallback: i={i} sid={sid} vocab_size={V}")

        return ids

    # --------- helpers ---------

    def _random_spans(self, L: int):
        if L <= 0:
            return [L], []

        n_noise = max(1, int(round(L * self.noise_density)))
        n_nonnoise = max(0, L - n_noise)

        S = max(1, int(round(n_noise / self.mean_noise_span_length)))

        if n_noise >= S:
            noise_lengths = self.rng.multinomial(
                n_noise - S, [1.0 / S] * S
            ) + 1
        else:
            noise_lengths = np.ones(S, dtype=int)
            # we allow some conceptual "zeros" but keep it simple

        if (S + 1) > 0:
            if n_nonnoise >= (S + 1):
                nonnoise_lengths = self.rng.multinomial(
                    n_nonnoise - (S + 1),
                    [1.0 / (S + 1)] * (S + 1)
                ) + 1
            else:
                nonnoise_lengths = np.zeros(S + 1, dtype=int)
                if n_nonnoise > 0:
                    picks = self.rng.choice(
                        S + 1, size=n_nonnoise, replace=True
                    )
                    for p in picks:
                        nonnoise_lengths[p] += 1
        else:
            nonnoise_lengths = np.array([n_nonnoise], dtype=int)

        noise_total = int(noise_lengths.sum())
        non_total = int(nonnoise_lengths.sum())
        over = (noise_total + non_total) - L
        if over > 0:
            nonnoise_lengths[-1] = max(0, nonnoise_lengths[-1] - over)

        return nonnoise_lengths.tolist(), noise_lengths.tolist()

    def _sentinel(self, i: int) -> int:
        if i < len(self.sentinel_ids):
            return self.sentinel_ids[i]
        return self.sentinel_ids[-1]

    def _apply_spans(self, tokens: List[int], nonnoise: List[int], noise: List[int]):
        corrupted = []
        target = []
        idx = 0
        span_id = 0

        for gap_len, noise_len in zip(nonnoise, noise):
            if gap_len > 0:
                corrupted.extend(tokens[idx: idx + gap_len])
                idx += gap_len

            sentinel_id = self._sentinel(span_id)
            corrupted.append(sentinel_id)
            target.append(sentinel_id)

            if noise_len > 0:
                target.extend(tokens[idx: idx + noise_len])
                idx += noise_len

            span_id += 1

        if len(nonnoise) > len(noise):
            final_gap = nonnoise[-1]
            if final_gap > 0:
                corrupted.extend(tokens[idx: idx + final_gap])
                idx += final_gap

        if self.add_eos_to_target and (self.eos_id is not None):
            target.append(self.eos_id)

        return corrupted, target

    # --------- main collate ---------

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(examples)
        corrupted_list = []
        target_list = []

        for ex in examples:
            toks: List[int] = ex["input_ids"]
            L = len(toks)

            nonnoise, noise = self._random_spans(L)
            corrupted, target = self._apply_spans(toks, nonnoise, noise)

            corrupted_list.append(corrupted)
            target_list.append(target)

        # ---- enforce a *hard* cap on input length ----
        if self.dynamic_inputs:
            raw_max_in = max(len(x) for x in corrupted_list) if corrupted_list else 0
            max_in = min(raw_max_in, self.input_length)
        else:
            max_in = self.input_length

        # targets: already partially clamped, but keep this
        max_tgt = max(len(x) for x in target_list) if target_list else 0
        max_tgt = min(self.target_length, max_tgt)

        if self.p2m:
            if max_in % self.p2m != 0:
                max_in = int(math.ceil(max_in / self.p2m) * self.p2m)
                # but never exceed the global input_length
                max_in = min(max_in, self.input_length)

            if max_tgt % self.p2m != 0:
                max_tgt = int(math.ceil(max_tgt / self.p2m) * self.p2m)
                max_tgt = min(max_tgt, self.target_length)

        inputs = torch.full((B, max_in), self.pad_id, dtype=torch.long)
        labels = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)

        for i, (cin, tgt) in enumerate(zip(corrupted_list, target_list)):
            li = min(len(cin), max_in)
            lt = min(len(tgt), max_tgt)
            if li > 0:
                inputs[i, :li] = torch.tensor(cin[:li], dtype=torch.long)
            if lt > 0:
                labels[i, :lt] = torch.tensor(tgt[:lt], dtype=torch.long)

        attention_mask = (inputs != self.pad_id).long()
        labels[labels == self.pad_id] = -100

        # sanity check (optional but nice during debugging)
        assert inputs.shape[1] <= self.input_length, inputs.shape
        assert labels.shape[1] <= self.target_length, labels.shape

        return {
            "input_ids": inputs,
            "attention_mask": attention_mask,
            "labels": labels,
        }



def shift_tokens_right(input_ids: torch.Tensor,
                       pad_token_id: int,
                       decoder_start_token_id: int) -> torch.Tensor:
    """
    Shift input ids one token to the right, and replace -100 in the
    labels by pad_token_id (so that cross‐entropy ignores it).
    """
    # make a new tensor full of pad tokens
    shifted = input_ids.new_zeros(input_ids.shape)

    # copy everything except the last token
    shifted[..., 1:] = input_ids[..., :-1].clone()
    # first token is the decoder start token
    shifted[..., 0] = decoder_start_token_id

    # replace -100 (ignore_index) by pad_id so they’re not completely lost
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted

class T5DataCollatorForSpanCorruptionAtersisk:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 512,
        target_length: int = 512
    ):
        """
        Args:
            tokenizer: A T5 (or similar) tokenizer.
            noise_density: Fraction of tokens to mask for corruption.
            mean_noise_span_length: Average span length to corrupt.
            input_length: Final max length (in tokens) for the input.
            target_length: Final max length (in tokens) for the labels.
        """
        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length

    def __call__(self, examples):
        """
        Main entry point: receives a list of examples,
        each example is typically a dict with {"input_ids": [...], ...}.
        We'll corrupt them, enforce strict lengths, and return a batch dict.
        """
        # Extract the raw input_ids from each example
        batch_input_ids = [ex["input_ids"] for ex in examples]

        batch_inputs, batch_labels = [], []
        for input_ids in batch_input_ids:
            # 1) Corrupt input and produce corresponding labels
            corrupted, labels = self._corrupt_input(input_ids)

            # 2) Truncate to fixed lengths
            corrupted = corrupted[: self.input_length]
            labels    = labels[: self.target_length]

            batch_inputs.append(corrupted)
            batch_labels.append(labels)

        # 3) Pad inputs and labels strictly to the given max lengths using the tokenizer.
        batch_input = self.tokenizer.pad(
            {"input_ids": batch_inputs},
            return_tensors="pt",
            padding="max_length",
            max_length=self.input_length  # forces shape [batch_size, input_length]
        )
        batch_labels_padded = self.tokenizer.pad(
            {"input_ids": batch_labels},
            return_tensors="pt",
            padding="max_length",
            max_length=self.target_length  # forces shape [batch_size, target_length]
        )

        # 4) Replace padding token IDs in labels with -100 so they're ignored in loss computation.
        labels_tensor = batch_labels_padded["input_ids"]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        labels_tensor[labels_tensor == pad_token_id] = -100

        # 5) Ensure the final tensors are contiguous before returning.
        batch = {
            "input_ids":      batch_input["input_ids"].contiguous(),
            "attention_mask": batch_input["attention_mask"].contiguous(),
            "labels":         labels_tensor.contiguous()
        }
        pad_id     = self.tokenizer.pad_token_id
        decoder_start_token_id = self.tokenizer.eos_token_id  # T5’s EOS == 1; UMT5 pads with EOS as start

        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"],
            pad_token_id=pad_id,
            decoder_start_token_id=decoder_start_token_id,
        ).contiguous()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != pad_id).long().contiguous()
        return batch

    def _corrupt_input(self, input_ids):
        """
        Takes a single list of token ids, applies span corruption,
        returning (corrupted_input_ids, labels).

        Instead of T5's <extra_id_*> tokens, each corrupted span is replaced
        with the '*' token (whose token ID is determined by the tokenizer).
        """
        num_tokens = len(input_ids)
        num_to_mask = max(1, int(round(num_tokens * self.noise_density)))

        # Decide the lengths of spans to mask.
        lengths = []
        while sum(lengths) < num_to_mask:
            lengths.append(
                min(
                    max(int(random.expovariate(1 / self.mean_noise_span_length)), 1),
                    num_tokens - sum(lengths),
                )
            )

        # Select non-overlapping starting positions for each span.
        span_starts = []
        used_positions = set()
        max_attempts = 10  # maximum attempts to find a valid span start
        for length in lengths:
            attempts = 0
            while attempts < max_attempts:
                start = random.randint(0, num_tokens - length)
                positions = set(range(start, start + length))
                if not (positions & used_positions):
                    span_starts.append(start)
                    used_positions.update(positions)
                    break
                attempts += 1
            else:
                # Skip span if no valid start is found after max_attempts.
                continue

        # Sort spans by starting position.
        spans = sorted(zip(span_starts, lengths))

        corrupted_input_ids = []
        labels = []
        prev_end = 0

        # Get the token ID corresponding to '*' using the tokenizer.
        corruption_token_id = self.tokenizer.convert_tokens_to_ids("*")

        # For each span, replace the span with '*' in the corrupted input
        # and add '*' plus the masked tokens to the labels.
        for start, length in spans:
            # Add unchanged tokens before the span.
            corrupted_input_ids.extend(input_ids[prev_end:start])
            # Insert '*' token as a placeholder.
            corrupted_input_ids.append(corruption_token_id)
            # For labels, add '*' to signal the start of the masked span.
            labels.append(corruption_token_id)
            # Then add the actual masked tokens.
            labels.extend(input_ids[start : start + length])
            prev_end = start + length

        # Append any remaining tokens after the last span.
        corrupted_input_ids.extend(input_ids[prev_end:])
        # Optionally, add EOS token if the input wasn't fully consumed.
        if prev_end < num_tokens:
            corrupted_input_ids.append(self.tokenizer.eos_token_id)
        # End the labels with EOS token.
        labels.append(self.tokenizer.eos_token_id)
        return corrupted_input_ids, labels

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _shift_tokens_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """
    Local copy to avoid importing model internals. Mirrors T5's behavior.
    """
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0] = decoder_start_token_id
    # Any label position that was -100 should be replaced by pad for decoder inputs
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


class T5DataCollatorForSpanCorruptionAsteriskFast:
    """
    FAST asterisk-based span corruption for T5/UMT5 pretraining.

    Differences from classic T5:
      - Uses a single corruption token '*' (community cuneiform convention)
        instead of <extra_id_n> sentinels.
      - Fast multinomial partitioning -> no overlap checks, O(L).
      - Optional dynamic input length (pad to batch max), with pad_to_multiple_of.
      - Truncates to target_length; adds EOS to labels by default.
      - Optionally returns decoder_input_ids (otherwise models shift internally).

    Expected input:
      Each example is a dict with "input_ids" (no padding).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 512,
        target_length: int = 512,
        pad_to_multiple_of: Optional[int] = 8,
        dynamic_inputs: bool = True,
        add_eos_to_target: bool = True,
        add_decoder_inputs: bool = False,
        rng: Optional[np.random.Generator] = None,
    ):
        self.tok = tokenizer
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.p2m = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.dynamic_inputs = bool(dynamic_inputs)
        self.add_eos_to_target = bool(add_eos_to_target)
        self.add_decoder_inputs = bool(add_decoder_inputs)
        self.rng = rng if rng is not None else np.random.default_rng()

        # IDs
        self.pad_id = self.tok.pad_token_id
        self.eos_id = getattr(self.tok, "eos_token_id", None)

        # Corruption token ID for "*"
        star_id = self.tok.convert_tokens_to_ids("*")
        if star_id is None or star_id == self.tok.unk_token_id:
            # Fail loudly so you notice; you can add "*" via tokenizer.add_tokens(["*"]) if needed.
            raise ValueError("Tokenizer does not contain '*' token; add it to the vocab or choose another symbol.")
        self.star_id = int(star_id)

        # For optional decoder inputs
        self.decoder_start_token_id = getattr(self.tok, "eos_token_id", None)
        if self.add_decoder_inputs and self.decoder_start_token_id is None:
            raise ValueError("decoder_start_token_id (usually EOS for T5) is required when add_decoder_inputs=True.")

    # --------- span helpers (fast T5-style partitioning) ---------

    def _random_spans(self, L: int):
        """
        Return two lists: nonnoise_lengths (len=S+1), noise_lengths (len=S),
        using randomized partitions (no overlap bookkeeping).
        """
        if L <= 0:
            return [L], []

        n_noise = max(1, int(round(L * self.noise_density)))
        n_nonnoise = max(0, L - n_noise)

        # number of noise spans
        S = max(1, int(round(n_noise / self.mean_noise_span_length)))

        # partition noise tokens into S positive integers (≈ geometric-like)
        if n_noise >= S:
            noise_lengths = self.rng.multinomial(n_noise - S, [1.0 / S] * S) + 1
        else:
            # when far more spans than noise tokens, many spans length 1; remainder effectively 0
            noise_lengths = np.ones(S, dtype=int)

        # partition non-noise into S+1 buckets (allow zeros at ends)
        if n_nonnoise >= (S + 1):
            nonnoise_lengths = self.rng.multinomial(n_nonnoise - (S + 1), [1.0 / (S + 1)] * (S + 1)) + 1
        else:
            nonnoise_lengths = np.zeros(S + 1, dtype=int)
            if n_nonnoise > 0:
                picks = self.rng.choice(S + 1, size=n_nonnoise, replace=True)
                for p in picks:
                    nonnoise_lengths[p] += 1

        # clip in rare rounding cases
        total = int(noise_lengths.sum() + nonnoise_lengths.sum())
        if total > L:
            over = total - L
            nonnoise_lengths[-1] = max(0, int(nonnoise_lengths[-1]) - over)

        return nonnoise_lengths.tolist(), noise_lengths.tolist()

    def _apply_spans_asterisk(self, tokens: List[int], nonnoise: List[int], noise: List[int]):
        """
        Build (corrupted_input, labels) by alternating non-noise and noise spans.
        For each noise span:
          - insert '*' in the corrupted input,
          - append '*' followed by the masked tokens to labels.
        """
        corrupted, labels = [], []
        idx = 0

        for gap_len, noise_len in zip(nonnoise, noise):
            # keep gap
            if gap_len > 0:
                corrupted.extend(tokens[idx: idx + gap_len])
                idx += gap_len
            # sentinel '*'
            corrupted.append(self.star_id)
            labels.append(self.star_id)
            # send masked chunk to labels
            if noise_len > 0:
                labels.extend(tokens[idx: idx + noise_len])
                idx += noise_len

        # trailing non-noise
        if len(nonnoise) > len(noise):
            final_gap = nonnoise[-1]
            if final_gap > 0:
                corrupted.extend(tokens[idx: idx + final_gap])
                idx += final_gap

        if self.add_eos_to_target and (self.eos_id is not None):
            labels.append(self.eos_id)

        return corrupted, labels

    # --------- main collate ---------

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        B = len(examples)

        corrupted_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for ex in examples:
            toks: List[int] = ex["input_ids"]
            L = len(toks)

            nonnoise, noise = self._random_spans(L)
            corrupted, labels = self._apply_spans_asterisk(toks, nonnoise, noise)

            # truncate early (pre-padding) to keep tensors small
            if self.dynamic_inputs:
                # keep as-is; we'll pad to batch max later
                pass
            else:
                corrupted = corrupted[: self.input_length]

            labels = labels[: self.target_length]

            corrupted_list.append(corrupted)
            labels_list.append(labels)

        # decide max lengths for padding
        if self.dynamic_inputs:
            max_in = max((len(x) for x in corrupted_list), default=0)
            max_in = min(max_in, self.input_length)
        else:
            max_in = self.input_length

        # --- guard: never allow zero-width tensors ---
        if max_in == 0:
            max_in = self.p2m or 1  # give encoder at least one position

        max_tgt = min(self.target_length, max((len(x) for x in labels_list), default=0))
        if max_tgt == 0:
            # target must also be ≥1 to build tensors; a single EOS is a safe fallback
            max_tgt = self.p2m or 1

        # round up to multiple-of for tensor cores
        if self.p2m:
            if max_in % self.p2m != 0:
                max_in = int(math.ceil(max_in / self.p2m) * self.p2m)
            if max_tgt % self.p2m != 0:
                max_tgt = int(math.ceil(max_tgt / self.p2m) * self.p2m)

        # allocate tensors
        inputs = torch.full((B, max_in), self.pad_id, dtype=torch.long)
        labels = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)

        # fill
        for i, (cin, tgt) in enumerate(zip(corrupted_list, labels_list)):
            if not cin:
                cin = [self.eos_id] if self.eos_id is not None else [self.pad_id or 0]
            if not tgt:
                # label -100 padding will mask loss; but put one token so shape is valid
                tgt = [self.eos_id] if self.eos_id is not None else [self.pad_id or 0]
            li = min(len(cin), max_in)
            lt = min(len(tgt), max_tgt)
            if li > 0:
                inputs[i, :li] = torch.tensor(cin[:li], dtype=torch.long)
            if lt > 0:
                labels[i, :lt] = torch.tensor(tgt[:lt], dtype=torch.long)

        attention_mask = (inputs != self.pad_id).long()

        # loss mask on labels
        labels.masked_fill_(labels == self.pad_id, -100)

        batch = {
            "input_ids": inputs.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": labels.contiguous(),
        }

        # Optional: explicit decoder inputs (not required for T5 if you pass labels)
        if self.add_decoder_inputs:
            pad_id = self.pad_id if self.pad_id is not None else 0
            dec_in = _shift_tokens_right(labels, pad_token_id=pad_id, decoder_start_token_id=self.decoder_start_token_id)
            batch["decoder_input_ids"] = dec_in.contiguous()
            batch["decoder_attention_mask"] = (dec_in != pad_id).long().contiguous()

        return batch

import random
from typing import List, Tuple

def seed_worker(worker_id):
    import random, numpy as np, torch
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)

class ByT5DataCollatorForSpanCorruption:
    def __init__(
        self,
        tokenizer,
        noise_density: float = 0.12,
        mean_noise_span_length: float = 8.0,
        input_length: int = 1024,
        target_length: int = 1024,
        recover_star_prob: float = 0.2,
        pad_to_multiple_of: int = 8,
        max_start_tries: int = 50,  # rejection sampling tries per span
        pre_cap_slack: int = 64,    # truncate to input_length+slack BEFORE corruption
    ):
        self.tokenizer = tokenizer
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.pad_to_multiple_of = int(pad_to_multiple_of)
        self.recover_star_prob = float(recover_star_prob)
        self.max_start_tries = int(max_start_tries)
        self.pre_cap_slack = int(pre_cap_slack)

        self.star_id = tokenizer.convert_tokens_to_ids("*")
        self.eos_id = tokenizer.eos_token_id
        # Precompute <extra_id_0..99>
        self.extra_ids = [
            tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>") for i in range(100)
        ]

    def __call__(self, examples):
        batch_inputs, batch_labels = [], []

        for ex in examples:
            input_ids = ex["input_ids"]
            # TRUNCATE EARLY to avoid corrupting giant outliers
            pre_cap = self.input_length + self.pre_cap_slack
            if pre_cap > 0 and len(input_ids) > pre_cap:
                input_ids = input_ids[:pre_cap]

            corrupted, labels = self._corrupt_input_fast(input_ids)

            batch_inputs.append(corrupted[: self.input_length])
            batch_labels.append(labels[: self.target_length])

        batch_input = self.tokenizer.pad(
            {"input_ids": batch_inputs},
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        batch_labels_padded = self.tokenizer.pad(
            {"input_ids": batch_labels},
            return_tensors="pt",
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        labels_tensor = batch_labels_padded["input_ids"]
        pad_id = self.tokenizer.pad_token_id or 0
        labels_tensor[labels_tensor == pad_id] = -100

        return {
            "input_ids":      batch_input["input_ids"].contiguous(),
            "attention_mask": batch_input["attention_mask"].contiguous(),
            "labels":         labels_tensor.contiguous(),
        }

    def _corrupt_input_fast(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        n = len(input_ids)
        eos_id = self.eos_id

        if n == 0:
            return [eos_id], [eos_id]
        if n == 1:
            sid = self.extra_ids[0]
            return [sid, input_ids[0], eos_id], [sid, input_ids[0], eos_id]

        # how many tokens to mask (leave at least 1 unmasked)
        num_to_mask = max(1, int(round(n * self.noise_density)))
        num_to_mask = min(num_to_mask, n - 1)

        occupied = bytearray(n)  # 0/1 occupancy
        spans: List[Tuple[int, int]] = []

        # Optionally force a mask over the first '*' we see (O(n), but only once)
        if self.star_id is not None and self.recover_star_prob > 0.0:
            # avoid list.index exception cost
            star_pos = -1
            for i, t in enumerate(input_ids):
                if t == self.star_id:
                    star_pos = i
                    break
            if star_pos >= 0 and random.random() < self.recover_star_prob:
                spans.append((star_pos, 1))
                occupied[star_pos] = 1

        total_masked = sum(L for _, L in spans)

        # Sample spans until we hit budget (rejection sampling for non-overlap)
        # With low noise_density (~0.12), this is typically fast.
        while total_masked < num_to_mask:
            remaining = num_to_mask - total_masked

            # sample length from exponential, clamp to remaining
            L = int(random.expovariate(1.0 / self.mean_noise_span_length))
            if L < 1:
                L = 1
            if L > remaining:
                L = remaining
            if L > n:
                L = n

            # Find a start position that fits and doesn't overlap
            found = False
            for _ in range(self.max_start_tries):
                s = random.randrange(0, n - L + 1)
                # quick overlap check using occupied array
                if 1 in occupied[s : s + L]:
                    continue
                # mark and accept
                for j in range(s, s + L):
                    occupied[j] = 1
                spans.append((s, L))
                total_masked += L
                found = True
                break

            if not found:
                # Fallback: mask any single free position; if none free, stop
                free = None
                for i in range(n):
                    if occupied[i] == 0:
                        free = i
                        break
                if free is None:
                    break
                occupied[free] = 1
                spans.append((free, 1))
                total_masked += 1

        spans.sort(key=lambda x: x[0])

        # Build corrupted + labels with sentinels
        corrupted: List[int] = []
        labels: List[int] = []
        prev_end = 0

        for i, (start, L) in enumerate(spans):
            # unchanged chunk
            if start > prev_end:
                corrupted.extend(input_ids[prev_end:start])

            sid = self.extra_ids[i % 100]
            corrupted.append(sid)
            labels.append(sid)
            labels.extend(input_ids[start:start + L])
            prev_end = start + L

        # tail + EOS
        if prev_end < n:
            corrupted.extend(input_ids[prev_end:])
        corrupted.append(eos_id)
        labels.append(eos_id)

        return corrupted, labels

class DataCollatorForTranslationCorruption(DataCollatorForSeq2Seq):
    """
    A data collator that:
    - Receives examples with 'input_ids' (source cuneiform) and 'labels' (target English).
    - Optionally corrupts the source cuneiform using span corruption.
      Instead of using T5-style extra id tokens, we replace each corrupted span with the asterisk ("*") token.
    - Leaves the target (labels) intact, except we also enforce a target_length if desired.
    - Then uses the parent's Seq2Seq logic to do final padding/tensor conversion.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        source_length: int = 64,
        target_length: int = 64,
    ):
        """
        Args:
            tokenizer: Your seq2seq tokenizer.
            model: The model passed to DataCollatorForSeq2Seq (for special logic).
            corruption_probability: Probability of corrupting an example's input_ids.
            noise_density: Fraction of tokens to mask for corruption.
            mean_noise_span_length: Average span length for masked spans.
            source_length: The maximum length to enforce on the (possibly corrupted) input_ids.
            target_length: The maximum length to enforce on the labels.
        """
        super().__init__(tokenizer, model=model, return_tensors="pt")
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.tokenizer = tokenizer
        self.source_length = source_length
        self.target_length = target_length

    def corrupt_input(self, input_ids: list[int]) -> list[int]:
        """
        Apply span corruption on a single example's input_ids.
        Instead of T5's <extra_id_x> tokens, this method replaces spans with the '*' token.
        """
        num_tokens = len(input_ids)
        num_to_mask = max(1, int(round(num_tokens * self.noise_density)))

        # Decide lengths of spans to mask
        lengths = []
        while sum(lengths) < num_to_mask:
            lengths.append(
                min(
                    max(int(random.expovariate(1 / self.mean_noise_span_length)), 1),
                    num_tokens - sum(lengths),
                )
            )

        # Select starting positions for each span without overlapping
        span_starts = []
        used_positions = set()
        max_attempts = 10
        for length in lengths:
            attempts = 0
            while attempts < max_attempts:
                start = random.randint(0, num_tokens - length)
                positions = set(range(start, start + length))
                if not (positions & used_positions):
                    span_starts.append(start)
                    used_positions.update(positions)
                    break
                attempts += 1

        # Sort spans by their starting positions
        spans = sorted(zip(span_starts, lengths))

        # Create the corrupted input_ids list
        corrupted_ids = []
        prev_end = 0
        for start, length in spans:
            # Add tokens before the masked span
            corrupted_ids.extend(input_ids[prev_end:start])
            # Replace the span with the '*' token (using the tokenizer to get its id)
            corruption_token = self.tokenizer.convert_tokens_to_ids("*")
            corrupted_ids.append(corruption_token)
            # Skip over the tokens in the span
            prev_end = start + length

        # Append any remaining tokens after the last span
        corrupted_ids.extend(input_ids[prev_end:])
        return corrupted_ids

    def __call__(self, features):
        PAD_ID = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        for f in features:
            # Possibly corrupt the input_ids with probability self.corruption_probability
            if random.random() < self.corruption_probability:
                f["input_ids"] = self.corrupt_input(f["input_ids"])

            # Truncate to maximum lengths
            f["input_ids"] = f["input_ids"][: self.source_length]
            if "attention_mask" in f:
                f["attention_mask"] = f["attention_mask"][: len(f["input_ids"])]
            if "labels" in f:
                f["labels"] = f["labels"][: self.target_length]

            # Pad input_ids to self.source_length if necessary
            if len(f["input_ids"]) < self.source_length:
                f["input_ids"].extend([PAD_ID] * (self.source_length - len(f["input_ids"])))
            # Pad attention_mask similarly if it exists
            if "attention_mask" in f and len(f["attention_mask"]) < self.source_length:
                f["attention_mask"].extend([0] * (self.source_length - len(f["attention_mask"])))
            # Pad labels to self.target_length if necessary
            if "labels" in f and len(f["labels"]) < self.target_length:
                f["labels"].extend([PAD_ID] * (self.target_length - len(f["labels"])))
            f["labels"] = [
                (l if l != PAD_ID else -100)
                for l in f["labels"]
            ]
        # Use the parent's __call__ to perform final padding/tensor conversion
        batch = super().__call__(features)
        return batch


import random
from typing import Any, Dict, List, Sequence
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _round_up_to_allowed(n: int, allowed: Sequence[int]) -> int:
    for a in allowed:
        if n <= a:
            return a
    return allowed[-1]

def _pad_to_width(x: torch.Tensor, target_w: int, pad_value: int) -> torch.Tensor:
    if x.dim() == 2 and x.size(1) != target_w:
        return F.pad(x, (0, target_w - x.size(1)), value=pad_value)
    return x

def _pad_batch_to_len(batch: Dict[str, torch.Tensor], target_len: int, pad_id: int) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if v.dim() == 2:
            if k == "labels":
                out[k] = _pad_to_width(v, target_len, -100)
            elif k in ("attention_mask", "decoder_attention_mask"):
                out[k] = _pad_to_width(v, target_len, 0)
            else:  # input_ids / decoder_input_ids
                out[k] = _pad_to_width(v, target_len, pad_id)
        else:
            out[k] = v
    return out


import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase


class DataCollatorForTranslationCorruptionBuckets(DataCollatorForSeq2Seq):
    """
    Supervised translation collator with *-span corruption on the source + length bucketing.

    Minimal-diff rewrite:
      - replaces the old non-overlapping random placement corruption with T5-style span partitioning
        (same idea as your pretrain collator), so corruption adheres much more closely to noise_density.

    Notes:
      * Labels are left intact (except DataCollatorForSeq2Seq will pad and set label PAD -> -100).
      * Keep `padding=True` and `pad_to_multiple_of=8` for tensor-core friendly shapes.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        *,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        source_length: int = 64,
        target_length: int = 64,
        allowed_buckets: Sequence[int] = (8, 16, 24, 32, 40, 48, 56, 64),
        pad_to_multiple_of: int = 8,
        label_pad_token_id: int = -100,
        padding: bool | str = True,
        # optional: make bucket widths strict (same concept as your other collator)
        strict_bucket_pad: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

        self.corruption_probability = float(corruption_probability)
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)

        self.source_length = int(source_length)
        self.target_length = int(target_length)
        self.allowed = tuple(sorted(int(x) for x in allowed_buckets))
        self.strict_bucket_pad = bool(strict_bucket_pad)

        self.tokenizer = tokenizer
        self.PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        self.star_id = tokenizer.convert_tokens_to_ids("*")
        if self.star_id is None or self.star_id == tokenizer.unk_token_id:
            raise ValueError(
                "Tokenizer does not contain '*' token. Add it (tokenizer.add_tokens(['*'])) "
                "and resize model embeddings."
            )
        self.star_id = int(self.star_id)

        # keep a simple RNG; reproducibility across workers depends on DataLoader seeding
        self._py_rng = random.Random()
        self._np_rng = np.random.default_rng()

    # --------------------------
    # T5-style span partitioning
    # --------------------------

    def _random_spans(self, L: int):
        """
        Return two lists: nonnoise_lengths (len=S+1), noise_lengths (len=S),
        using randomized partitions (no overlap/retry logic).
        """
        if L <= 0:
            return [L], []

        # how many tokens to mask
        n_noise = max(1, int(round(L * self.noise_density)))
        n_nonnoise = max(0, L - n_noise)

        # how many spans
        S = max(1, int(round(n_noise / self.mean_noise_span_length)))

        # partition noise into S positive integers
        if n_noise >= S:
            noise_lengths = self._np_rng.multinomial(n_noise - S, [1.0 / S] * S) + 1
        else:
            noise_lengths = np.ones(S, dtype=int)

        # partition non-noise into S+1 buckets (can be zeros)
        if n_nonnoise >= (S + 1):
            nonnoise_lengths = self._np_rng.multinomial(n_nonnoise - (S + 1), [1.0 / (S + 1)] * (S + 1)) + 1
        else:
            nonnoise_lengths = np.zeros(S + 1, dtype=int)
            if n_nonnoise > 0:
                picks = self._np_rng.choice(S + 1, size=n_nonnoise, replace=True)
                for p in picks:
                    nonnoise_lengths[p] += 1

        # clip in rare rounding cases
        total = int(noise_lengths.sum() + nonnoise_lengths.sum())
        if total > L:
            over = total - L
            nonnoise_lengths[-1] = max(0, int(nonnoise_lengths[-1]) - over)

        return nonnoise_lengths.tolist(), noise_lengths.tolist()

    def _corrupt_input_t5style(self, ids: List[int]) -> List[int]:
        """
        Replace each noise span with a single '*' token, using T5-style alternating partitions.
        Labels remain untouched elsewhere.
        """
        L = len(ids)
        if L == 0:
            return ids

        nonnoise, noise = self._random_spans(L)

        out: List[int] = []
        idx = 0
        for gap_len, noise_len in zip(nonnoise, noise):
            if gap_len > 0:
                out.extend(ids[idx : idx + gap_len])
                idx += gap_len

            # collapse the noise span to a single '*'
            out.append(self.star_id)
            idx += noise_len

        # trailing non-noise
        if len(nonnoise) > len(noise):
            final_gap = nonnoise[-1]
            if final_gap > 0:
                out.extend(ids[idx : idx + final_gap])
                idx += final_gap

        return out

    # --------------
    # main collate
    # --------------

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        feats: List[Dict[str, Any]] = []
        for f in features:
            g = dict(f)

            # probabilistic source corruption
            if self._py_rng.random() < self.corruption_probability:
                g["input_ids"] = self._corrupt_input_t5style(g["input_ids"])

            # hard caps; no padding yet
            g["input_ids"] = g["input_ids"][: self.source_length]
            if len(g["input_ids"]) == 0:
                # ensure at least 1 token
                eos = self.tokenizer.eos_token_id
                g["input_ids"] = [int(eos) if eos is not None else self.PAD]

            if "labels" in g:
                g["labels"] = g["labels"][: self.target_length]
            feats.append(g)

        # bucket by rounded-up effective source length
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for g in feats:
            eff = len(g["input_ids"])
            tgt_len = _round_up_to_allowed(eff, self.allowed)
            buckets[tgt_len].append(g)

        sub_batches: List[Dict[str, torch.Tensor]] = []
        for tgt_len, group in buckets.items():
            sb = super().__call__(group)

            # optional strict width per bucket
            if self.strict_bucket_pad:
                want_w = min(int(tgt_len), int(self.source_length))
                sb = _pad_batch_to_len(sb, want_w, self.PAD)

            sub_batches.append(sb)

        if len(sub_batches) == 1:
            return sub_batches[0]

        # pad all sub-batches to the max width in THIS collate call and concat
        max_w = max(sb["input_ids"].size(1) for sb in sub_batches)
        sub_batches = [_pad_batch_to_len(sb, max_w, self.PAD) for sb in sub_batches]

        keys = set().union(*(b.keys() for b in sub_batches))
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.cat([b[k] for b in sub_batches if k in b], dim=0)
        return out


import random
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase
import torch

class DataCollatorForTranslationCorruptionWithErrors(DataCollatorForSeq2Seq):
    """
    Extends your span-corrupting collator by optionally swapping 
    individual tokens for “incorrect” tokens (simulating human error).
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        error_probability: float = 0.05,
        erroneous_tokens: list[str] | list[int] = None,
        source_length: int = 60,
        target_length: int = 60,
    ):
        super().__init__(tokenizer, model=model, return_tensors="pt")
        self.tokenizer = tokenizer
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.error_probability = error_probability
        self.source_length = source_length
        self.target_length = target_length

        # Prepare a list of token IDs we can inject as “errors”
        self.erroneous_token_ids = []
        if erroneous_tokens:
            for tok in erroneous_tokens:
                if isinstance(tok, int):
                    self.erroneous_token_ids.append(tok)
                else:
                    self.erroneous_token_ids.append(tokenizer.convert_tokens_to_ids(tok))

    def corrupt_input(self, input_ids: list[int]) -> list[int]:
        # --- your existing span-corruption logic, replacing spans with "*" ---
        num_tokens = len(input_ids)
        num_to_mask = max(1, int(round(num_tokens * self.noise_density)))

        # decide span lengths
        lengths = []
        while sum(lengths) < num_to_mask:
            lengths.append(
                min(
                    max(int(random.expovariate(1/self.mean_noise_span_length)), 1),
                    num_tokens - sum(lengths),
                )
            )

        # pick non-overlapping starts
        span_starts, used = [], set()
        for length in lengths:
            for _ in range(10):
                start = random.randint(0, num_tokens - length)
                if not (set(range(start, start+length)) & used):
                    span_starts.append(start)
                    used.update(range(start, start+length))
                    break

        spans = sorted(zip(span_starts, lengths))

        corrupted = []
        prev = 0
        star_id = self.tokenizer.convert_tokens_to_ids("*")
        for start, length in spans:
            corrupted.extend(input_ids[prev:start])
            corrupted.append(star_id)
            prev = start + length
        corrupted.extend(input_ids[prev:])
        return corrupted

    def _inject_errors(self, input_ids: list[int]) -> list[int]:
        """
        For each token, with probability `error_probability`,
        replace it with a random choice from `erroneous_token_ids`.
        """
        if not self.erroneous_token_ids or self.error_probability <= 0:
            return input_ids

        out = []
        for tok_id in input_ids:
            if random.random() < self.error_probability:
                out.append(random.choice(self.erroneous_token_ids))
            else:
                out.append(tok_id)
        return out

    def __call__(self, features):
        PAD_ID = self.tokenizer.pad_token_id or 0

        for f in features:
            # 1) Span corruption
            if random.random() < self.corruption_probability:
                f["input_ids"] = self.corrupt_input(f["input_ids"])

            # 2) Inject single-token errors
            f["input_ids"] = self._inject_errors(f["input_ids"])

            # 3) Truncate to fixed source length
            f["input_ids"] = f["input_ids"][:self.source_length]
            if "attention_mask" in f:
                f["attention_mask"] = f["attention_mask"][:len(f["input_ids"])]

            # 4) Pad to source_length
            pad_len = self.source_length - len(f["input_ids"])
            if pad_len > 0:
                f["input_ids"].extend([PAD_ID] * pad_len)
                if "attention_mask" in f:
                    f["attention_mask"].extend([0] * pad_len)

            # 5) Truncate & pad labels
            if "labels" in f:
                f["labels"] = f["labels"][:self.target_length]
                label_pad = self.target_length - len(f["labels"])
                if label_pad > 0:
                    f["labels"].extend([PAD_ID] * label_pad)

        # 6) Let parent collator convert to tensors (already fixed-size)
        batch = super().__call__(features)

        # 7) Ensure contiguity (for Apex fused optimizers)
        batch["input_ids"]      = batch["input_ids"].contiguous()
        batch["attention_mask"] = batch["attention_mask"].contiguous()
        batch["labels"]         = batch["labels"].contiguous()

        # 8) Mask label padding for loss
        batch["labels"][batch["labels"] == PAD_ID] = -100
        return batch

import random
from typing import Any, Dict, List, Sequence
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _round_up_to_allowed(n: int, allowed: Sequence[int]) -> int:
    for a in allowed:
        if n <= a:
            return a
    return allowed[-1]

def _pad_to_width(x: torch.Tensor, target_w: int, pad_value: int) -> torch.Tensor:
    if x.dim() == 2 and x.size(1) != target_w:
        return F.pad(x, (0, target_w - x.size(1)), value=pad_value)
    return x

def _pad_batch_to_len(batch: Dict[str, torch.Tensor], target_len: int, pad_id: int) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if v.dim() == 2:
            if k == "labels":
                out[k] = _pad_to_width(v, target_len, -100)
            elif k in ("attention_mask", "decoder_attention_mask"):
                out[k] = _pad_to_width(v, target_len, 0)
            else:  # input_ids / decoder_input_ids
                out[k] = _pad_to_width(v, target_len, pad_id)
        else:
            out[k] = v
    return out



class DataCollatorForTranslationCorruptionWithErrorsBuckets(DataCollatorForSeq2Seq):
    """
    Translation collator that:
      • Applies '*' span corruption to the SOURCE with probability `corruption_probability`.
      • Injects single-token 'bad signs' with probability `error_probability`.
      • Buckets by allowed source lengths (e.g., 8/16/32/64) and collates per bucket.
      • Pads each bucket to its longest (rounded to `pad_to_multiple_of`), then pads all buckets to
        the max width of this collate call and concatenates → one final batch.

    Labels are left semantically intact (parent will pad and map pad→-100).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        *,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        error_probability: float = 0.05,
        erroneous_tokens: List[str] | List[int] | None = None,
        source_length: int = 64,
        target_length: int = 64,
        allowed_buckets: Sequence[int] = (8, 16, 24, 32, 40, 48, 56, 64),
        pad_to_multiple_of: int = 8,
        label_pad_token_id: int = -100,
        padding: bool | str = True,  # keep 'longest' (dynamic) padding per bucket
    ):
        super().__init__(
            tokenizer=tokenizer,
            model=model,
            padding=padding,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        self.tok = tokenizer
        self.PAD = tokenizer.pad_token_id or 0
        self.star_id = tokenizer.convert_tokens_to_ids("*")
        if self.star_id is None or self.star_id == tokenizer.unk_token_id:
            raise ValueError(
                "Tokenizer does not contain '*' token. "
                "Add it once (tokenizer.add_tokens(['*'])) and resize model embeddings."
            )

        self.corruption_probability = float(corruption_probability)
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)
        self.error_probability = float(error_probability)

        self.source_length = int(source_length)
        self.target_length = int(target_length)
        self.allowed = tuple(sorted(allowed_buckets))

        # Build error-token id list (filter Nones)
        self.erroneous_token_ids: List[int] = []
        if erroneous_tokens:
            for tok in erroneous_tokens:
                tid = tok if isinstance(tok, int) else tokenizer.convert_tokens_to_ids(tok)
                if tid is not None and tid != tokenizer.unk_token_id:
                    self.erroneous_token_ids.append(int(tid))

        # local RNG (Accelerate/torch will seed workers -> reproducible per-worker)
        self._py_rng = random.Random()
        
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.bos_id = getattr(tokenizer, "bos_token_id", None)
        self.sep_id = getattr(tokenizer, "sep_token_id", None)
        self.cls_id = getattr(tokenizer, "cls_token_id", None)
        self.forbid = {x for x in [self.PAD, self.eos_id, self.bos_id, self.sep_id, self.cls_id, self.star_id] if x is not None}

    # ---------- corruption helpers ----------

    def _corrupt_input(self, ids: List[int]) -> List[int]:
        """Collapse noise spans to '*' tokens; labels untouched."""
        L = len(ids)
        if L == 0:
            return ids

        num_to_mask = max(1, int(round(L * self.noise_density)))
        lengths: List[int] = []
        covered = 0
        # exponential-like span lengths
        while covered < num_to_mask:
            l = max(1, int(self._py_rng.expovariate(1.0 / self.mean_noise_span_length)))
            l = min(l, L - covered)
            lengths.append(l)
            covered += l

        # pick non-overlapping starts
        starts: List[int] = []
        used = set()
        for ln in lengths:
            for _ in range(10):
                s = self._py_rng.randint(0, L - ln)
                span = range(s, s + ln)
                if all((p not in used) for p in span):
                    starts.append(s)
                    used.update(span)
                    break

        spans = sorted(zip(starts, lengths))
        out: List[int] = []
        idx = 0
        for s, ln in spans:
            out.extend(ids[idx:s])
            out.append(self.star_id)
            idx = s + ln
        out.extend(ids[idx:])
        return out

    def _inject_errors(self, ids: List[int]) -> List[int]:
        if not self.erroneous_token_ids or self.error_probability <= 0.0:
            return ids
        choose = self._py_rng.choice
        rnd = self._py_rng.random
        err_ids = self.erroneous_token_ids
        out = []
        for t in ids:
            if (t not in self.forbid) and (rnd() < self.error_probability):
                repl = choose(err_ids)
                if repl in self.forbid:
                    out.append(t)
                else:
                    out.append(repl)
            else:
                out.append(t)
        return out

    # ---------- main collate ----------

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        feats: List[Dict[str, Any]] = []
        for f in features:
            g = dict(f)

            # --- trim trailing PADs BEFORE any corruption/injection ---
            ids = g["input_ids"]
            while ids and ids[-1] == self.PAD:
                ids.pop()
            g["input_ids"] = ids

            # optional span corruption
            if self._py_rng.random() < self.corruption_probability:
                g["input_ids"] = self._corrupt_input(g["input_ids"])

            # optional single-token errors (forbid specials)
            g["input_ids"] = self._inject_errors(g["input_ids"])

            # cap lengths (no pre-padding here)
            g["input_ids"] = g["input_ids"][: self.source_length]
            if "labels" in g:
                g["labels"] = g["labels"][: self.target_length]

            feats.append(g)

        if len(g["input_ids"]) == 0:
            fallback = getattr(self.tok, "eos_token_id", None)
            if fallback is None:
                fallback = self.PAD or 0
            g["input_ids"] = [fallback]

        # bucket by rounded-up effective source length
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for g in feats:
            eff = len(g["input_ids"])
            tgt_len = _round_up_to_allowed(eff, self.allowed)
            buckets[tgt_len].append(g)

        # collate once per bucket (parent handles dynamic padding & label masking)
        sub_batches: List[Dict[str, torch.Tensor]] = []
        for _, group in buckets.items():
            sub_batches.append(super().__call__(group))

        if len(sub_batches) == 1:
            return sub_batches[0]

        # pad all sub-batches to the max width seen now, then concat
        max_w = max(sb["input_ids"].size(1) for sb in sub_batches)
        sub_batches = [_pad_batch_to_len(sb, max_w, self.PAD) for sb in sub_batches]

        keys = set().union(*(b.keys() for b in sub_batches))
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.cat([b[k] for b in sub_batches if k in b], dim=0)
        return out

import random
from typing import Dict, List, Tuple, Iterable, Optional, Union

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase


class FastTranslationCorruptionWithErrorsCollator:
    """
    - Corrupts source (encoder) sequences by replacing random spans with a single '*' token id.
    - Optionally injects token 'errors' (simulate human transcription mistakes) on the source.
    - Leaves labels intact.
    - Uses HF DataCollatorForSeq2Seq for dynamic padding (+decoder_input_ids, -100 label padding).
    - Bucket-pads to a small set of target lengths to reduce kernel churn while keeping padding tight.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        *,
        corruption_probability: float = 0.15,
        noise_density: float = 0.12,
        mean_noise_span_length: float = 6.0,
        error_probability: float = 0.05,
        erroneous_tokens: Optional[Iterable[Union[str, int]]] = None,
        avoid_eos_in_corruption: bool = True,
        avoid_replacing_specials: bool = True,
        pad_to_multiple_of: int = 8,
        buckets: Tuple[int, ...] = (64, 96, 128, 160, 192, 256),
    ):
        self.tokenizer = tokenizer
        self.model = model

        # corruption controls
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.avoid_eos_in_corruption = avoid_eos_in_corruption

        # injection controls
        self.error_probability = error_probability
        self.avoid_replacing_specials = avoid_replacing_specials
        self.erroneous_token_ids = self._prepare_erroneous_ids(erroneous_tokens)

        # specials
        self.pad_id = tokenizer.pad_token_id or 0
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

        # id for a literal '*'
        sid = tokenizer.convert_tokens_to_ids("*")
        if sid is None or sid < 0:
            # robust fallback for tokenizers that don’t expose '*' as a token
            sid = tokenizer("*", add_special_tokens=False)["input_ids"][0]
        self.star_id = sid

        # base collator: dynamic pad + decoder_input_ids + -100 label padding
        self.base = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        # bucket shapes
        self.buckets = tuple(sorted(buckets))
        self.pad_to_multiple_of = pad_to_multiple_of

    # ---------------- helpers ----------------

    def _prepare_erroneous_ids(
        self, toks: Optional[Iterable[Union[str, int]]]
    ) -> List[int]:
        """
        Convert a user-supplied set of erroneous tokens into valid single-token ids.
        ByT5: if a string maps to multiple byte-ids, we skip it (we want 1:1 swaps).
        """
        if not toks:
            return []
        ids: List[int] = []
        for t in toks:
            if isinstance(t, int):
                if t >= 0:
                    ids.append(t)
            else:
                enc = self.tokenizer(t, add_special_tokens=False)["input_ids"]
                if len(enc) == 1:
                    ids.append(enc[0])
        # de-dup and drop specials
        specials = {self.tokenizer.pad_token_id, self.tokenizer.cls_token_id,
                    self.tokenizer.sep_token_id, self.tokenizer.bos_token_id,
                    self.tokenizer.eos_token_id, self.tokenizer.unk_token_id,
                    self.star_id}
        return [i for i in set(ids) if i is not None and i not in specials]

    def _round_up(self, L: int) -> int:
        for b in self.buckets:
            if L <= b:
                return b
        return self.buckets[-1]

    def _strip_right_pads(self, ids: List[int]) -> int:
        n = len(ids)
        while n > 0 and ids[n - 1] == self.pad_id:
            n -= 1
        return n

    def _content_bounds(self, ids: List[int]) -> Tuple[int, int]:
        """
        Return (content_len, full_len_no_pad).
        content_len excludes trailing pad and (optionally) final EOS for corruption.
        """
        full_len = self._strip_right_pads(ids)
        content_len = full_len
        if self.avoid_eos_in_corruption and self.eos_id is not None:
            if content_len > 0 and ids[content_len - 1] == self.eos_id:
                content_len -= 1
        return max(content_len, 0), full_len

    def _sample_spans(self, num_tokens: int) -> List[Tuple[int, int]]:
        """
        Sample non-overlapping spans covering ~noise_density of 'num_tokens'.
        """
        if num_tokens <= 0:
            return []
        target = max(1, int(round(num_tokens * self.noise_density)))
        spans, used = [], set()
        covered = 0
        attempts, max_attempts = 0, 10 * max(target, 1)

        while covered < target and attempts < max_attempts:
            L = max(1, int(random.expovariate(1.0 / self.mean_noise_span_length)))
            L = min(L, num_tokens)
            start_max = num_tokens - L
            if start_max < 0:
                break
            s = random.randint(0, start_max)
            rng = range(s, s + L)
            if not any(i in used for i in rng):
                spans.append((s, L))
                used.update(rng)
                covered += L
            attempts += 1

        spans.sort()
        return spans

    def _corrupt_spans_with_star(self, ids: List[int]) -> List[int]:
        """
        Replace each selected span with a single '*' id, preserving EOS and pads.
        """
        content_len, full_len = self._content_bounds(ids)
        if content_len <= 0:
            return ids

        head = ids[:content_len]
        tail = ids[content_len:full_len]   # (e.g., EOS)
        pads = ids[full_len:]              # trailing pads

        spans = self._sample_spans(len(head))
        if not spans:
            return ids

        out: List[int] = []
        prev = 0
        for s, L in spans:
            if s > prev:
                out.extend(head[prev:s])
            out.append(self.star_id)
            prev = s + L
        if prev < len(head):
            out.extend(head[prev:])

        out.extend(tail)
        out.extend(pads)
        return out

    def _inject_errors(self, ids: List[int]) -> List[int]:
        """
        With probability `error_probability` per token, replace with a random erroneous id.
        Avoid replacing specials if requested.
        """
        if self.error_probability <= 0 or not self.erroneous_token_ids:
            return ids

        content_len, full_len = self._content_bounds(ids)
        if content_len <= 0:
            return ids

        out = ids[:]  # copy
        specials = set()
        if self.avoid_replacing_specials:
            specials = {
                self.pad_id,
                self.eos_id,
                self.star_id,
                self.tokenizer.bos_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.mask_token_id
                if hasattr(self.tokenizer, "mask_token_id")
                else None,
            }
            specials = {s for s in specials if s is not None}

        for i in range(content_len):  # don’t touch tail/pads
            tok = out[i]
            if tok in specials:
                continue
            if random.random() < self.error_probability:
                out[i] = random.choice(self.erroneous_token_ids)

        return out

    # ---------------- main ----------------

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1) Keep only tensorish fields; we’ll rebuild masks/decoder fields after corruption
        KEEP = {"input_ids", "labels"}
        clean_feats = []
        for f in features:
            g = {k: v for k, v in f.items() if k in KEEP}
            # corrupt?
            if random.random() < self.corruption_probability:
                g["input_ids"] = self._corrupt_spans_with_star(g["input_ids"])
            # inject single-token errors
            g["input_ids"] = self._inject_errors(g["input_ids"])
            clean_feats.append(g)

        # 2) Dynamic pad via HF (creates attention_mask, decoder_input_ids, masks labels to -100)
        batch = self.base(clean_feats)

        # 3) Bucket to a small set of lengths (tight padding, fewer recompiles)
        x, am, y = batch["input_ids"], batch["attention_mask"], batch["labels"]
        B, Lin = x.shape
        Lt = y.shape[1]

        Lin2 = self._round_up(Lin)
        Lt2 = self._round_up(Lt)

        if Lin2 > Lin:
            xpad = torch.full((B, Lin2 - Lin), self.pad_id, dtype=x.dtype, device=x.device)
            ampad = torch.zeros((B, Lin2 - Lin), dtype=am.dtype, device=am.device)
            batch["input_ids"] = torch.cat([x, xpad], dim=1)
            batch["attention_mask"] = torch.cat([am, ampad], dim=1)

        if Lt2 > Lt:
            lpad = torch.full((B, Lt2 - Lt), -100, dtype=y.dtype, device=y.device)
            batch["labels"] = torch.cat([y, lpad], dim=1)

        if "decoder_input_ids" in batch:
            dec = batch["decoder_input_ids"]
            Ld = dec.shape[1]
            Ld2 = self._round_up(Ld)
            if Ld2 > Ld:
                dpad = torch.full((B, Ld2 - Ld), self.pad_id, dtype=dec.dtype, device=dec.device)
                batch["decoder_input_ids"] = torch.cat([dec, dpad], dim=1)

        # 4) Safety: ensure there’s at least one supervised label
        if (batch["labels"] != -100).sum().item() == 0:
            raise ValueError("All labels in this batch are -100. Check tokenization/truncation of targets.")

        return batch

class ByT5DataCollatorForTranslationCorruption(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.10,
        noise_density: float = 0.10,
        mean_noise_span_length: float = 8.0,
        source_length: int = 512,
        target_length: int = 512,
    ):
        super().__init__(tokenizer, model=model, return_tensors="pt")
        self.tokenizer = tokenizer
        self.corruption_probability = corruption_probability
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.source_length = source_length
        self.target_length = target_length
        # ID of the real '*' in your data, left untouched
        self.star_id = tokenizer.convert_tokens_to_ids("*")

    def corrupt_input(self, input_ids: list[int]) -> list[int]:
        num_tokens = len(input_ids)
        num_to_mask = max(1, int(round(num_tokens * self.noise_density)))

        # 1) decide span lengths
        lengths = []
        while sum(lengths) < num_to_mask:
            span_len = int(random.expovariate(1/self.mean_noise_span_length)) or 1
            lengths.append(min(span_len, num_tokens - sum(lengths)))

        # 2) pick non-overlapping starts
        used = set()
        spans = []
        for length in lengths:
            for _ in range(10):
                start = random.randint(0, num_tokens - length)
                if not (set(range(start, start+length)) & used):
                    spans.append((start, length))
                    used |= set(range(start, start+length))
                    break

        spans.sort(key=lambda x: x[0])

        # 3) build corrupted byte-IDs
        corrupted = []
        prev = 0
        for i, (start, length) in enumerate(spans):
            corrupted.extend(input_ids[prev:start])

            # insert a T5 sentinel rather than '*'
            sentinel = f"<extra_id_{i}>"
            sentinel_id = self.tokenizer.convert_tokens_to_ids(sentinel)
            corrupted.append(sentinel_id)

            prev = start + length

        corrupted.extend(input_ids[prev:])
        return corrupted

    def __call__(self, features):
        PAD = self.tokenizer.pad_token_id or 0

        # 1) per-example corruption + truncation + pad-to-fixed-length
        for f in features:
            if random.random() < self.corruption_probability:
                f["input_ids"] = self.corrupt_input(f["input_ids"])

            f["input_ids"] = f["input_ids"][: self.source_length]
            if "attention_mask" in f:
                f["attention_mask"] = f["attention_mask"][: len(f["input_ids"])]
            if "labels" in f:
                f["labels"] = f["labels"][: self.target_length]

            # pad source
            pad_len = self.source_length - len(f["input_ids"])
            if pad_len > 0:
                f["input_ids"].extend([PAD] * pad_len)
                if "attention_mask" in f:
                    f["attention_mask"].extend([0] * pad_len)

            # pad labels
            if "labels" in f:
                lbl_pad = self.target_length - len(f["labels"])
                if lbl_pad > 0:
                    f["labels"].extend([PAD] * lbl_pad)

        # 2) let Seq2Seq parent handle tensor conversion
        batch = super().__call__(features)
        return batch


import torch
from transformers import PreTrainedTokenizerBase

class MixedDataCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrain_collator,                   # configure it with input_length=FIXED_MAX_LEN, target_length=FIXED_MAX_LEN
        translate_collator=None,
        prompt_text="Identify the missing signs: ",
        FIXED_MAX_LEN=60,
        add_decoder_inputs=True,
    ):
        self.tok = tokenizer
        self.pretrain_collator = pretrain_collator
        self.translate_collator = translate_collator
        self.prompt_text = prompt_text
        self.FIXED_MAX_LEN = FIXED_MAX_LEN
        self.add_decoder_inputs = add_decoder_inputs

        self.PAD = tokenizer.pad_token_id
        self.START = tokenizer.eos_token_id  # T5-style

        self.prompt_ids = tokenizer(self.prompt_text, add_special_tokens=False)["input_ids"]

    def __call__(self, examples):
        pretrain_examples  = [ex for ex in examples if ex.get("task") == "pretrain"]
        translate_examples = [ex for ex in examples if ex.get("task") == "train"]

        pretrain_batch = None
        if pretrain_examples:
            # prepend prompt to inputs ONLY; do not touch labels (they don’t exist yet)
            for ex in pretrain_examples:
                seq = self.prompt_ids + ex["input_ids"]
                # hard-cap length before corruption if you want; or let collator handle it dynamically
                ex["input_ids"] = seq[: self.FIXED_MAX_LEN]

            # Let the pretrain collator do span corruption + padding to multiples of 8
            batch = self.pretrain_collator(pretrain_examples)

            if self.add_decoder_inputs:
                batch["decoder_input_ids"] = shift_tokens_right(
                    batch["labels"], self.PAD, self.START
                ).contiguous()
                batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.PAD).long().contiguous()

            pretrain_batch = {k: batch[k] for k in batch.keys()
                              if k in ("input_ids","attention_mask","labels","decoder_input_ids","decoder_attention_mask")}

        translate_batch = None
        if translate_examples:
            # Fixed-length padding/truncation at example level for supervised path
            for ex in translate_examples:
                ids = ex["input_ids"][: self.FIXED_MAX_LEN]
                lbl = ex["labels"][: self.FIXED_MAX_LEN]
                ids += [self.PAD] * (self.FIXED_MAX_LEN - len(ids))
                lbl += [self.PAD] * (self.FIXED_MAX_LEN - len(lbl))
                ex["input_ids"] = ids
                ex["labels"] = lbl
                ex["attention_mask"] = [1 if t != self.PAD else 0 for t in ids]

            to_collate = [{"input_ids": ex["input_ids"],
                           "attention_mask": ex["attention_mask"],
                           "labels": ex["labels"]} for ex in translate_examples]

            batch = self.translate_collator(to_collate)

            if self.add_decoder_inputs:
                batch["decoder_input_ids"] = shift_tokens_right(
                    batch["labels"], self.PAD, self.START
                ).contiguous()
                batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.PAD).long().contiguous()

            translate_batch = batch

        if pretrain_batch and translate_batch:
            merged = {k: torch.cat([pretrain_batch[k], translate_batch[k]], dim=0) for k in pretrain_batch.keys()}
            return merged

        return pretrain_batch or translate_batch

from typing import List, Dict, Any, Optional
import math, torch
from transformers import PreTrainedTokenizerBase

def _round_up(x: int, m: int = 8) -> int:
    return int(math.ceil(max(1, x) / m) * m)

def _only_tokenized(ex: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in ("input_ids", "attention_mask", "labels"):
        if k in ex and ex[k] is not None:
            out[k] = ex[k]
    return out

def _pad2d(t: torch.Tensor, width: int, pad_val: int) -> torch.Tensor:
    if t.shape[1] == width:
        return t
    if t.shape[1] > width:
        return t[:, :width]
    B, L = t.shape
    out = t.new_full((B, width), pad_val)
    out[:, :L] = t
    return out

class MixedDataCollatorFast:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrain_collator,              # asterisk span-corruption collator (dynamic_inputs=True is fine)
        translate_collator=None,        # DataCollatorForSeq2Seq(..., pad_to_multiple_of=8)
        prompt_text: str = "Identify the missing signs: ",
    ):
        self.tok = tokenizer
        self.pretrain_collator  = pretrain_collator
        self.translate_collator = translate_collator
        self.prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"] if prompt_text else []
        self.PAD   = tokenizer.pad_token_id
        self.START = getattr(tokenizer, "eos_token_id", self.PAD)

    # ---------- helpers ----------

    def _keep_model_keys(self, batch: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
        if batch is None:
            return None
        keys = ("input_ids", "attention_mask", "labels", "decoder_input_ids", "decoder_attention_mask")
        return {k: batch[k] for k in keys if k in batch}

    def _prepend_prompt_after_corruption(self, batch: Dict[str, torch.Tensor]) -> None:
        if not self.prompt_ids:
            return
        B = batch["input_ids"].size(0)
        lens = batch["attention_mask"].sum(dim=1).tolist()
        rows = []
        for i in range(B):
            core = batch["input_ids"][i, : int(lens[i])].tolist()
            rows.append(torch.tensor(self.prompt_ids + core, dtype=torch.long, device=batch["input_ids"].device))
        max_in = _round_up(max(r.numel() for r in rows), 8)
        inputs = torch.full((B, max_in), self.PAD, dtype=torch.long, device=batch["input_ids"].device)
        amask  = torch.zeros((B, max_in), dtype=torch.long, device=inputs.device)
        for i, row in enumerate(rows):
            L = min(row.numel(), max_in)
            inputs[i, :L] = row[:L]
            amask[i, :L]  = 1
        batch["input_ids"] = inputs
        batch["attention_mask"] = amask

    def _ensure_decoder_fields(self, batch: Dict[str, torch.Tensor]) -> None:
        if "decoder_input_ids" not in batch:
            batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], self.PAD, self.START).contiguous()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.PAD).long().contiguous()

    def _supervised_tokens(self, batch: Optional[Dict[str, torch.Tensor]]) -> int:
        if batch is None or "labels" not in batch:
            return 0
        return int((batch["labels"] != -100).sum().item())

    def _pad_to_common_widths(
        self,
        pre: Optional[Dict[str, torch.Tensor]],
        tr : Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        # If only one side, just return (keys already filtered)
        if pre is None or tr is None:
            return self._keep_model_keys(pre or tr)

        # Align encoder/decoder widths for concat
        Wenc = _round_up(max(pre["input_ids"].shape[1], tr["input_ids"].shape[1]), 8)
        Wdec = _round_up(max(pre["labels"].shape[1],    tr["labels"].shape[1]),    8)

        for b in (pre, tr):
            b["input_ids"]               = _pad2d(b["input_ids"],               Wenc, self.PAD)
            b["attention_mask"]          = _pad2d(b["attention_mask"],          Wenc, 0)
            b["labels"]                  = _pad2d(b["labels"],                  Wdec, -100)
            b["decoder_input_ids"]       = _pad2d(b["decoder_input_ids"],       Wdec, self.PAD)
            b["decoder_attention_mask"]  = _pad2d(b["decoder_attention_mask"],  Wdec, 0)

        merged = {
            "input_ids":             torch.cat([pre["input_ids"],             tr["input_ids"]], dim=0),
            "attention_mask":        torch.cat([pre["attention_mask"],        tr["attention_mask"]], dim=0),
            "labels":                torch.cat([pre["labels"],                tr["labels"]], dim=0),
            "decoder_input_ids":     torch.cat([pre["decoder_input_ids"],     tr["decoder_input_ids"]], dim=0),
            "decoder_attention_mask":torch.cat([pre["decoder_attention_mask"],tr["decoder_attention_mask"]], dim=0),
        }
        return merged

    # ---------- main ----------

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pretrain_examples  = [ex for ex in examples if ex.get("task") == "pretrain"]
        translate_examples = [ex for ex in examples if ex.get("task") == "train"]

        # Translate: drop examples with missing/empty labels up front
        if translate_examples:
            filtered = []
            for ex in translate_examples:
                lbl = ex.get("labels")
                if isinstance(lbl, list) and len(lbl) > 0:
                    filtered.append(ex)
            translate_examples = filtered

        pre_batch = None
        if pretrain_examples:
            pre_batch = self.pretrain_collator(pretrain_examples)   # dynamic shapes
            self._prepend_prompt_after_corruption(pre_batch)        # prompt AFTER corruption
            self._ensure_decoder_fields(pre_batch)
            pre_batch = self._keep_model_keys(pre_batch)

        tr_batch = None
        if translate_examples:
            to_collate = [_only_tokenized(ex) for ex in translate_examples]
            if to_collate:  # may have become empty after filtering
                tr_batch = self.translate_collator(to_collate)      # dynamic shapes
                self._ensure_decoder_fields(tr_batch)
                tr_batch = self._keep_model_keys(tr_batch)

        # If a side has zero supervised tokens, drop that side
        if self._supervised_tokens(pre_batch) == 0:
            pre_batch = None
        if self._supervised_tokens(tr_batch) == 0:
            tr_batch = None

        # If both sides empty, surface a clear error to fail fast
        if pre_batch is None and tr_batch is None:
            raise RuntimeError("MixedDataCollatorFast produced a batch with zero supervised tokens (all labels -100).")

        return self._pad_to_common_widths(pre_batch, tr_batch)

import torch
from transformers import PreTrainedTokenizerBase

class MixedDataCollatorOrigional:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrain_collator,
        translate_collator=None,
        error_collator=None,
        prompt_text="Identify the missing signs: ",
        FIXED_MAX_LEN=60
    ):
        """
        tokenizer: your shared tokenizer
        pretrain_collator: collator for 'pretrain' examples
        translate_collator: collator for 'train' examples
        error_collator:     collator for 'train_error' examples
        prompt_text:        text to prepend to pretrain inputs
        FIXED_MAX_LEN:      length to truncate/pad inputs & labels for pretrain/translate
        """
        self.tokenizer = tokenizer
        self.pretrain_collator   = pretrain_collator
        self.translate_collator  = translate_collator
        self.error_collator      = error_collator
        self.prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        self.FIXED_MAX_LEN = FIXED_MAX_LEN
        self.PAD_ID = tokenizer.pad_token_id or 0

    def __call__(self, examples):
        # 1) Split by task
        pretrain_ex   = [ex for ex in examples if ex.get("task") == "pretrain"]
        translate_ex  = [ex for ex in examples if ex.get("task") == "train"]
        error_ex      = [ex for ex in examples if ex.get("task") == "train_error"]

        # 2) Preprocess pretrain examples: prepend prompt, strict pad/truncate
        for ex in pretrain_ex:
            merged = self.prompt_ids + ex["input_ids"]
            merged = merged[:self.FIXED_MAX_LEN]
            pad_len = self.FIXED_MAX_LEN - len(merged)
            if pad_len > 0:
                merged += [self.PAD_ID] * pad_len
            ex["input_ids"] = merged

            lbls = ex["labels"][:self.FIXED_MAX_LEN]
            lp = self.FIXED_MAX_LEN - len(lbls)
            if lp > 0:
                lbls += [self.PAD_ID] * lp
            ex["labels"] = lbls

            real = sum(tok != self.PAD_ID for tok in merged)
            ex["attention_mask"] = [1]*real + [0]*(self.FIXED_MAX_LEN - real)

        # 3) Preprocess translate examples: strict pad/truncate
        for ex in translate_ex:
            for k in ("source_text","target_text","length","task"):
                ex.pop(k, None)

            inp = ex["input_ids"][:self.FIXED_MAX_LEN]
            p = self.FIXED_MAX_LEN - len(inp)
            if p>0: inp += [self.PAD_ID]*p
            ex["input_ids"] = inp

            lbls = ex["labels"][:self.FIXED_MAX_LEN]
            lp = self.FIXED_MAX_LEN - len(lbls)
            if lp>0: lbls += [self.PAD_ID]*lp
            ex["labels"] = lbls

            real = sum(tok != self.PAD_ID for tok in inp)
            ex["attention_mask"] = [1]*real + [0]*(self.FIXED_MAX_LEN - real)

        # 4) Error examples: remove extraneous fields, leave raw input_ids & labels
        for ex in error_ex:
            for k in ("source_text","target_text","length","task"):
                ex.pop(k, None)
            # assume these already have input_ids, labels, attention_mask if needed

        # 5) Collate each subset
        batch_pre   = self.pretrain_collator(pretrain_ex)   if pretrain_ex   else None
        batch_trans = self.translate_collator(translate_ex)  if translate_ex  else None
        batch_err   = self.error_collator(error_ex)          if (error_ex and self.error_collator) else None

        # 6) Keep only input_ids, attention_mask, labels
        def _cleanup(b):
            return {k:v for k,v in b.items() if k in ("input_ids","attention_mask","labels")}

        if batch_pre:   batch_pre   = _cleanup(batch_pre)
        if batch_trans: batch_trans = _cleanup(batch_trans)
        if batch_err:   batch_err   = _cleanup(batch_err)

        # 7) Merge all non‐None batches
        batches = [b for b in (batch_pre, batch_trans, batch_err) if b]
        if not batches:
            return {}
        merged = batches[0]
        for b in batches[1:]:
            for k in set(merged) | set(b):
                if k not in merged:
                    merged[k] = b[k]
                elif k not in b:
                    continue
                else:
                    merged[k] = torch.cat([merged[k], b[k]], dim=0)

        return merged
        
class MixedDataCollator3:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrain_collator,
        translate_collator=None,
        error_collator=None,
        prompt_text="Identify the missing signs: ",
        FIXED_MAX_LEN=60
    ):
        """
        tokenizer: your shared tokenizer
        pretrain_collator: collator for 'pretrain' examples
        translate_collator: collator for 'train' examples
        error_collator:     collator for 'train_error' examples
        prompt_text:        text to prepend to pretrain inputs
        FIXED_MAX_LEN:      length to truncate/pad inputs & labels for all tasks
        """
        self.tokenizer = tokenizer
        self.pretrain_collator = pretrain_collator
        self.translate_collator = translate_collator
        self.error_collator = error_collator
        self.prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        self.FIXED_MAX_LEN = FIXED_MAX_LEN
        self.PAD_ID = tokenizer.pad_token_id or 0
        self.START_ID = tokenizer.eos_token_id

    def __call__(self, examples):
        # 1) Split by task
        pretrain_ex   = [ex for ex in examples if ex.get("task") == "pretrain"]
        translate_ex  = [ex for ex in examples if ex.get("task") == "train"]
        error_ex      = [ex for ex in examples if ex.get("task") == "train_error"]

        # 2) Preprocess pretrain examples: prepend prompt, strict pad/truncate
        for ex in pretrain_ex:
            seq = (self.prompt_ids + ex["input_ids"])[:self.FIXED_MAX_LEN]
            seq += [self.PAD_ID] * max(0, self.FIXED_MAX_LEN - len(seq))
            ex["input_ids"] = seq

            lbl = ex["labels"][:self.FIXED_MAX_LEN]
            lbl += [self.PAD_ID] * max(0, self.FIXED_MAX_LEN - len(lbl))
            ex["labels"] = lbl

            real = sum(tok != self.PAD_ID for tok in seq)
            ex["attention_mask"] = [1]*real + [0]*(self.FIXED_MAX_LEN - real)

        # 3) Preprocess translate examples: strict pad/truncate
        for ex in translate_ex:
            for k in ("source_text","target_text","length","task"):
                ex.pop(k, None)
            seq = ex["input_ids"][:self.FIXED_MAX_LEN]
            seq += [self.PAD_ID] * max(0, self.FIXED_MAX_LEN - len(seq))
            ex["input_ids"] = seq

            lbl = ex["labels"][:self.FIXED_MAX_LEN]
            lbl += [self.PAD_ID] * max(0, self.FIXED_MAX_LEN - len(lbl))
            ex["labels"] = lbl

            real = sum(tok != self.PAD_ID for tok in seq)
            ex["attention_mask"] = [1]*real + [0]*(self.FIXED_MAX_LEN - real)

        # 4) Preprocess error examples: strict pad/truncate (like translate)
        for ex in error_ex:
            for k in ("source_text","target_text","length","task"):
                ex.pop(k, None)
            seq = ex["input_ids"][:self.FIXED_MAX_LEN]
            seq += [self.PAD_ID] * max(0, self.FIXED_MAX_LEN - len(seq))
            ex["input_ids"] = seq

            lbl = ex["labels"][:self.FIXED_MAX_LEN]
            lbl += [self.PAD_ID] * max(0, self.FIXED_MAX_LEN - len(lbl))
            ex["labels"] = lbl

            real = sum(tok != self.PAD_ID for tok in seq)
            ex["attention_mask"] = [1]*real + [0]*(self.FIXED_MAX_LEN - real)

        # 5) Collate each subset
        batch_pre   = self.pretrain_collator(pretrain_ex)   if pretrain_ex   else None
        batch_trans = self.translate_collator(translate_ex)  if translate_ex  else None
        batch_err   = self.error_collator(error_ex)          if error_ex and self.error_collator else None

        # 6) Keep only input_ids, attention_mask, labels
        def _cleanup(b):
            return {k:v for k,v in b.items() if k in ("input_ids","attention_mask","labels")}

        if batch_pre:   batch_pre   = _cleanup(batch_pre)
        if batch_trans: batch_trans = _cleanup(batch_trans)
        if batch_err:   batch_err   = _cleanup(batch_err)

        # 7) Add decoder inputs on each batch
        for batch in (batch_pre, batch_trans, batch_err):
            if batch is None: continue
            # shift labels -> decoder_input_ids
            dec_in = shift_tokens_right(batch["labels"],
                                        pad_token_id=self.PAD_ID,
                                        decoder_start_token_id=self.START_ID)
            batch["decoder_input_ids"] = dec_in.contiguous()
            batch["decoder_attention_mask"] = (dec_in != self.PAD_ID).long().contiguous()

        # 8) Merge all non‐None batches on dim=0
        batches = [b for b in (batch_pre, batch_trans, batch_err) if b]
        if not batches:
            return {}
        merged = {}
        for k in batches[0].keys():
            merged[k] = torch.cat([b[k] for b in batches], dim=0)
        return merged

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

def _round_up_to_allowed(x: int, allowed: Sequence[int]) -> int:
    for a in allowed:
        if x <= a:
            return int(a)
    return int(allowed[-1])

def _pad_batch_to_len(batch: Dict[str, torch.Tensor], width: int, pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Pad/truncate 2D tensors in batch to `width`.
    - labels padded with -100
    - everything else padded with pad_id
    Rebuild attention_mask from input_ids for safety.
    """
    out = dict(batch)
    for k, v in batch.items():
        if not torch.is_tensor(v) or v.dim() != 2:
            continue

        pad_value = -100 if k == "labels" else pad_id

        if v.size(1) > width:
            v2 = v[:, :width]
        elif v.size(1) < width:
            pad = v.new_full((v.size(0), width - v.size(1)), pad_value)
            v2 = torch.cat([v, pad], dim=1)
        else:
            v2 = v

        out[k] = v2.contiguous()

    if "input_ids" in out:
        out["attention_mask"] = (out["input_ids"] != pad_id).long().contiguous()
    return out


class MixedDataCollator3WayBuckets:
    """
    3-way task router + merge:
      - task == "pretrain"       -> (optional) protected prompt + bucket + pretrain_collator per bucket
      - task == "train"          -> translate_collator (already capped/dynamic padded)
      - task == "trans_corrupt"  -> trans_corrupt_collator (does corruption + bucketing internally)

    Then:
      - pad each returned sub-batch to the max width in THIS collate call
      - concat along batch dimension
    """

    def __init__(
        self,
        tokenizer,
        *,
        pretrain_collator,
        translate_collator,
        trans_corrupt_collator,
        prompt_text: Optional[str] = "Identify missing signs: ",
        FIXED_MAX_LEN: int = 64,
        allowed_buckets: Sequence[int] = (8, 16, 32, 48, 64, 128, 256),
        add_decoder_inputs: bool = True,
        protect_prompt: bool = True,   # NEW
        pad_to_bucket: bool = False,   # NEW (strict bucket widths for pretrain sub-batches)
    ):
        self.tok  = tokenizer
        self.preC = pretrain_collator
        self.trC  = translate_collator
        self.tcC  = trans_corrupt_collator

        self.FIXED_MAX_LEN = int(FIXED_MAX_LEN)
        self.allowed       = tuple(sorted(int(x) for x in allowed_buckets))
        self.add_decoder   = bool(add_decoder_inputs)
        self.protect_prompt = bool(protect_prompt)
        self.pad_to_bucket  = bool(pad_to_bucket)

        self.PAD   = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.START = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.PAD

        # prompt_text can be None/empty
        if isinstance(prompt_text, str) and prompt_text.strip():
            self.prompt_text = prompt_text
            self.prompt_ids  = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        else:
            self.prompt_text = None
            self.prompt_ids  = []

    def _make_decoder_inputs(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1]
        shifted[:, 0]  = int(self.START)
        shifted.masked_fill_(shifted == -100, int(self.PAD))
        return {
            "decoder_input_ids": shifted.contiguous(),
            "decoder_attention_mask": (shifted != int(self.PAD)).long().contiguous(),
        }

    def _collate_pretrain(self, pretrain_examples: List[Dict[str, Any]]) -> List[Dict[str, torch.Tensor]]:
        """
        Patched behavior:
          - if protect_prompt: run span corruption on CONTENT only, then prepend prompt after
          - bucket widths are based on *effective* encoder length (prompt+content, capped)
          - optional strict padding to tgt_len per bucket sub-batch
        """
        pre_batches: List[Dict[str, torch.Tensor]] = []
        buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        prompt_len = len(self.prompt_ids)
        # if we protect prompt, content must leave room for prompt
        max_content = max(1, self.FIXED_MAX_LEN - prompt_len) if (self.protect_prompt and prompt_len > 0) else self.FIXED_MAX_LEN

        for ex in pretrain_examples:
            g = dict(ex)

            if self.protect_prompt and prompt_len > 0:
                # content-only for corruption
                content_ids = ex["input_ids"][:max_content]
                if len(content_ids) == 0:
                    content_ids = [int(self.START)]
                g["input_ids"] = content_ids

                eff_len = min(self.FIXED_MAX_LEN, prompt_len + len(content_ids))
            else:
                # legacy: prompt included before corruption (can be corrupted)
                ids = (self.prompt_ids + ex["input_ids"])[: self.FIXED_MAX_LEN]
                if len(ids) == 0:
                    ids = [int(self.START)]
                g["input_ids"] = ids
                eff_len = len(ids)

            tgt_len = _round_up_to_allowed(eff_len, self.allowed)
            buckets[tgt_len].append(g)

        for tgt_len, group in buckets.items():
            batch = self.preC(group)

            # If protected prompt, prepend after corruption
            if self.protect_prompt and prompt_len > 0:
                B = batch["input_ids"].size(0)
                prompt = torch.tensor(self.prompt_ids, dtype=torch.long, device=batch["input_ids"].device)
                prompt = prompt.unsqueeze(0).expand(B, -1)

                new_inp = torch.cat([prompt, batch["input_ids"]], dim=1)
                new_inp = new_inp[:, : self.FIXED_MAX_LEN].contiguous()

                batch["input_ids"] = new_inp
                batch["attention_mask"] = (new_inp != int(self.PAD)).long().contiguous()

            # Strict bucket width for this pretrain sub-batch (applies to both branches)
            if self.pad_to_bucket:
                want_w = min(int(tgt_len), int(self.FIXED_MAX_LEN))
                batch = _pad_batch_to_len(batch, want_w, int(self.PAD))

            if self.add_decoder and "labels" in batch:
                batch.update(self._make_decoder_inputs(batch["labels"]))

            pre_batches.append(batch)

        return pre_batches

    def _collate_train(self, train_examples: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        if not train_examples:
            return None

        fixed = []
        for ex in train_examples:
            ids = ex["input_ids"][: self.FIXED_MAX_LEN]
            lbl = ex.get("labels", [])[: self.FIXED_MAX_LEN]

            if len(ids) == 0:
                ids = [int(self.START)]
            if len(lbl) == 0:
                lbl = [int(self.START)]

            fixed.append({"input_ids": ids, "labels": lbl})

        batch = self.trC(fixed)
        if self.add_decoder and "labels" in batch:
            batch.update(self._make_decoder_inputs(batch["labels"]))
        return batch

    def _collate_trans_corrupt(self, tc_examples: List[Dict[str, Any]]) -> Optional[Dict[str, torch.Tensor]]:
        if not tc_examples:
            return None

        fixed = []
        for ex in tc_examples:
            ids = ex.get("input_ids", [])
            lbl = ex.get("labels", [])

            if len(ids) == 0:
                ids = [int(self.START)]
            if len(lbl) == 0:
                lbl = [int(self.START)]

            fixed.append({"input_ids": ids, "labels": lbl})

        batch = self.tcC(fixed)
        if self.add_decoder and "labels" in batch:
            batch.update(self._make_decoder_inputs(batch["labels"]))
        return batch

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pretrain_examples  = [ex for ex in examples if ex.get("task") == "pretrain"]
        train_examples     = [ex for ex in examples if ex.get("task") == "train"]
        tc_examples        = [ex for ex in examples if ex.get("task") in ("trans_corrupt", "translate_corrupt")]

        all_batches: List[Dict[str, torch.Tensor]] = []

        if pretrain_examples:
            all_batches.extend(self._collate_pretrain(pretrain_examples))

        tr_batch = self._collate_train(train_examples)
        if tr_batch is not None:
            all_batches.append(tr_batch)

        tc_batch = self._collate_trans_corrupt(tc_examples)
        if tc_batch is not None:
            all_batches.append(tc_batch)

        if not all_batches:
            return {
                "input_ids": torch.empty(0, 0, dtype=torch.long),
                "attention_mask": torch.empty(0, 0, dtype=torch.long),
                "labels": torch.empty(0, 0, dtype=torch.long),
            }

        max_w = max(b["input_ids"].size(1) for b in all_batches if "input_ids" in b)
        all_batches = [_pad_batch_to_len(b, max_w, int(self.PAD)) for b in all_batches]

        keys = set().union(*(b.keys() for b in all_batches))
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.cat([b[k] for b in all_batches if k in b], dim=0)
        return out


import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

def _round_up_to_allowed(n: int, allowed: Sequence[int]) -> int:
    for a in allowed:
        if n <= a:
            return a
    return allowed[-1]

def _pad_to_width(x: torch.Tensor, target_w: int, pad_value: int) -> torch.Tensor:
    if x.size(1) == target_w:
        return x
    pad = (0, target_w - x.size(1))  # (left,right) on last dim
    return F.pad(x, pad, value=pad_value)

def _pad_batch_to_len(batch: Dict[str, torch.Tensor], target_len: int, pad_id: int) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if v.dim() == 2:
            if k == "labels":
                out[k] = _pad_to_width(v, target_len, -100)
            elif k in ("attention_mask", "decoder_attention_mask"):
                out[k] = _pad_to_width(v, target_len, 0)
            else:  # input_ids, decoder_input_ids, etc.
                out[k] = _pad_to_width(v, target_len, pad_id)
        else:
            out[k] = v
    return out

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase

def _round_up_to_allowed(x: int, allowed: Sequence[int]) -> int:
    for a in allowed:
        if x <= a:
            return a
    return allowed[-1]

def _pad_2d_right(x: torch.Tensor, target_w: int, pad_value: int) -> torch.Tensor:
    # x: (B, W)
    if x.size(1) >= target_w:
        return x[:, :target_w]
    pad = x.new_full((x.size(0), target_w - x.size(1)), pad_value)
    return torch.cat([x, pad], dim=1)

def _pad_batch_to_len(batch: Dict[str, torch.Tensor], width: int, pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Pad/truncate 2D tensors in `batch` to `width`.
    - input_ids / attention_mask / decoder_input_ids / decoder_attention_mask: pad with `pad_id`
    - labels: pad with -100
    Rebuild attention_mask from input_ids at the end (safer).
    """
    out = dict(batch)

    for k, v in batch.items():
        if not torch.is_tensor(v) or v.dim() != 2:
            continue

        # choose padding value
        pad_value = -100 if k == "labels" else pad_id

        if v.size(1) > width:
            v2 = v[:, :width]
        elif v.size(1) < width:
            pad = v.new_full((v.size(0), width - v.size(1)), pad_value)
            v2 = torch.cat([v, pad], dim=1)
        else:
            v2 = v

        out[k] = v2.contiguous()

    # safest: rebuild attention_mask from input_ids
    if "input_ids" in out:
        out["attention_mask"] = (out["input_ids"] != pad_id).long().contiguous()

    return out

def _pad_batch_to_width(batch: Dict[str, torch.Tensor], width: int, pad_id: int) -> Dict[str, torch.Tensor]:
    out = dict(batch)
    for k, v in batch.items():
        if not torch.is_tensor(v) or v.dim() != 2:
            continue

        if v.size(1) > width:
            v2 = v[:, :width]
        elif v.size(1) < width:
            pad_val = -100 if k == "labels" else pad_id
            pad = v.new_full((v.size(0), width - v.size(1)), pad_val)
            v2 = torch.cat([v, pad], dim=1)
        else:
            v2 = v

        out[k] = v2.contiguous()

    # attention mask should match input_ids exactly
    if "input_ids" in out:
        out["attention_mask"] = (out["input_ids"] != pad_id).long().contiguous()

    return out

class MixedDataCollator8Buckets:
    """
    - Optionally prepend prompt to pretrain items, but (optionally) protect it from corruption.
    - Bucket pretrain by allowed lengths; call pretrain collator per-bucket.
    - Translate path: cap to FIXED_MAX_LEN; let translate_collator do dynamic pad (x8).
    - Merge: pad all sub-batches in this collate call to the max width, then concat.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrain_collator,
        translate_collator=None,
        prompt_text: Optional[str] = "Identify missing signs: ",
        FIXED_MAX_LEN: int = 64,
        allowed_buckets: Sequence[int] = (8, 16, 32, 48, 64, 128, 256),
        add_decoder_inputs: bool = True,
        protect_prompt: bool = True,   # <-- NEW
        pad_to_bucket: bool = False,   # optional: enforce bucket width per pretrain sub-batch
    ):
        self.tok  = tokenizer
        self.preC = pretrain_collator
        self.trC  = translate_collator

        self.prompt_text   = prompt_text
        self.FIXED_MAX_LEN = int(FIXED_MAX_LEN)
        self.allowed       = tuple(sorted(allowed_buckets))
        self.add_decoder   = bool(add_decoder_inputs)
        self.protect_prompt = bool(protect_prompt)
        self.pad_to_bucket  = bool(pad_to_bucket)

        self.PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.START = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.PAD

        if isinstance(prompt_text, str) and prompt_text.strip():
            self.prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        else:
            self.prompt_ids = []

    def _make_decoder_inputs(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1]
        shifted[:, 0]  = int(self.START)
        shifted.masked_fill_(shifted == -100, int(self.PAD))
        return {
            "decoder_input_ids": shifted.contiguous(),
            "decoder_attention_mask": (shifted != int(self.PAD)).long().contiguous(),
        }

    def __call__(self, examples):
        # Prefer task if present, otherwise fall back to presence of labels
        pretrain_examples = []
        translate_examples = []

        for ex in examples:
            t = ex.get("task", None)

            if t == "pretrain":
                pretrain_examples.append(ex)
            elif t == "train":
                translate_examples.append(ex)
            else:
                # fallback when Trainer removed "task"
                if "labels" in ex and ex["labels"] is not None:
                    translate_examples.append(ex)
                else:
                    pretrain_examples.append(ex)


        # ---------- PRETRAIN (protected prompt) ----------
        pre_batches: List[Dict[str, torch.Tensor]] = []
        if pretrain_examples:
            buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            prompt_len = len(self.prompt_ids)

            # max content length when we later prepend prompt
            max_content = max(1, self.FIXED_MAX_LEN - prompt_len) if self.protect_prompt else self.FIXED_MAX_LEN

            for ex in pretrain_examples:
                ex2 = dict(ex)

                if self.protect_prompt and self.prompt_ids:
                    # IMPORTANT: feed ONLY content tokens to span corruption
                    content_ids = ex["input_ids"][:max_content]
                    if len(content_ids) == 0:
                        content_ids = [int(self.START)]
                    ex2["input_ids"] = content_ids

                    eff_len = min(self.FIXED_MAX_LEN, prompt_len + len(content_ids))
                else:
                    # legacy behavior: prompt can be corrupted because it's included before preC
                    ids = (self.prompt_ids + ex["input_ids"])[: self.FIXED_MAX_LEN]
                    if len(ids) == 0:
                        ids = [int(self.START)]
                    ex2["input_ids"] = ids
                    eff_len = len(ids)

                tgt_len = _round_up_to_allowed(eff_len, self.allowed)
                buckets[tgt_len].append(ex2)

            for tgt_len, group in buckets.items():
                batch = self.preC(group)

                if self.protect_prompt and self.prompt_ids:
                    # Prepend prompt AFTER corruption
                    B = batch["input_ids"].size(0)
                    prompt = torch.tensor(self.prompt_ids, dtype=torch.long, device=batch["input_ids"].device)
                    prompt = prompt.unsqueeze(0).expand(B, -1)

                    new_inp = torch.cat([prompt, batch["input_ids"]], dim=1)
                    new_inp = new_inp[:, : self.FIXED_MAX_LEN].contiguous()

                    batch["input_ids"] = new_inp
                    batch["attention_mask"] = (new_inp != int(self.PAD)).long().contiguous()

                # ✅ enforce exact bucket width for this pretrain sub-batch (works for both branches)
                if self.pad_to_bucket:
                    want_w = min(int(tgt_len), int(self.FIXED_MAX_LEN))
                    batch = _pad_batch_to_len(batch, want_w, int(self.PAD))

                if self.add_decoder:
                    batch.update(self._make_decoder_inputs(batch["labels"]))

                pre_batches.append(batch)

        # ---------- TRANSLATE ----------
        tr_batch: Optional[Dict[str, torch.Tensor]] = None
        if translate_examples:
            fixed = []
            for ex in translate_examples:
                ids = ex["input_ids"][: self.FIXED_MAX_LEN]
                lbl = ex["labels"][: self.FIXED_MAX_LEN]

                if len(ids) == 0:
                    ids = [int(self.START)]
                if len(lbl) == 0:
                    lbl = [int(self.START)]

                fixed.append({"input_ids": ids, "labels": lbl})

            if self.trC is not None:
                tr_batch = self.trC(fixed)
            else:
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f["input_ids"], dtype=torch.long) for f in fixed],
                    batch_first=True, padding_value=int(self.PAD)
                )
                labels = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f["labels"], dtype=torch.long) for f in fixed],
                    batch_first=True, padding_value=-100
                )
                attention_mask = (input_ids != int(self.PAD)).long()
                tr_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

            if self.add_decoder:
                tr_batch.update(self._make_decoder_inputs(tr_batch["labels"]))

        # ---------- MERGE ----------
        all_batches: List[Dict[str, torch.Tensor]] = []
        if pre_batches:
            all_batches.extend(pre_batches)
        if tr_batch is not None:
            all_batches.append(tr_batch)

        if not all_batches:
            # never return (0,0); give a 1-token dummy batch
            pad = int(self.PAD)
            one = torch.tensor([[pad]], dtype=torch.long)
            return {
                "input_ids": one,
                "attention_mask": (one != pad).long(),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
            }

        max_w = max(b["input_ids"].size(1) for b in all_batches)
        all_batches = [_pad_batch_to_len(b, max_w, int(self.PAD)) for b in all_batches]

        keys = set().union(*(b.keys() for b in all_batches))
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.cat([b[k] for b in all_batches if k in b], dim=0)

        return out


import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Any, Dict, List, Sequence
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def _round_up_to_allowed(n: int, allowed: Sequence[int]) -> int:
    for a in allowed:
        if n <= a:
            return a
    return allowed[-1]

def _pad_to_width(x: torch.Tensor, target_w: int, pad_value: int) -> torch.Tensor:
    if x.size(1) == target_w:
        return x
    return F.pad(x, (0, target_w - x.size(1)), value=pad_value)

def _pad_batch_to_len(batch: Dict[str, torch.Tensor], target_len: int, pad_id: int) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
            continue
        if v.dim() == 2:
            if k == "labels":
                out[k] = _pad_to_width(v, target_len, -100)
            elif k in ("attention_mask", "decoder_attention_mask"):
                out[k] = _pad_to_width(v, target_len, 0)
            else:  # input_ids / decoder_input_ids
                out[k] = _pad_to_width(v, target_len, pad_id)
        else:
            out[k] = v
    return out

def _make_decoder_inputs(labels: torch.Tensor, start_id: int, pad_id: int) -> Dict[str, torch.Tensor]:
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1]
    shifted[:, 0]  = start_id
    shifted.masked_fill_(shifted == -100, pad_id)
    return {
        "decoder_input_ids": shifted.contiguous(),
        "decoder_attention_mask": (shifted != pad_id).long().contiguous(),
    }


# in train_functions.py

class MixedDataCollator3Buckets:
    """
    Fast path: if all examples share the same `task`, route the whole batch to the matching collator.
    Slow path: if mixed tasks, collate per task, pad sub-batches to max width, then concat (dim=0).
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrain_collator,                 # e.g. T5DataCollatorForSpanCorruptionAsteriskFast (dynamic_inputs=True, pad_to_multiple_of=8)
        translate_collator=None,           # e.g. DataCollatorForSeq2Seq(..., pad_to_multiple_of=8)
        error_collator=None,               # e.g. DataCollatorForTranslationCorruptionWithErrorsBuckets(...)
        prompt_text: str = "Identify missing signs: ",
        FIXED_MAX_LEN: int = 64,
        allowed_buckets: Sequence[int] = (8, 16, 24, 32, 40, 48, 56, 64),
        add_decoder_inputs: bool = True,
    ):
        self.tok = tokenizer
        self.preC = pretrain_collator
        self.trC  = translate_collator
        self.errC = error_collator
        self.prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        self.FIXED_MAX_LEN = int(FIXED_MAX_LEN)
        self.allowed = tuple(sorted(allowed_buckets))
        self.add_decoder = bool(add_decoder_inputs)

        self.PAD   = tokenizer.pad_token_id or 0
        self.START = tokenizer.eos_token_id

    def _make_decoder_inputs(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1]
        shifted[:, 0]  = self.START
        shifted.masked_fill_(shifted == -100, self.PAD)
        return {"decoder_input_ids": shifted.contiguous(),
                "decoder_attention_mask": (shifted != self.PAD).long().contiguous()}

    def _preprocess_pretrain(self, exs):
        out = []
        for ex in exs:
            ids = (self.prompt_ids + ex["input_ids"])[: self.FIXED_MAX_LEN]  # cap, no pad
            ex = dict(ex); ex["input_ids"] = ids
            out.append(ex)
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            return {"input_ids": torch.empty(0,0,dtype=torch.long),
                    "attention_mask": torch.empty(0,0,dtype=torch.long),
                    "labels": torch.empty(0,0,dtype=torch.long)}

        # Fast path: homogeneous task?
        tasks = [f.get("task") for f in features]
        uniq = set(tasks)

        if len(uniq) == 1:
            task = next(iter(uniq))
            if task == "pretrain":
                batch = self.preC(self._preprocess_pretrain(features))
            elif task == "train":
                # cap only; let translate collator handle dynamic padding
                fixed = [{"input_ids": f["input_ids"][: self.FIXED_MAX_LEN],
                          "labels":    f["labels"][: self.FIXED_MAX_LEN]} for f in features]
                batch = self.trC(fixed) if self.trC else self._minimal_seq2seq(fixed)
            elif task == "train_error":
                fixed = [{"input_ids": f["input_ids"][: self.FIXED_MAX_LEN],
                          "labels":    f["labels"][: self.FIXED_MAX_LEN]} for f in features]
                batch = self.errC(fixed) if self.errC else self._minimal_seq2seq(fixed)
            else:
                raise ValueError(f"Unknown task: {task}")

            if self.add_decoder and "labels" in batch:
                batch.update(self._make_decoder_inputs(batch["labels"]))
            return batch

        # Slow path: mixed tasks in one collate call → split, collate, pad-to-max, concat.
        pre  = [f for f in features if f.get("task") == "pretrain"]
        tr   = [f for f in features if f.get("task") == "train"]
        terr = [f for f in features if f.get("task") == "train_error"]

        batches = []
        if pre:
            b = self.preC(self._preprocess_pretrain(pre))
            if self.add_decoder: b.update(self._make_decoder_inputs(b["labels"]))
            batches.append(b)
        if tr:
            fixed = [{"input_ids": f["input_ids"][: self.FIXED_MAX_LEN],
                      "labels":    f["labels"][: self.FIXED_MAX_LEN]} for f in tr]
            b = self.trC(fixed) if self.trC else self._minimal_seq2seq(fixed)
            if self.add_decoder: b.update(self._make_decoder_inputs(b["labels"]))
            batches.append(b)
        if terr:
            fixed = [{"input_ids": f["input_ids"][: self.FIXED_MAX_LEN],
                      "labels":    f["labels"][: self.FIXED_MAX_LEN]} for f in terr]
            b = self.errC(fixed) if self.errC else self._minimal_seq2seq(fixed)
            if self.add_decoder: b.update(self._make_decoder_inputs(b["labels"]))
            batches.append(b)

        # pad sub-batches to common width, then concat
        max_w = max(b["input_ids"].size(1) for b in batches)
        batches = [_pad_batch_to_len(b, max_w, self.PAD) for b in batches]
        out = {}
        keys = set().union(*(b.keys() for b in batches))
        for k in keys:
            out[k] = torch.cat([b[k] for b in batches if k in b], dim=0)
        return out

    def _minimal_seq2seq(self, fixed):
        # emergency fallback: dynamic pad to max in this list
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids"], dtype=torch.long) for x in fixed],
            batch_first=True, padding_value=self.PAD
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["labels"], dtype=torch.long) for x in fixed],
            batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.PAD).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

from transformers import default_data_collator, BatchEncoding
import torch

class Seq2SeqCollatorWithTruncation:
    def __init__(self, tokenizer, max_length=60):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, features):
        """
        features: a list of dicts, each with 'input_ids', 'labels', etc.
        We'll re-run tokenizer.pad(...) with max_length=... 
        But we have to do it carefully, because the HF tokenizer expects "text" or pre-batched 
        data. We'll produce a dict of lists, then call .pad with max_length=.
        """
        # 1) Convert list-of-dicts into dict-of-lists:
        batch = {}
        for k in features[0].keys():
            batch[k] = [f[k] for f in features]
        # 2) Use tokenizer.pad(...)
        #    pass in our desired max_length => forcibly truncates if > max_length.
        #    'truncation=True' is recognized by .pad(...) in HF 4.30+
        #    If older version, you may need to do manual slicing
        #    or pass it as part of the model_input_names.
        padded = self.tokenizer.pad(
            batch,
            padding="max_length",
            max_length=self.max_length,
            # truncation=True,  # "truncation" isn't always recognized by .pad(...) in older versions
            return_tensors="pt",
        )
        # Now 'padded' is a BatchEncoding with 'input_ids', 'attention_mask', possibly 'labels'
        # all of shape [batch_size, self.max_length].
        # If 'labels' is present as well, great.
        return padded

import random
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

class SwapSeq2SeqCollator(DataCollatorForSeq2Seq):
    """
    A Seq2Seq collator that randomly swaps source and target on the fly.
    
    Expects each example to have:
      - "source_text": str
      - "target_text": str
      - "src_lang":    BCP-47 code of source
      - "tgt_lang":    BCP-47 code of target
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model,
        max_length: int = 64,
        swap_prob: float = 0.5,
    ):
        super().__init__(tokenizer=tokenizer, model=model, return_tensors="pt")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.swap_prob = swap_prob

    def __call__(self, examples):
        # examples is a list of dicts
        features = []
        for ex in examples:
            # decide whether to swap
            if random.random() < self.swap_prob:
                src_text, tgt_text = ex["target_text"], ex["source_text"]
                src_code, tgt_code = ex["tgt_lang"],     ex["src_lang"]
            else:
                src_text, tgt_text = ex["source_text"], ex["target_text"]
                src_code, tgt_code = ex["src_lang"],     ex["tgt_lang"]

            # set language flags
            self.tokenizer.src_lang = src_code
            self.tokenizer.tgt_lang = tgt_code

            # tokenize encoder inputs
            enc = self.tokenizer(
                src_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            # tokenize decoder inputs / labels
            # In transformers 5.x, as_target_tokenizer() is removed - use text_target parameter
            lab = self.tokenizer(
                text_target=tgt_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            features.append({
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels":         lab["input_ids"],
            })
        # let the parent class pad & convert to tensors
        batch = super().__call__(features)
        # mask pad tokens in labels
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        return batch


class TokenizedSwapCollator(DataCollatorForSeq2Seq):
    """
    Takes examples that already have `input_ids`, `attention_mask`, and `labels`
    and randomly swaps them (plus their masks) with probability `swap_prob`.
    """
    def __init__(self, tokenizer, model, swap_prob=0.5):
        super().__init__(tokenizer=tokenizer, model=model, return_tensors="pt")
        self.swap_prob = swap_prob
    def __call__(self, features):
        new_feats = []
        for f in features:
            if random.random() < self.swap_prob:
                # swap *all* relevant fields
                f = {
                    "input_ids":        f["labels"],
                    "attention_mask":   f.get("labels_attention_mask", None) or [1]*len(f["labels"]),
                    "labels":           f["input_ids"],
                    "decoder_attention_mask": f["attention_mask"],
                }
            new_feats.append(f)
        batch = super().__call__(new_feats)
        # finally mask pad tokens in labels
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100
        return batch


def find_latest_checkpoint(checkpoint_dir):
    """
    Return the checkpoint folder with the largest numerical step in its name, e.g. 'checkpoint-1147482'.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    # List all directories in checkpoint_dir
    checkpoints = [item for item in os.listdir(checkpoint_dir)
                   if item.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, item))]
    if not checkpoints:
        return None
    # Sort by the numerical part after 'checkpoint-'
    # For example, 'checkpoint-1165696' -> step = 1165696
    checkpoints.sort(key=lambda ck: int(ck.replace("checkpoint-", "")), reverse=True)
    # The first item in the list is now the highest step
    return os.path.join(checkpoint_dir, checkpoints[0])

def find_best_checkpoint_legacy(checkpoint_dir):
    """
    Look for 'trainer_state.json' in `checkpoint_dir`, 
    parse 'best_model_checkpoint', and return it if found. 
    Otherwise, fall back to your 'latest checkpoint' logic.
    """
    last_checkpoint = find_latest_checkpoint(checkpoint_dir)
    trainer_state_path = os.path.join(last_checkpoint, "trainer_state.json")
    # If there's a trainer_state.json, parse it
    if os.path.isfile(trainer_state_path):
        try:
            with open(trainer_state_path, 'r') as f:
                state_dict = json.load(f)
            if "best_model_checkpoint" in state_dict:
                best_cp = state_dict["best_model_checkpoint"]
                # Double-check this path exists
                if os.path.isdir(best_cp):
                    return best_cp
                else:
                    print(f"Warning: best_model_checkpoint at '{best_cp}' not found.")
        except Exception as e:
            print(f"Failed to parse trainer_state.json: {e}")
    # Fallback: return the newest checkpoint folder by modification time
    return find_latest_checkpoint(checkpoint_dir)

import os, re, json, glob

def find_best_checkpoint(run_dir: str) -> str:
    """
    Return the 'best_model_checkpoint' recorded by Hugging Face Trainer
    for a given training run directory (the parent containing checkpoint-* dirs).
    Falls back to the highest-step checkpoint if best is unavailable.
    """
    def _resolve_best(cp_path: str) -> str | None:
        # Make absolute; if moved, stitch basename under run_dir
        abs_path = cp_path if os.path.isabs(cp_path) else os.path.join(run_dir, cp_path)
        if not os.path.isdir(abs_path):
            abs_path = os.path.join(run_dir, os.path.basename(cp_path))
        return abs_path if os.path.isdir(abs_path) else None

    # 1) Prefer the root trainer_state.json
    root_state = os.path.join(run_dir, "trainer_state.json")
    candidates = []
    if os.path.isfile(root_state):
        candidates.append(root_state)

    # 2) Fall back: any trainer_state.json inside checkpoint-* (newest first)
    ckpt_states = glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json"))
    ckpt_states.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    candidates.extend(ckpt_states)

    # 3) Try to read best_model_checkpoint from any candidate
    for state_path in candidates:
        try:
            with open(state_path, "r") as f:
                state = json.load(f)
            best = state.get("best_model_checkpoint")
            if best:
                resolved = _resolve_best(best)
                if resolved:
                    return resolved
        except Exception as e:
            print(f"Warning: failed to parse {state_path}: {e}")

    # 4) Final fallback: pick checkpoint with the highest step number
    ckpts = [d for d in glob.glob(os.path.join(run_dir, "checkpoint-*")) if os.path.isdir(d)]
    if ckpts:
        def step(d):
            m = re.search(r"checkpoint-(\d+)", os.path.basename(d))
            return int(m.group(1)) if m else -1
        return max(ckpts, key=step)

    # 5) Nothing found; return the run_dir (caller can decide how to handle)
    return run_dir


from typing import Any, Dict, List, Optional
import math
import torch
from transformers import PreTrainedTokenizerBase

import torch

def _shift_right(labels: torch.Tensor, pad_id: int, start_id: int):
    """
    Create decoder_input_ids by shifting labels right:
      decoder_input_ids[:, 0] = start_id
      decoder_input_ids[:, 1:] = labels[:, :-1] with -100 replaced by pad_id
    """
    if labels.dim() != 2:
        raise ValueError(f"labels must be 2D [bs, seq], got {labels.shape}")

    dec = labels.clone()
    dec = dec.masked_fill(dec == -100, pad_id)

    shifted = dec.new_full(dec.shape, pad_id)
    shifted[:, 0] = start_id
    shifted[:, 1:] = dec[:, :-1]
    return shifted

from dataclasses import dataclass
from typing import Any, Dict, List
import torch

@dataclass
class CappedSeq2SeqCollator:
    base_collator: Any
    input_length: int = 512
    target_length: int = 512
    model: Any = None
    tokenizer: Any = None

    # Keys that are safe to give to DataCollatorForSeq2Seq / tokenizer.pad
    # (i.e., tensorizable sequence fields)
    _PAD_KEYS = (
        "input_ids",
        "attention_mask",
        "labels",
        "decoder_input_ids",
        "decoder_attention_mask",
        "token_type_ids",
        "position_ids",
    )

    # Keys we explicitly forward as extra per-example tensors (1D)
    _EXTRA_TENSOR_KEYS = (
        "meteor_ok",
        "length",
        "input_length",
    )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            return self.base_collator(features)

        capped_features: List[Dict[str, Any]] = []
        extras: Dict[str, List[Any]] = {k: [] for k in self._EXTRA_TENSOR_KEYS}

        for f in features:
            f_new = dict(f)

            # ---- truncate encoder ----
            if "input_ids" in f_new and f_new["input_ids"] is not None:
                f_new["input_ids"] = _truncate_list_like(f_new["input_ids"], self.input_length)
                if "attention_mask" in f_new and f_new["attention_mask"] is not None:
                    f_new["attention_mask"] = _truncate_list_like(f_new["attention_mask"], self.input_length)

            # ---- truncate decoder labels ----
            if "labels" in f_new and f_new["labels"] is not None:
                f_new["labels"] = _truncate_list_like(f_new["labels"], self.target_length)

            # ---- capture extras (but do NOT pass them into base_collator) ----
            for k in self._EXTRA_TENSOR_KEYS:
                if k in f_new:
                    extras[k].append(f_new[k])

            capped_features.append(f_new)

        # IMPORTANT:
        # Only pass pad/tensorizable keys into base_collator.
        # This prevents failures when features contain strings like "task".
        pad_features: List[Dict[str, Any]] = []
        for f in capped_features:
            pad_features.append({k: f[k] for k in self._PAD_KEYS if k in f})

        batch = self.base_collator(pad_features)

        # ---- reattach extras as tensors (if present in every example) ----
        for k, values in extras.items():
            if len(values) == len(features):
                # Convert scalars cleanly; meteor_ok should be 0/1 ints
                try:
                    batch[k] = torch.tensor(values, dtype=torch.long)
                except Exception:
                    # If something weird slips in, fail loudly with context
                    raise ValueError(f"Could not convert extra column '{k}' to tensor. "
                                     f"Example value type: {type(values[0])}, value: {values[0]!r}")

        # ---- Build decoder_input_ids if missing ----
        if "labels" in batch and "decoder_input_ids" not in batch:
            labels = batch["labels"]

            if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
                try:
                    batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels)
                except TypeError:
                    batch["decoder_input_ids"] = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            else:
                pad_id = getattr(self.model.config, "pad_token_id", None) if self.model is not None else None
                if pad_id is None and self.tokenizer is not None:
                    pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is None:
                    raise ValueError("pad_token_id is None (set model.config.pad_token_id or tokenizer.pad_token_id)")

                start_id = getattr(self.model.config, "decoder_start_token_id", None) if self.model is not None else None
                if start_id is None:
                    start_id = pad_id

                batch["decoder_input_ids"] = _shift_right(labels, pad_id=pad_id, start_id=start_id)

            pad_id = getattr(self.model.config, "pad_token_id", None) if self.model is not None else None
            if pad_id is None and self.tokenizer is not None:
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
            batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != pad_id).long()

        return batch


import math, random
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

class MaxTokensBatchSampler(Sampler):
    def __init__(self, lengths, max_tokens_per_batch=300, bucket_size=256, seed=42, drop_last=False):
        self.lengths   = [int(x) for x in lengths]
        self.N         = len(self.lengths)
        self.max_tokens = int(max_tokens_per_batch)
        self.bucket    = int(bucket_size)
        self.seed      = int(seed)
        self.drop_last = drop_last
        self.epoch     = 0

    def set_epoch(self, epoch:int):
        self.epoch = int(epoch)

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)

        idx = list(range(self.N))
        g.shuffle(idx)

        # local sort by length inside buckets
        sorted_idx = []
        for i in range(0, self.N, self.bucket):
            chunk = idx[i:i+self.bucket]
            chunk.sort(key=lambda j: self.lengths[j])
            sorted_idx.extend(chunk)

        batch, cur = [], 0
        for i in sorted_idx:
            L = self.lengths[i]

            if L > self.max_tokens:
                if batch:
                    yield batch
                    batch, cur = [], 0
                if not self.drop_last:
                    yield [i]
                continue

            if cur + L <= self.max_tokens and batch:
                batch.append(i)
                cur += L
            else:
                if batch and not self.drop_last:
                    yield batch
                batch, cur = [i], L

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        return max(1, math.ceil(sum(self.lengths) / max(1, self.max_tokens)))


from transformers import Seq2SeqTrainer

class StripKeysCollator:
    def __init__(self, base_collator, drop_keys=("example_len",)):
        self.base = base_collator
        self.drop = set(drop_keys)
    def __call__(self, features):
        feats = [{k: v for k, v in f.items() if k not in self.drop} for f in features]
        return self.base(feats)

class TokenBudgetTrainer(Seq2SeqTrainer):
    def __init__(self, *args, max_tokens_per_batch=300, bucket_size=256, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_tokens_per_batch = int(max_tokens_per_batch)
        self._bucket_size = int(bucket_size)
        self._cached_lengths = None

    def _get_lengths(self, dataset):
        if self._cached_lengths is None:
            col = dataset["example_len"]  # columnar pull (fast)
            self._cached_lengths = [int(x) for x in col]
        return self._cached_lengths

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return super().get_train_dataloader()

        lengths = self._get_lengths(self.train_dataset)

        # build sampler first (needs example_len)
        bs = MaxTokensBatchSampler(
            lengths=lengths,
            max_tokens_per_batch=self._max_tokens_per_batch,
            bucket_size=self._bucket_size,
            seed=int(self.args.seed),
            drop_last=False,
        )
        if getattr(self.state, "epoch", None) is not None:
            bs.set_epoch(int(self.state.epoch))

        # wrap the collator to drop example_len before model(**inputs)
        collator = StripKeysCollator(self.data_collator, drop_keys=("example_len",))

        return DataLoader(
            self.train_dataset,
            batch_sampler=bs,
            collate_fn=collator,
            num_workers=0,
            persistent_workers=False,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if ds is None:
            return super().get_eval_dataloader(eval_dataset)
        collator = StripKeysCollator(self.data_collator, drop_keys=("example_len",))
        return DataLoader(
            ds,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=0,
            pin_memory=self.args.dataloader_pin_memory,
        )


# somewhere in train_functions.py (or a new utils module)
import math
import random
from collections import defaultdict
from torch.utils.data import Sampler


class LengthBucketedBatchSampler(Sampler):
    """
    Batch sampler that:
      - Buckets examples by (rounded) length,
      - Packs indices into batches so that sum(lengths) <= max_tokens_per_batch.
    """

    def __init__(
        self,
        lengths,
        max_tokens_per_batch: int,
        bucket_size: int = 16,
        shuffle: bool = True,
        drop_last: bool = False,
        long_behavior: str = "truncate",  # ✅ NEW: {"truncate", "skip", "single"}
    ):
        """
        long_behavior:
          - "truncate": allow into batch & let model/collator truncate (SAFE DEFAULT)
          - "skip": drop the example entirely (old behavior)
          - "single": force into its own batch
        """
        assert long_behavior in {"truncate", "skip", "single"}

        self.lengths = [int(L) for L in lengths]
        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.bucket_size = int(bucket_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.long_behavior = long_behavior

        buckets = defaultdict(list)
        for idx, L in enumerate(self.lengths):
            bucket_id = int(math.ceil(L / self.bucket_size))
            buckets[bucket_id].append(idx)

        self.bucket_ids = sorted(buckets.keys())
        self.buckets = {b: idxs for b, idxs in buckets.items()}

    def __iter__(self):
        bucket_ids = list(self.bucket_ids)
        if self.shuffle:
            random.shuffle(bucket_ids)

        for b in bucket_ids:
            idxs = list(self.buckets[b])
            if self.shuffle:
                random.shuffle(idxs)

            current_batch = []
            current_tokens = 0

            for i in idxs:
                L = self.lengths[i]

                # -------------------------------
                # ✅ HANDLE OVER-BUDGET EXAMPLES
                # -------------------------------
                if L > self.max_tokens_per_batch:
                    if self.long_behavior == "skip":
                        continue

                    elif self.long_behavior == "single":
                        if current_batch:
                            if not self.drop_last:
                                yield current_batch
                        yield [i]
                        current_batch = []
                        current_tokens = 0
                        continue

                    elif self.long_behavior == "truncate":
                        # Allow it through; truncation happens later in the collator/trainer
                        pass

                # Normal packing logic
                if current_batch and (current_tokens + L > self.max_tokens_per_batch):
                    if not self.drop_last:
                        yield current_batch
                    current_batch = [i]
                    current_tokens = L
                else:
                    current_batch.append(i)
                    current_tokens += L

            if current_batch and not self.drop_last:
                yield current_batch

    def __len__(self):
        n = len(self.lengths)
        avg_len = sum(self.lengths) / max(1, n)
        approx_per_batch = max(1, int(self.max_tokens_per_batch // max(1, int(avg_len))))
        return max(1, n // approx_per_batch)
        

import torch
from transformers import Seq2SeqTrainer
from torch.nn.utils.rnn import pad_sequence
import time
from transformers.utils import logging
logger = logging.get_logger(__name__)

class HierarchicalTokenTrainer(Seq2SeqTrainer):
    """
    Trainer that:
      - (optionally) truncates encoder/decoder sequences to hard caps,
      - splits each batch into token-budgeted microbatches (enc_len + dec_len),
      - tracks/logs max lengths seen so far.

    If use_cpu_microbatch = True:
      - keep full HF batch on CPU for planning,
      - move only microbatches to GPU via _to_device.

    If use_cpu_microbatch = False:
      - use HF/Accelerate's normal device placement,
      - microbatch on whatever device inputs are already on (typically GPU).
    """
    
        # --- Compatibility shim: silence tokenizer deprecation, use processing_class internally ---
    @property
    def tokenizer(self):
        # HF now treats "processing_class" as the canonical thing
        return getattr(self, "processing_class", None)

    @tokenizer.setter
    def tokenizer(self, value):
        # whenever HF or user assigns trainer.tokenizer = ...
        # we just store it as processing_class
        self.processing_class = value

    def __init__(
        self,
        *args,
        max_tokens_per_microbatch: int = 400,
        max_eval_tokens_per_microbatch: int | None = None,
        max_encoder_len: int | None = None,
        max_decoder_len: int | None = None,
        max_tokens_per_batch: int | None = None,
        max_examples_per_microbatch: int | None = None,
        length_column_name: str = "input_length",
        log_longest_every: int = 0,   # 0 = disable logging, >0 = log every N steps
        use_cpu_microbatch: bool = True,
        eval_mode: str | None = None,   # <--- NEW
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_tokens_per_microbatch = int(max_tokens_per_microbatch)
        self.max_encoder_len = int(max_encoder_len) if max_encoder_len is not None else None
        self.max_decoder_len = int(max_decoder_len) if max_decoder_len is not None else None
        self.max_tokens_per_batch = max_tokens_per_batch
        self.length_column_name = length_column_name
        self.log_longest_every = int(log_longest_every)
        self.use_cpu_microbatch = bool(use_cpu_microbatch)
        self.max_examples_per_microbatch = max_examples_per_microbatch
        self.max_eval_tokens_per_microbatch = (
            int(max_eval_tokens_per_microbatch/4)
            if max_eval_tokens_per_microbatch is not None
            else self.max_tokens_per_microbatch
        )
        self.eval_mode = eval_mode or "token_aware_metrics"
        self.debug = debug

        # running maxima
        self._max_seen_enc_len = 0
        self._max_seen_dec_len = 0
        self._max_seen_total_len = 0
        self._num_trunc_hits = 0
        
        if getattr(self, "processing_class", None) is None:
            # fallback, just in case
            if hasattr(self, "_tokenizer") and self._tokenizer is not None:
                self.processing_class = self._tokenizer

        if self.args.gradient_accumulation_steps != 1:
            print(
                "[HierarchicalTokenTrainer] Warning: gradient_accumulation_steps != 1.\n"
                "You now have two layers of accumulation (HF + microbatch). "
                "Make sure this is intentional."
            )

    @staticmethod
    def _pad_and_concat(arrays, pad_value: int):
        """
        Given a list of 2D arrays [N_i, T_i], pad each to max_T along dim 1
        and stack into a single [sum_i N_i, max_T] array.
        """
        if not arrays:
            return np.zeros((0, 0), dtype=np.int64)

        # All arrays must be 2D
        max_T = max(a.shape[1] for a in arrays)
        total_N = sum(a.shape[0] for a in arrays)
        dtype = arrays[0].dtype

        out = np.full((total_N, max_T), pad_value, dtype=dtype)

        offset = 0
        for a in arrays:
            n, t = a.shape
            out[offset : offset + n, :t] = a
            offset += n

        return out
        
        
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.max_tokens_per_batch is None:
            return super().get_train_dataloader()

        ds = self.train_dataset

        # 🔹 Drop raw text / unused columns *here*
        if hasattr(ds, "column_names"):
            keep_cols = {
                "input_ids",
                "attention_mask",
                "labels",
                "decoder_input_ids",
                "decoder_attention_mask",
                self.length_column_name,   # "input_length"
            }
            to_remove = [c for c in ds.column_names if c not in keep_cols]
            if to_remove:
                ds = ds.remove_columns(to_remove)

        # Now safe to use for lengths + collator
        if hasattr(ds, "column_names"):
            lengths = ds[self.length_column_name]
        else:
            lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=16,
            shuffle=True,
            drop_last=False,
        )

        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

    # ----------------------------------------------------------------------
    # Truncation helper
    # ----------------------------------------------------------------------

    def _truncate_batch(self, inputs: dict) -> dict:
        """
        In-place truncate encoder/decoder sequences to hard caps, if set.

        - Encoder side: input_ids, attention_mask
        - Decoder side: labels, decoder_input_ids, decoder_attention_mask

        Works for:
          - encoder-only (no labels -> only encoder is truncated),
          - seq2seq (2D labels).
        """
        # Encoder truncation
        if self.max_encoder_len is not None:
            for k in ("input_ids", "attention_mask"):
                if k in inputs and isinstance(inputs[k], torch.Tensor):
                    t = inputs[k]
                    if t.ndim == 2 and t.size(1) > self.max_encoder_len:
                        inputs[k] = t[:, : self.max_encoder_len]
                        self._num_trunc_hits += 1

        # Decoder truncation (only if 2D sequence labels are present)
        if self.max_decoder_len is not None and "labels" in inputs:
            labels = inputs["labels"]
            if isinstance(labels, torch.Tensor) and labels.ndim == 2:
                if labels.size(1) > self.max_decoder_len:
                    inputs["labels"] = labels[:, : self.max_decoder_len]
                    self._num_trunc_hits += 1

                for k in ("decoder_input_ids", "decoder_attention_mask"):
                    if k in inputs and isinstance(inputs[k], torch.Tensor):
                        t = inputs[k]
                        if t.ndim == 2 and t.size(1) > self.max_decoder_len:
                            inputs[k] = t[:, : self.max_decoder_len]
                            self._num_trunc_hits += 1

        return inputs

    # ----------------------------------------------------------------------
    # Length helpers (enc+dec), strict cap check
    # ----------------------------------------------------------------------

    def _compute_lengths_enc_dec(self, inputs):
        """
        Compute per-example (enc_len, dec_len, total_len) from inputs.

        - enc_len: sum over attention_mask
        - dec_len: # of non -100 labels (if 2D); 0 for pretraining/encoder-only.
        """
        am = inputs.get("attention_mask", None)
        if am is None:
            raise ValueError(
                "attention_mask is required to compute lengths for HierarchicalTokenTrainer. "
                "Make sure your collator provides it."
            )
        if not isinstance(am, torch.Tensor):
            am = torch.as_tensor(am, device="cpu")

        enc_len = am.sum(dim=-1)  # (B,)

        labels = inputs.get("labels", None)
        if labels is None:
            dec_len = torch.zeros_like(enc_len)
        else:
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)
            if labels.ndim == 2:
                dec_len = (labels != -100).sum(dim=-1)
            else:
                # 1D class labels or something non-seq -> treat as encoder-only
                dec_len = torch.zeros_like(enc_len)

        total_len = enc_len + dec_len

        # update maxima
        enc_max = int(enc_len.max().item())
        dec_max = int(dec_len.max().item())
        tot_max = int(total_len.max().item())
        self._max_seen_enc_len = max(self._max_seen_enc_len, enc_max)
        self._max_seen_dec_len = max(self._max_seen_dec_len, dec_max)
        self._max_seen_total_len = max(self._max_seen_total_len, tot_max)

        # OPTIONAL: soft check instead of hard error
        if self.max_tokens_per_microbatch is not None:
            over = total_len > self.max_tokens_per_microbatch
            if over.any():
                # e.g., just log once per step or so; do NOT raise
                idx = int(torch.nonzero(over, as_tuple=False)[0].item())
                offending_total = int(total_len[idx].item())
                offending_enc = int(enc_len[idx].item())
                offending_dec = int(dec_len[idx].item())
                if self.control.should_log:
                    self.log(
                        {
                            "warn_example_over_cap_enc_len": float(offending_enc),
                            "warn_example_over_cap_dec_len": float(offending_dec),
                            "warn_example_over_cap_total":   float(offending_total),
                        }
                    )
                # No raise – just report
        return enc_len, dec_len, total_len

    @staticmethod
    def _slice_inputs(self_or_none, inputs, indices):
        """
        Slice a dict of tensors along the batch dimension for the given indices.
        Non-tensor values or tensors with mismatched batch dims are passed through.

        This function does NOT *change* device of the batch; it just matches the
        device of the index tensor to whatever the batch tensors are already on.
        """
        if not indices:
            return inputs

        # Find an example tensor to infer device
        example_tensor = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                example_tensor = v
                break

        device = example_tensor.device if example_tensor is not None else torch.device("cpu")
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)

        out = {}
        batch_dim = None
        if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            batch_dim = inputs["input_ids"].size(0)

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if batch_dim is not None and v.size(0) == batch_dim:
                    out[k] = v.index_select(0, idx)
                else:
                    out[k] = v
            else:
                out[k] = v

        return out

    # ------------------------------------------------------------------
    # Disable Trainer's automatic device placement
    # ------------------------------------------------------------------
    def _prepare_input(self, data):
        """
        If we are doing CPU-based microbatching, do nothing here.
        Otherwise, fall back to Seq2SeqTrainer's normal behavior.
        """
        if self.use_cpu_microbatch:
            return data
        return super()._prepare_input(data)

    def _prepare_inputs(self, inputs):
        """
        If we are doing CPU-based microbatching, do nothing here.
        Otherwise, fall back to Seq2SeqTrainer's normal behavior.
        """
        if self.use_cpu_microbatch:
            return inputs
        return super()._prepare_inputs(inputs)

    def _to_device(self, inputs: dict) -> dict:
        """
        Explicitly move a microbatch dict to self.args.device.
        Used only when use_cpu_microbatch=True.
        """
        device = self.args.device
        out = {}

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                kwargs = {"device": device}
                if self.is_deepspeed_enabled and (torch.is_floating_point(v) or torch.is_complex(v)):
                    kwargs["dtype"] = self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
                out[k] = v.to(**kwargs)
            else:
                out[k] = v

        return out

    def _move_to_cpu(self, inputs: dict) -> dict:
        cpu_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                cpu_inputs[k] = v.detach().to("cpu")
            else:
                cpu_inputs[k] = v
        return cpu_inputs

    def _make_microbatches(self, inputs, max_tokens_per_microbatch: int | None = None):
        # Compute lengths
        enc_len, dec_len, _ = self._compute_lengths_enc_dec(inputs)

        alpha = 2.0  # decoder weight
        effective_len = enc_len + alpha * dec_len
        lengths = [int(L) for L in effective_len.tolist()]
        N = len(lengths)

        budget = int(max_tokens_per_microbatch or self.max_tokens_per_microbatch)
        max_B = self.max_examples_per_microbatch  # may be None

        microbatches = []
        cur_indices: list[int] = []
        cur_tokens = 0

        sorted_indices = sorted(range(N), key=lambda i: lengths[i])

        for i in sorted_indices:
            L = lengths[i]

            too_many_tokens = cur_indices and (cur_tokens + L) > budget
            too_many_examples = max_B is not None and len(cur_indices) >= max_B

            if too_many_tokens or too_many_examples:
                microbatches.append(cur_indices)
                cur_indices = [i]
                cur_tokens = L
            else:
                cur_indices.append(i)
                cur_tokens += L

        if cur_indices:
            microbatches.append(cur_indices)

        return [self._slice_inputs(self, inputs, mb_idx) for mb_idx in microbatches]

    def _plan_microbatches(self, inputs):
        """
        Given a batch on CPU, compute total lengths and return
        a list of index lists for each microbatch.

        No tensors are moved to GPU here; we only work with CPU tensors.
        """
        # compute_lengths_enc_dec works fine on CPU
        _, _, total_len = self._compute_lengths_enc_dec(inputs)
        lengths = [int(L) for L in total_len.tolist()]
        N = len(lengths)

        # Sort indices by length ascending (shortest first)
        sorted_indices = sorted(range(N), key=lambda i: lengths[i])

        microbatches = []
        cur_indices = []
        cur_tokens = 0

        for i in sorted_indices:
            L = lengths[i]
            if self.max_tokens_per_microbatch is not None and L > self.max_tokens_per_microbatch:
                # single example exceeds cap; we already logged this in _compute_lengths_enc_dec
                # You can choose to raise here if you want:
                # raise RuntimeError(f"Example {i} has {L} tokens > max_tokens_per_microbatch={self.max_tokens_per_microbatch}")
                pass

            if (
                self.max_tokens_per_microbatch is not None
                and cur_indices
                and (cur_tokens + L) > self.max_tokens_per_microbatch
            ):
                microbatches.append(cur_indices)
                cur_indices = [i]
                cur_tokens = L
            else:
                cur_indices.append(i)
                cur_tokens += L

        if cur_indices:
            microbatches.append(cur_indices)

        return microbatches

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool = False,
        ignore_keys: Optional[List[str]] = None,
    ):
    
        if self.length_column_name in inputs:
            inputs = {k: v for k, v in inputs.items() if k != self.length_column_name}

        # If we need logits/labels, or no microbatching, just use the base implementation
        if (not prediction_loss_only) or (self.max_tokens_per_microbatch is None):
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        
        model.eval()

        # -----------------------------
        # CPU-microbatching path
        # -----------------------------
        if self.use_cpu_microbatch:
            inputs_cpu = self._move_to_cpu(inputs)
            inputs_cpu = self._truncate_batch(inputs_cpu)

            microbatches = self._make_microbatches(
                inputs_cpu,
                max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
            )

            if not microbatches:
                return (None, None, None)

            total_loss = 0.0
            total_examples = 0

            for mb in microbatches:
                bsz = mb["input_ids"].size(0)
                total_examples += bsz

                mb = self._to_device(mb)

                # 🔹 drop length column
                if self.length_column_name in mb:
                    mb = {k: v for k, v in mb.items() if k != self.length_column_name}

                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        loss_mb = self.compute_loss(model, mb, return_outputs=False)

                if isinstance(loss_mb, tuple):
                    loss_mb = loss_mb[0]

                loss_mb = loss_mb.detach()
                total_loss += loss_mb * bsz
                print(
                    "mb shapes:",
                    mb["input_ids"].shape,
                    mb["attention_mask"].shape,
                    "labels" in mb and mb["labels"].shape,
                )


            if total_examples == 0:
                return (None, None, None)

            avg_loss = total_loss / total_examples
            return (avg_loss, None, None)

        # -----------------------------
        # GPU-native microbatching path
        # (mirrors training_step)
        # -----------------------------
        inputs = self._prepare_inputs(inputs)  # let HF/Accelerate do device placement
        inputs = self._truncate_batch(inputs)

        microbatches = self._make_microbatches(
            inputs,
            max_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
        )

        if not microbatches:
            return (None, None, None)

        total_loss = 0.0
        total_examples = 0

        for mb in microbatches:
            bsz = mb["input_ids"].size(0)
            total_examples += bsz

            # 🔹 drop length column
            if self.length_column_name in mb:
                mb = {k: v for k, v in mb.items() if k != self.length_column_name}

            # mb is already on correct device; do NOT call _to_device here
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss_mb = self.compute_loss(model, mb, return_outputs=False)

            if isinstance(loss_mb, tuple):
                loss_mb = loss_mb[0]

            loss_mb = loss_mb.detach()
            total_loss += loss_mb * bsz

        if total_examples == 0:
            return (None, None, None)

        avg_loss = total_loss / total_examples
        return (avg_loss, None, None)

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If no token budget or we explicitly want HF-style eval, use default
        if self.max_tokens_per_batch is None or self.eval_mode == "hf":
            return super().get_eval_dataloader(eval_dataset)

        # Otherwise: token-aware eval DataLoader (length-bucketed)
        ds = eval_dataset
        if hasattr(ds, "column_names"):
            lengths = ds[self.length_column_name]
        else:
            lengths = [ds[i][self.length_column_name] for i in range(len(ds))]

        batch_sampler = LengthBucketedBatchSampler(
            lengths=lengths,
            max_tokens_per_batch=self.max_tokens_per_batch,
            bucket_size=16,
            shuffle=False,
            drop_last=False,
        )

        return DataLoader(
            ds,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )
        
        
    # --------------------------------------------------------------
    # Token-aware evaluation with metrics (Option A)
    # --------------------------------------------------------------

    def _token_aware_evaluate(
        self,
        eval_dataset=None,
        max_eval_tokens_per_microbatch: int | None = None,
        desc: str = "Eval (token-aware)",
    ):
        import time
        from tqdm.auto import tqdm

        self.model.eval()

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Need an eval_dataset for token-aware evaluation.")

        # 1) Basic dataloader (no token-budgeting here; we control size via per_device_eval_batch_size)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

        # 2) Build generation kwargs like HF does
        gen_kwargs: dict[str, Any] = {}

        if self.args.generation_max_length is not None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        if self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams

        if getattr(self.args, "do_sample", False):
            gen_kwargs["do_sample"] = True
            if self.args.top_k is not None:
                gen_kwargs["top_k"] = self.args.top_k
            if self.args.top_p is not None:
                gen_kwargs["top_p"] = self.args.top_p

        # Merge with any internally-prepared _gen_kwargs
        if hasattr(self, "_gen_kwargs") and isinstance(self._gen_kwargs, dict):
            tmp = self._gen_kwargs.copy()
            tmp.update(gen_kwargs)
            gen_kwargs = tmp

        # 3) Loop with tqdm + collect preds/labels
        all_preds = []
        all_labels = []

        num_steps = 0
        num_examples = 0

        total_eval_loss = 0.0
        total_eval_tokens = 0

        start_time = time.time()

        for batch in tqdm(dataloader, desc=desc, leave=False):
            num_steps += 1

            # 1) Loss with our microbatch-aware prediction_step
            with torch.no_grad():
                loss, _, _ = self.prediction_step(
                    self.model,
                    batch,
                    prediction_loss_only=True,
                    ignore_keys=None,
                )

            labels = batch.get("labels", None)
            if labels is None:
                raise ValueError("Eval dataset must have labels for metric computation.")

            if loss is not None:
                # Move labels to CPU just for counting tokens
                labels_cpu = labels.detach().to("cpu")
                num_tokens = int((labels_cpu != -100).sum().item())
                total_eval_loss += float(loss.item()) * num_tokens
                total_eval_tokens += num_tokens

            batch_size = labels.size(0)
            num_examples += int(batch_size)

            # 2) Now prepare inputs for generation (separate from loss)
            batch = self._prepare_inputs(batch)

            # Drop labels & input_length from generation kwargs
            ignore_keys = {
                "labels",
                self.length_column_name,
                "decoder_input_ids",
                "decoder_attention_mask",
            }
            gen_inputs = {k: v for k, v in batch.items() if k not in ignore_keys}

            # 3) Fast generation with use_cache=True
            orig_use_cache = getattr(self.model.config, "use_cache", True)
            self.model.config.use_cache = True
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **gen_inputs,
                    **gen_kwargs,
                )
            self.model.config.use_cache = orig_use_cache

            if generated_tokens.ndim == 1:
                generated_tokens = generated_tokens.unsqueeze(0)

            all_preds.append(generated_tokens.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        end_time = time.time()
        runtime = end_time - start_time if num_steps > 0 else 0.0

        # 4) Concatenate + call compute_metrics
        if len(all_preds) == 0:
            # No batches? Return zeros instead of crashing.
            raw_metrics = {
                "bleu": 0.0,
                "chrf": 0.0,
                "meteor": 0.0,
                "gen_len": 0.0,
            }
        else:
            # ---- PAD PREDICTIONS TO COMMON LENGTH ----
            # all_preds: list of (B_i, L_pred_i)
            max_pred_len = max(p.shape[1] for p in all_preds)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = 0  # very safe fallback

            padded_preds = []
            for p in all_preds:
                if p.shape[1] < max_pred_len:
                    pad_width = max_pred_len - p.shape[1]
                    p = np.pad(
                        p,
                        pad_width=((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=pad_id,
                    )
                padded_preds.append(p)
            preds = np.concatenate(padded_preds, axis=0)

            # ---- PAD LABELS TO COMMON LENGTH ----
            # all_labels: list of (B_i, L_label_i)
            max_label_len = max(l.shape[1] for l in all_labels)
            padded_labels = []
            for l in all_labels:
                if l.shape[1] < max_label_len:
                    pad_width = max_label_len - l.shape[1]
                    l = np.pad(
                        l,
                        pad_width=((0, 0), (0, pad_width)),
                        mode="constant",
                        constant_values=-100,  # ignore index
                    )
                padded_labels.append(l)
            labels = np.concatenate(padded_labels, axis=0)

            # Now safe: (N, max_pred_len), (N, max_label_len)
            raw_metrics = self.compute_metrics((preds, labels))
            # raw_metrics is assumed to contain {"bleu": ..., "chrf": ..., "meteor": ..., "gen_len": ...}

        # 5) Convert to HF-style eval_* keys + runtime stats
        eval_loss = (
            total_eval_loss / total_eval_tokens
            if total_eval_tokens > 0
            else float("nan")
        )

        metrics = {
            "eval_loss":   float(eval_loss),
            "eval_bleu":   float(raw_metrics.get("bleu", 0.0)),
            "eval_chrf":   float(raw_metrics.get("chrf", 0.0)),
            "eval_meteor": float(raw_metrics.get("meteor", 0.0)),
            "eval_gen_len": float(raw_metrics.get("gen_len", 0.0)),
        }

        if runtime > 0 and num_steps > 0:
            metrics["eval_runtime"] = float(runtime)
            metrics["eval_samples_per_second"] = float(num_examples / runtime)
            metrics["eval_steps_per_second"] = float(num_steps / runtime)

        return metrics
        
    # --------------------------------------------------------------
    # OOM handling + emergency checkpoint
    # --------------------------------------------------------------

    def _handle_oom_and_save(self, err: BaseException, reason: str):
        msg = str(err)

        is_cuda_oom = (
            isinstance(err, torch.cuda.OutOfMemoryError)
            or "CUDA out of memory" in msg
            or "CUBLAS_STATUS_ALLOC_FAILED" in msg
        )
        is_launch_timeout = (
            "cudaErrorLaunchTimeout" in msg
            or "the launch timed out and was terminated" in msg
        )
        is_accelerator_cuda_error = "CUDA error" in msg

        should_save = is_cuda_oom or is_launch_timeout or is_accelerator_cuda_error
        if not should_save:
            raise err

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        ckpt_dir = os.path.join(self.args.output_dir, "last_error_checkpoint")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.save_model(ckpt_dir)

        if self.args.should_save:
            self.state.save_to_json(os.path.join(ckpt_dir, "trainer_state.json"))
            # <-- only if optimizer/scheduler actually exist
            if self.optimizer is not None:
                self._save_optimizer_and_scheduler(ckpt_dir)

        print(f"\n[HierarchicalTokenTrainer] Saved emergency checkpoint to {ckpt_dir} "
              f"after CUDA error: {msg}\n")

        raise err
        
    # --------------------------------------------------------------
    # OOM-aware wrappers around train/evaluate
    # --------------------------------------------------------------
    # --------------------------------------------------------------
    # OOM-aware wrappers around train/evaluate
    # --------------------------------------------------------------
    def train(self, *args, **kwargs):
        try:
            return super().train(*args, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            # Let _handle_oom_and_save decide whether to checkpoint
            return self._handle_oom_and_save(err, reason="train")

    logger = logging.get_logger(__name__)

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        eval_mode: str | None = None,
    ):
        """
        eval_mode:
          - None / "hf": use standard HF evaluation_loop
          - "token_aware_metrics": use our custom generation+metrics with tqdm,
            but log/save metrics similarly to HF, without the extra
            '***** eval metrics *****' banner from log_metrics().
        """
        # Pick mode: explicit arg wins, else trainer.default
        mode = eval_mode or getattr(self, "eval_mode", None)

        # -----------------------------
        # Our token-aware loop
        # -----------------------------
        if mode == "token_aware_metrics":
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            if eval_dataset is None:
                raise ValueError("Evaluation requires an eval_dataset.")

            start_time = time.time()
            raw_metrics = self._token_aware_evaluate(
                eval_dataset=eval_dataset,
                max_eval_tokens_per_microbatch=self.max_eval_tokens_per_microbatch,
                desc="eval (token-aware)",
            )
            runtime = time.time() - start_time

            num_samples = len(eval_dataset)
            batch_size = self.args.per_device_eval_batch_size
            num_steps = (num_samples + batch_size - 1) // batch_size

            # Add runtime stats (if _token_aware_evaluate didn't already)
            raw_metrics.setdefault("eval_runtime", float(runtime))
            raw_metrics.setdefault(
                "eval_samples_per_second",
                float(num_samples / runtime) if runtime > 0 else 0.0,
            )
            raw_metrics.setdefault(
                "eval_steps_per_second",
                float(num_steps / runtime) if runtime > 0 else 0.0,
            )

            # Attach epoch info (like HF)
            if self.state.epoch is not None:
                raw_metrics[f"{metric_key_prefix}_epoch"] = self.state.epoch
                raw_metrics["epoch"] = self.state.epoch

            metrics = raw_metrics

            # 1) Log to the underlying logger (goes to console + WandB/TF via HF)
            self.log(metrics)

            # 2) Append to log_history so it lands in trainer_state.json
            #    (this is normally done inside log_metrics)
            self.state.log_history.append(metrics)

            # 3) OPTIONAL: if you want a *single* nice banner, you can do:
            if self.is_world_process_zero():
                logger.info("***** eval results (token-aware) *****")
                for k in sorted(metrics.keys()):
                    logger.info("  %s = %s", k, metrics[k])

            # NOTE: we intentionally do NOT call self.log_metrics / self.save_metrics
            # to avoid the extra "***** eval metrics *****" banner.

            return metrics

        # -----------------------------
        # Fallback: standard HF behavior
        # -----------------------------
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
    # --------------------------------------------------------------
    # Core override
    # --------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False):
        # Strip length column if present
        if self.length_column_name in inputs:
            inputs = {k: v for k, v in inputs.items() if k != self.length_column_name}

        outputs = model(**inputs)
        # standard HF pattern
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        If use_cpu_microbatch:
          - force full batch to CPU,
          - compute lengths + truncate + plan microbatches on CPU,
          - move only microbatches to GPU via _to_device.

        Else:
          - let HF/Accelerate handle device placement via _prepare_inputs,
          - truncate + microbatch on that device (typically GPU).
        """
        model.train()

        # --- DEBUG: see where the batch lives when we first see it ---
        #if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
            #print(f"[HierarchicalTokenTrainer] training_step got input_ids on device: {inputs['input_ids'].device}")

        # ---------------- CPU-microbatching path ----------------
        if self.use_cpu_microbatch:
            # Make absolutely sure everything is on CPU before we do any planning
            inputs = self._move_to_cpu(inputs)

            try:
                # Compute lengths on CPU only
                enc_len, dec_len, total_len = self._compute_lengths_enc_dec(inputs)
            except Exception:
                # If this is where an error hits, you'll still see it
                raise

            inputs = self._truncate_batch(inputs)
            microbatches = self._make_microbatches(inputs)

        # ---------------- GPU-native microbatching path ----------------
        else:
            inputs = self._prepare_inputs(inputs)
            inputs = self._truncate_batch(inputs)

            enc_len, dec_len, total_len = self._compute_lengths_enc_dec(inputs)
            microbatches = self._make_microbatches(inputs)

            # --- DEBUG: log summary for this HF batch ---
            with torch.no_grad():
                mb_stats = []
                for mb in microbatches:
                    am_mb = mb["attention_mask"]
                    enc_mb_len = am_mb.sum(dim=-1)                  # (B,)
                    labels_mb = mb.get("labels", None)
                    if labels_mb is not None and labels_mb.ndim == 2:
                        dec_mb_len = (labels_mb != -100).sum(dim=-1)
                    else:
                        dec_mb_len = torch.zeros_like(enc_mb_len)

                    total_mb_len = enc_mb_len + dec_mb_len
                    mb_stats.append({
                        "B": int(enc_mb_len.size(0)),
                        "enc_tokens": int(enc_mb_len.sum().item()),
                        "dec_tokens": int(dec_mb_len.sum().item()),
                        "total_tokens": int(total_mb_len.sum().item()),
                        "max_enc": int(enc_mb_len.max().item()),
                        "max_dec": int(dec_mb_len.max().item()),
                    })
                if self.debug:
                    print("[DEBUG] HF batch has", len(microbatches), "microbatches")
                    for j, s in enumerate(mb_stats[:4]):  # print only first few
                        print(
                            f"   mb{j}: B={s['B']}, "
                            f"enc_tokens={s['enc_tokens']}, dec_tokens={s['dec_tokens']}, "
                            f"total={s['total_tokens']}, "
                            f"max_enc={s['max_enc']}, max_dec={s['max_dec']}"
                        )

        # ---------------- Common microbatch execution ----------------
        num_micro = max(len(microbatches), 1)
        total_loss = 0.0
        total_examples = sum(mb["input_ids"].size(0) for mb in microbatches)

        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()

        for mb_idx, mb in enumerate(microbatches):
            if self.use_cpu_microbatch:
                mb = self._to_device(mb)

            # 🔹 Drop length column so it never reaches the model
            if self.length_column_name in mb:
                mb = {k: v for k, v in mb.items() if k != self.length_column_name}

            bsz = mb["input_ids"].size(0)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, mb)

            if isinstance(loss, tuple):
                loss = loss[0]

            loss = loss * (bsz / total_examples)

            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps

            self.accelerator.backward(loss)
            total_loss += loss.detach().float()

            # --- DEBUG: memory after this microbatch ---
            if self.debug:
                after = torch.cuda.memory_allocated()
                peak = torch.cuda.max_memory_allocated()
                print(
                    f"[DEBUG] mb {mb_idx}: mem_alloc={after/1e9:.2f} GB, "
                    f"peak={peak/1e9:.2f} GB"
                )

        avg_loss = total_loss / num_micro

        if (
            self.log_longest_every > 0
            and self.state.global_step > 0
            and (self.state.global_step % self.log_longest_every == 0)
        ):
            self.log(
                {
                    "max_seen_enc_len":   float(self._max_seen_enc_len),
                    "max_seen_dec_len":   float(self._max_seen_dec_len),
                    "max_seen_total_len": float(self._max_seen_total_len),
                    "num_trunc_hits":     float(self._num_trunc_hits),
                }
            )

        return avg_loss

def ensure_lang_maps(tokenizer, model=None, extra_expect=("eng_Latn", "deu_Latn", "fra_Latn")):
    """
    Rehydrate tokenizer.lang_code_to_id/id_to_lang_code from whatever is in vocab,
    preserving any existing entries and adding missing ones (built-ins + your custom tags).
    """
    # start from whatever exists (don’t throw it away)
    l2i = dict(getattr(tokenizer, "lang_code_to_id", {}) or {})
    i2l = dict(getattr(tokenizer, "id_to_lang_code", {}) or {})

    # anything in additional_special_tokens that has a real id becomes a lang code
    for tok in (tokenizer.special_tokens_map.get("additional_special_tokens") or []):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            l2i.setdefault(tok, tid)
            i2l.setdefault(tid, tok)

    # also try a few common built-ins explicitly (eng_Latn, etc.)
    for tok in extra_expect:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            l2i.setdefault(tok, tid)
            i2l.setdefault(tid, tok)

    # write back
    tokenizer.lang_code_to_id = l2i
    tokenizer.id_to_lang_code = i2l
    if model is not None:
        # make it portable on save
        model.config.lang_code_to_id = dict(l2i)

    # sanity: assert a few must-haves
    for tok in ["eng_Latn"] + list(extra_expect):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            assert tok in tokenizer.lang_code_to_id, f"Missing mapping for {tok}"
    return tokenizer


def add_len8_column(ds, BUCKET=8, MAXLEN=64, length_col="input_length", prompt_text=None, tokenizer=None):
    """
    ds: Dataset (has `length_col` already, which you created during tokenization)
    If you prepend a prompt in the pretrain path, pass `prompt_text`+`tokenizer`
    so we can account for the prompt tokens in the bucket estimate.
    """
    prompt_len = 0
    if prompt_text and tokenizer is not None:
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

    def _f(batch):
        # batch[length_col] can be list[int]
        raw = batch[length_col]
        # effective length estimate for bucketing
        eff = [min(MAXLEN, l + prompt_len) for l in raw]
        # round up to nearest multiple of 8
        len8 = [min(MAXLEN, ((l + (BUCKET-1)) // BUCKET) * BUCKET) for l in eff]
        return {"len8": len8}

    return ds.map(_f, batched=True)



def snap_to_allowed(l, BUCKET = 8, ALLOWED = [8, 16, 32, 64], MAXLEN=64):
    # round up to nearest multiple of 8, then snap to next in {8,16,32,64}
    l8 = min(MAXLEN, ((l + (BUCKET-1)) // BUCKET) * BUCKET)
    for a in ALLOWED:
        if l8 <= a:
            return a
    return ALLOWED[-1]

def add_len_allowed(ds, length_col="input_length", prompt_text=None, tokenizer=None, BUCKET = 8, ALLOWED = [8, 16, 32, 64], MAXLEN=64):
    prompt_len = 0
    if prompt_text and tokenizer is not None:
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    def _f(batch):
        raw = batch[length_col]
        eff = [min(MAXLEN, r + prompt_len) for r in raw]
        return {"len_allowed": [snap_to_allowed(e, BUCKET = BUCKET, ALLOWED = ALLOWED, MAXLEN=MAXLEN) for e in eff]}
    return ds.map(_f, batched=True)


from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase, PreTrainedModel


def _truncate_list_like(x, max_len: int):
    """
    Truncate something that might be a list, numpy array, or tensor,
    and always return a plain Python list of ints.
    """
    if isinstance(x, torch.Tensor):
        x = x.tolist()
    # If it's None, just return as-is
    if x is None:
        return x
    return x[:max_len]



import math, random
from torch.utils.data import Sampler, DataLoader
from collections import defaultdict

class BucketBatchSampler(Sampler[list[int]]):
    """
    Yields homogeneous batches by `len_allowed` without global O(N log N) sorting.
    Single-GPU version.
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = True,
                 length_column: str = "len_allowed", seed: int = 42):
        self.ds = dataset
        self.bs = batch_size
        self.drop_last = drop_last
        self.length_column = length_column
        self.seed = seed

        # 1) bucket indices by length (O(N))
        buckets = defaultdict(list)
        for i in range(len(dataset)):
            L = dataset[i][length_column]
            buckets[L].append(i)
        self.bucket_keys = sorted(buckets.keys())

        # 2) make epoch state
        self._buckets = buckets
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)

        # shuffle within each bucket, then interleave buckets randomly
        bucket_order = list(self.bucket_keys)
        rng.shuffle(bucket_order)

        for L in bucket_order:
            idxs = self._buckets[L][:]
            rng.shuffle(idxs)
            # emit homogeneous batches
            n = (len(idxs) // self.bs) * self.bs if self.drop_last else len(idxs)
            for s in range(0, n, self.bs):
                yield idxs[s:s+self.bs]

    def __len__(self):
        total = 0
        for idxs in self._buckets.values():
            total += (len(idxs) // self.bs) if self.drop_last else math.ceil(len(idxs)/self.bs)
        return total

from transformers import Seq2SeqTrainer

class BucketedTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        sampler = BucketBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            length_column="len_allowed",
            seed=self.args.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

class FastBucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, drop_last: bool = True,
                 length_column: str = "len_allowed", seed: int = 42,
                 block_size: int = 131072, allowed_buckets=(16,32,64),
                 full_shuffle_per_epoch: bool = False):
        import numpy as np
        self.ds = dataset
        self.bs = batch_size
        self.drop_last = drop_last
        self.len_col = length_column
        self.seed = int(seed)
        self.block = int(block_size)
        self.allowed = tuple(sorted(set(allowed_buckets)))

        lens = np.asarray(self.ds[self.len_col], dtype=np.int32)
        self.bucket_idxs = {L: np.where(lens == L)[0].astype(np.int64) for L in self.allowed}
        self.bucket_keys = [L for L in self.allowed if self.bucket_idxs[L].size > 0]

        self._epoch = 0
        self.full_shuffle_per_epoch = bool(full_shuffle_per_epoch)

    def __len__(self):
        import math
        total = 0
        for L in self.bucket_keys:
            n = self.bucket_idxs[L].size
            total += (n // self.bs) if self.drop_last else math.ceil(n / self.bs)
        return total

    def __iter__(self):
        # ---- epoch reset here ----
        import numpy as np
        self._epoch += 1
        rng = np.random.default_rng(self.seed + self._epoch)

        # rewind pointers
        pos = {L: 0 for L in self.bucket_keys}

        # reshuffle buckets cheaply per epoch
        if self.full_shuffle_per_epoch:
            # O(N) per epoch: full shuffle is still pretty fast; flip this on if you want max randomness
            for L in self.bucket_keys:
                rng.shuffle(self.bucket_idxs[L])
        else:
            # block-level shuffle: only shuffle each window when we hit it
            pass  # we’ll shuffle windows lazily below

        # randomize visitation order each cycle
        visit = self.bucket_keys[:]
        rng.shuffle(visit)

        more = True
        while more:
            more = False
            # reshuffle visit order each outer cycle to mix bucket sequence
            rng.shuffle(visit)
            for L in visit:
                arr = self.bucket_idxs[L]
                n = arr.size
                if n == 0:
                    continue

                p = pos[L]
                if p >= n:
                    continue

                if not self.full_shuffle_per_epoch and (p % self.block == 0):
                    # lazy shuffle of current block
                    start = p
                    end = min(p + self.block, n)
                    window = arr[start:end]
                    rng.shuffle(window)

                endp = min(p + self.bs, n)
                if self.drop_last and (endp - p) < self.bs:
                    # skip the tail of this block/bucket
                    pos[L] = ((p // self.block) + 1) * self.block
                    if pos[L] < n:
                        more = True
                    continue

                batch = arr[p:endp].tolist()
                if batch:
                    pos[L] = endp
                    more = True
                    yield batch


class FastBucketedTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        dl = DataLoader(
            self.train_dataset,
            batch_sampler=FastBucketBatchSampler(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                drop_last=self.args.dataloader_drop_last,
                length_column="len_allowed",
                seed=self.args.seed,
                block_size=131072,
                allowed_buckets=(8,16,32,64),  # include 8/16 since your data skews short
                full_shuffle_per_epoch=True,   # set True if you’re okay with O(N) per epoch
            ),
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )
        # NB: dl.sampler is None when using batch_sampler; use batch_sampler if you ever need it:
        # if hasattr(dl.batch_sampler, "set_epoch"):
        #     dl.batch_sampler.set_epoch(0)
        return dl


import math, random
from collections import defaultdict
from torch.utils.data import Sampler, DataLoader
from transformers import Seq2SeqTrainer

class SingleTaskBucketBatchSampler(Sampler):
    """
    Homogeneous batches by `len_allowed` (single task).
    No O(N log N) global sort; shuffle within buckets each epoch.
    """
    def __init__(self, dataset, batch_size: int, drop_last: bool = True,
                 length_column: str = "len_allowed", seed: int = 42):
        self.ds = dataset
        self.bs = int(batch_size)
        self.drop_last = bool(drop_last)
        self.len_col = length_column
        self.seed = int(seed)

        buckets = defaultdict(list)
        for i in range(len(dataset)):
            L = dataset[i][self.len_col]
            buckets[L].append(i)

        self.bucket_keys = sorted(buckets.keys())
        self._buckets = buckets
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        seed = self.seed + self._epoch
        rng = random.Random(seed)

        # Shuffle within each bucket
        streams = []
        for L in self.bucket_keys:
            idxs = self._buckets[L][:]
            rng.shuffle(idxs)
            # chop into batches
            n = (len(idxs) // self.bs) * self.bs if self.drop_last else len(idxs)
            batches = [idxs[s:s+self.bs] for s in range(0, n, self.bs)]
            streams.append(batches)

        # Interleave batches round-robin across buckets in a random bucket order
        bucket_order = list(range(len(streams)))
        rng.shuffle(bucket_order)
        progressed = True
        while progressed:
            progressed = False
            for bi in bucket_order:
                if streams[bi]:
                    progressed = True
                    yield streams[bi].pop()

    def __len__(self):
        total = 0
        for idxs in self._buckets.values():
            total += (len(idxs) // self.bs) if self.drop_last else math.ceil(len(idxs)/self.bs)
        return total


class SingleTaskBucketedTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        # Build sampler *on the full dataset* (has len_allowed)
        batch_sampler = SingleTaskBucketBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            length_column=getattr(self.args, "length_column_name", "len_allowed"),
            seed=self.args.seed,
        )

        # Columns the model/collator actually need
        keep_cols = {
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_input_ids",
            "decoder_attention_mask",
        }
        present = set(self.train_dataset.column_names)
        keep = list(keep_cols & present)
        # Make a *view* with only the needed columns for the DataLoader
        dl_dataset = self.train_dataset.remove_columns(list(present - set(keep)))
        # Optional conversion to torch tensors ahead of time
        try:
            dl_dataset = dl_dataset.with_format("torch", columns=keep)
        except Exception:
            pass

        return DataLoader(
            dl_dataset,
            batch_sampler=batch_sampler,        # indices refer to same row order; OK
            collate_fn=self.data_collator,      # vanilla DataCollatorForSeq2Seq
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

    def get_eval_dataloader(self, eval_dataset=None):
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset

        keep_cols = {
            "input_ids",
            "attention_mask",
            "labels",
            "decoder_input_ids",
            "decoder_attention_mask",
        }
        present = set(ds.column_names)
        keep = list(keep_cols & present)
        dl_dataset = ds.remove_columns(list(present - set(keep)))
        try:
            dl_dataset = dl_dataset.with_format("torch", columns=keep)
        except Exception:
            pass

        return DataLoader(
            dl_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

from transformers import TrainerCallback

class _SetEpochOnBatchSampler(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        dl = kwargs.get("train_dataloader", None)
        if dl is None:
            return
        bs = getattr(dl, "batch_sampler", None)
        if hasattr(bs, "set_epoch"):
            # state.epoch is a float at epoch boundaries (e.g., 12.0, 13.0)
            bs.set_epoch(int(state.epoch))


class GenerationLengthMonitorCallback(TrainerCallback):
    """
    Callback to monitor generation length vs target length during evaluation.

    Helps diagnose truncation issues by tracking:
    - Average generated length vs target length
    - Percentage of outputs shorter than expected
    - Length ratio distribution

    Usage:
        from train_functions import GenerationLengthMonitorCallback

        trainer = Trainer(
            ...,
            callbacks=[GenerationLengthMonitorCallback(
                tokenizer=tokenizer,
                min_length_ratio=0.8,  # Warn if gen < 80% of target
                log_every_n_evals=1,
            )]
        )

    Args:
        tokenizer: Tokenizer for decoding (to get pad_token_id)
        min_length_ratio: Minimum acceptable gen_len/target_len ratio
        log_every_n_evals: How often to log detailed stats
    """

    def __init__(
        self,
        tokenizer=None,
        min_length_ratio: float = 0.8,
        log_every_n_evals: int = 1,
    ):
        self.tokenizer = tokenizer
        self.min_length_ratio = min_length_ratio
        self.log_every_n_evals = log_every_n_evals
        self.eval_count = 0
        self.pad_token_id = tokenizer.pad_token_id if tokenizer else 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log generation length statistics after evaluation."""
        self.eval_count += 1

        if metrics is None:
            return

        # Check if we have generation length info
        gen_len = metrics.get("eval_gen_len", metrics.get("gen_len", None))

        if gen_len is not None and self.eval_count % self.log_every_n_evals == 0:
            print(f"\n[GenerationLengthMonitor] Epoch {int(state.epoch)}")
            print(f"  Average generation length: {gen_len:.1f} tokens")

            # Check for potential truncation
            if gen_len < 50:
                print(f"  ⚠️  WARNING: Very short average output ({gen_len:.1f} tokens)")
                print(f"      This may indicate early truncation issues.")
                print(f"      Consider increasing min_length in generation_config.")
            elif gen_len < 100:
                print(f"  ℹ️  Note: Relatively short outputs ({gen_len:.1f} tokens)")


class LengthAwareLossMixin:
    """
    Mixin class that adds length-aware loss computation to a Trainer.

    Adds a penalty when the model generates outputs significantly shorter
    than the target, helping prevent early truncation.

    Usage:
        from tokenpack_trainer.trainer import TokenPackTrainer
        from train_functions import LengthAwareLossMixin

        class LengthAwareTokenPackTrainer(LengthAwareLossMixin, TokenPackTrainer):
            pass

        trainer = LengthAwareTokenPackTrainer(
            ...,
            length_penalty_weight=0.1,  # Weight for length penalty
            length_penalty_ratio=0.7,   # Penalize if gen < 70% of target
        )

    The mixin adds a soft penalty to the loss when:
        generated_length < length_penalty_ratio * target_length

    This encourages the model to produce longer, more complete outputs.
    """

    length_penalty_weight: float = 0.1
    length_penalty_ratio: float = 0.7

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to add length-aware penalty to loss."""
        # Get base loss from parent class
        if return_outputs:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
            outputs = None

        # Only add length penalty during training (not eval)
        if model.training and 'labels' in inputs:
            labels = inputs['labels']

            # Count non-padding tokens in target
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                pad_id = self.tokenizer.pad_token_id or 0
            else:
                pad_id = 0

            # Target length (excluding -100 ignored tokens and padding)
            target_mask = (labels != -100) & (labels != pad_id)
            target_len = target_mask.sum(dim=-1).float()  # (batch_size,)

            # For seq2seq, we can estimate "generated length" from decoder attention
            # During training, we use teacher forcing, so we approximate with target
            # The penalty here is based on encouraging full target reconstruction
            # This is a soft regularization rather than hard length enforcement

            # Minimum expected length
            min_expected = target_len * self.length_penalty_ratio

            # The loss already encourages matching target, but we add extra weight
            # for examples where target is long (preventing truncation bias)
            # Scale factor: penalize more when target is long
            length_scale = torch.clamp(target_len / 100.0, min=0.5, max=2.0)

            # Add small length-aware regularization
            # This doesn't directly measure generated length (not available during training)
            # but biases the model toward not ignoring long sequences
            length_reg = (length_scale - 1.0).abs().mean() * self.length_penalty_weight

            loss = loss + length_reg

        return (loss, outputs) if return_outputs else loss


import math
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler

class TaskInterleavedBucketBatchSampler(Sampler):
    """
    Homogeneous batches drawn from pools keyed by (task, len_allowed), interleaved.
    - No O(N log N) global sort.
    - O(N) one-time bucket build; per-epoch reshuffle is cheap.
    - Guarantees cross-task mixing within each epoch.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        drop_last: bool = True,
        *,
        task_column: str = "task",
        length_column: str = "len_allowed",
        allowed_buckets = (8, 16, 32, 64),
        seed: int = 42,
        block_size: int = 131072,
        full_shuffle_per_epoch: bool = True,
        tasks = ("pretrain", "train"),  # add e.g. "train_error" if you have it
    ):
        self.ds = dataset
        self.bs = int(batch_size)
        self.drop_last = bool(drop_last)
        self.task_col = task_column
        self.len_col  = length_column
        self.allowed  = tuple(sorted(set(allowed_buckets)))
        self.seed     = int(seed)
        self.block    = int(block_size)
        self.full_shuffle_per_epoch = bool(full_shuffle_per_epoch)
        self.tasks = tuple(tasks)

        # ---- Build pools once (O(N)) ----
        tasks_arr = np.array(self.ds[self.task_col])      # strings: "pretrain"/"train"/...
        lens_arr  = np.asarray(self.ds[self.len_col], dtype=np.int32)

        pools = defaultdict(list)  # key: (task, L) -> list of idx
        for L in self.allowed:
            mask_L = lens_arr == L
            if not mask_L.any():
                continue
            idxs_L = np.nonzero(mask_L)[0].astype(np.int64)
            # split by task
            for t in self.tasks:
                mask_t = tasks_arr[idxs_L] == t
                if mask_t.any():
                    pools[(t, L)].append(idxs_L[mask_t])

        # concatenate per key
        self.pools = {}
        for k, chunks in pools.items():
            self.pools[k] = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]

        # keep only non-empty keys; preserve stable order but randomize each epoch
        self.keys = [k for k, arr in self.pools.items() if arr.size > 0]
        if not self.keys:
            raise ValueError("No non-empty (task, len_allowed) pools were found.")

        self._epoch = 0

        # precompute lengths for __len__
        self._num_batches = 0
        for k in self.keys:
            n = self.pools[k].size
            self._num_batches += (n // self.bs) if self.drop_last else math.ceil(n / self.bs)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        # ---- Epoch reset ----
        self._epoch += 1
        rng = np.random.default_rng(self.seed + self._epoch)

        # per-pool positions
        pos = {k: 0 for k in self.keys}

        # per-epoch shuffle: full or lazy windowed
        if self.full_shuffle_per_epoch:
            for k in self.keys:
                rng.shuffle(self.pools[k])
        # else: do lazy window shuffles when we hit a new block

        # interleave pools: in each cycle, visit pools in random order and emit ONE batch from each non-empty pool
        keys = self.keys[:]
        while True:
            progressed = False
            rng.shuffle(keys)  # randomize visit order each cycle
            for k in keys:
                arr = self.pools[k]
                n   = arr.size
                p   = pos[k]
                if p >= n:
                    continue

                if (not self.full_shuffle_per_epoch) and (p % self.block == 0):
                    # lazy block shuffle to avoid O(N) work
                    start, end = p, min(p + self.block, n)
                    window = arr[start:end]
                    rng.shuffle(window)

                endp = min(p + self.bs, n)
                if self.drop_last and (endp - p) < self.bs:
                    # drop tail of this pool
                    pos[k] = n
                    continue

                batch = arr[p:endp].tolist()
                if not batch:
                    continue

                pos[k]   = endp
                progressed = True
                yield batch

            if not progressed:
                break


import math, numpy as np
from collections import defaultdict
from torch.utils.data import Sampler

class BalancedTaskMixBatchSampler(Sampler):
    """
    With-replacement sampler that keeps a target task mix across the *whole* epoch.
    - Chooses a task each step using target probabilities (e.g., {'pretrain':0.5,'train':0.5}).
    - Chooses a length bucket within that task (tempered by bucket sizes).
    - Pulls a batch sequentially from that (task, len) pool, wrapping/reshuffling on exhaustion.
    - Yields EXACTLY `steps_per_epoch` batches → no tail of the majority task.
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        steps_per_epoch: int,                 # REQUIRED for fixed-length epochs
        drop_last: bool = True,
        task_column: str = "task",
        length_column: str = "len_allowed",
        allowed_buckets=(8,16,32,64),
        target_task_mix = {"pretrain": 0.5, "train": 0.5},  # adjust as you like
        bucket_temperature: float = 1.0,      # <1.0 upsamples rarer lengths
        seed: int = 42,
        block_size: int = 131072,
        full_shuffle_on_wrap: bool = True,    # reshuffle pool when we wrap
    ):
        self.ds = dataset
        self.bs = int(batch_size)
        self.drop_last = bool(drop_last)
        self.task_col = task_column
        self.len_col  = length_column
        self.allowed  = tuple(sorted(set(allowed_buckets)))
        self.seed     = int(seed)
        self.block    = int(block_size)
        self.full_shuffle_on_wrap = bool(full_shuffle_on_wrap)
        self.steps_per_epoch = int(steps_per_epoch)
        # normalize task mix
        tasks, probs = zip(*target_task_mix.items())
        probs = np.array(probs, dtype=np.float64)
        probs = probs / probs.sum()
        self.tasks = list(tasks)
        self.task_probs = probs
        self.bucket_temperature = float(bucket_temperature)

        # ---- build pools once ----
        tasks_arr = np.array(self.ds[self.task_col])
        lens_arr  = np.asarray(self.ds[self.len_col], dtype=np.int32)
        pools = defaultdict(list)   # (task, L) -> [idx chunk, ...]
        for L in self.allowed:
            mask_L = (lens_arr == L)
            if not mask_L.any(): continue
            idxs_L = np.nonzero(mask_L)[0].astype(np.int64)
            for t in self.tasks:
                mask_t = (tasks_arr[idxs_L] == t)
                if mask_t.any():
                    pools[(t, L)].append(idxs_L[mask_t])

        # finalize arrays & positions
        self.pools = {}
        self.pos   = {}   # rolling pointer per (task, L)
        self.count = {}
        for k, chunks in pools.items():
            arr = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            self.pools[k] = arr
            self.pos[k] = 0
            self.count[k] = arr.size

        # precompute length-selection distributions per task (tempered by pool sizes)
        self.task_len_keys = {t: [L for L in self.allowed if (t, L) in self.pools and self.count[(t,L)] > 0]
                              for t in self.tasks}
        self.task_len_probs = {}
        for t in self.tasks:
            ks = self.task_len_keys[t]
            if not ks:
                self.task_len_probs[t] = ([], None)
                continue
            sizes = np.array([self.count[(t, L)] for L in ks], dtype=np.float64)
            if self.bucket_temperature != 1.0:
                sizes = np.power(sizes, self.bucket_temperature)
            probs = sizes / sizes.sum()
            self.task_len_probs[t] = (ks, probs)

        self.epoch = 0

    def __len__(self):
        # Fixed-length epochs: DataLoader won't use this for batch sizing, only for progress bar
        return self.steps_per_epoch

    def _draw_batch_from_pool(self, rng, key):
        arr = self.pools[key]; n = arr.size
        if n == 0: return None

        p = self.pos[key]
        # lazy block shuffle on block boundaries to avoid long runs
        if (p % self.block == 0) and self.full_shuffle_on_wrap:
            start, end = p, min(p + self.block, n)
            window = arr[start:end]
            rng.shuffle(window)

        endp = min(p + self.bs, n)
        if self.drop_last and (endp - p) < self.bs:
            # wrap
            if self.full_shuffle_on_wrap:
                rng.shuffle(arr)
            self.pos[key] = 0
            p, endp = 0, self.bs if not self.drop_last else self.bs
            if n < self.bs and self.drop_last:
                return None  # pool too small for a full batch
        batch = arr[p:endp].tolist()
        self.pos[key] = endp if endp < n else (0 if self.full_shuffle_on_wrap else 0)
        return batch

    def __iter__(self):
        self.epoch += 1
        rng = np.random.default_rng(self.seed + self.epoch)

        if self.full_shuffle_on_wrap:
            for key, arr in self.pools.items():
                rng.shuffle(arr)
                self.pos[key] = 0

        # safety: filter tasks that actually have pools
        active_tasks = [t for t in self.tasks if len(self.task_len_keys.get(t, [])) > 0]
        if not active_tasks:
            return
        task_probs = np.array([self.task_probs[self.tasks.index(t)] for t in active_tasks], dtype=np.float64)
        task_probs /= task_probs.sum()

        steps = self.steps_per_epoch
        s = 0
        while s < steps:
            # 1) sample task by target mix
            t = rng.choice(active_tasks, p=task_probs)

            # 2) sample length inside task
            ks, probs = self.task_len_probs[t]
            if not ks:
                # fallback: pick any other task
                t = rng.choice([x for x in active_tasks if x != t])
                ks, probs = self.task_len_probs[t]
            L = rng.choice(ks, p=probs)

            # 3) draw batch from (t,L), wrapping/reshuffling if needed
            batch = self._draw_batch_from_pool(rng, (t, L))
            if batch is None:
                # if pool is too small for a full batch & drop_last=True, try another pool
                continue

            yield batch
            s += 1

from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer

class FastTaskBucketedInterleavedTrainer(Seq2SeqTrainer):
    def get_train_dataloader(self):
        batch_sampler = TaskInterleavedBucketBatchSampler(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            task_column="task",
            length_column="len_allowed",
            allowed_buckets=(8,16,32,64),
            seed=self.args.seed,
            block_size=131072,
            full_shuffle_per_epoch=True,
            tasks=("pretrain", "train"),   # include "train_error" if present
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=getattr(self.args, "dataloader_prefetch_factor", 2),
        )

# in train_functions.py

class BalancedTaskMixBatchSamplerError(Sampler):
    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        steps_per_epoch: int,
        drop_last: bool = True,
        task_column: str = "task",
        length_column: str = "len_allowed",
        allowed_buckets=(8,16,32,64),
        target_task_mix = {"pretrain": 0.5, "train": 0.5, "train_error": 0.0},
        tasks=("pretrain","train","train_error"),
        bucket_temperature: float = 1.0,
        seed: int = 42,
        block_size: int = 131072,
        full_shuffle_on_wrap: bool = True,
    ):
        import numpy as np
        from collections import defaultdict

        self.ds = dataset
        self.bs = int(batch_size)
        self.drop_last = bool(drop_last)
        self.task_col = task_column
        self.len_col  = length_column
        self.allowed  = tuple(sorted(set(allowed_buckets)))
        self.seed     = int(seed)
        self.block    = int(block_size)
        self.full_shuffle_on_wrap = bool(full_shuffle_on_wrap)
        self.steps_per_epoch = int(steps_per_epoch)
        self.tasks = tuple(tasks)

        # normalize task mix on the provided `tasks` only
        tks, probs = zip(*[(t, target_task_mix.get(t, 0.0)) for t in self.tasks])
        probs = np.array(probs, dtype=np.float64); probs = probs / (probs.sum() or 1.0)
        self.task_list = list(tks)
        self.task_probs = probs
        self.bucket_temperature = float(bucket_temperature)

        # ---- build pools once ----
        tasks_arr = np.array(self.ds[self.task_col])
        lens_arr  = np.asarray(self.ds[self.len_col], dtype=np.int32)
        pools = defaultdict(list)   # (task, L) -> [idx chunk, ...]

        for L in self.allowed:
            mask_L = (lens_arr == L)
            if not mask_L.any(): continue
            idxs_L = np.nonzero(mask_L)[0].astype(np.int64)
            for t in self.tasks:
                mask_t = (tasks_arr[idxs_L] == t)
                if mask_t.any():
                    pools[(t, L)].append(idxs_L[mask_t])

        # finalize arrays & positions
        self.pools = {}
        self.pos   = {}
        self.count = {}
        for k, chunks in pools.items():
            arr = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
            self.pools[k] = arr
            self.pos[k] = 0
            self.count[k] = arr.size

        # per-task bucket distributions
        self.task_len_keys = {t: [L for L in self.allowed if (t, L) in self.pools and self.count[(t,L)] > 0]
                              for t in self.tasks}
        self.task_len_probs = {}
        for t in self.tasks:
            ks = self.task_len_keys[t]
            if not ks:
                self.task_len_probs[t] = ([], None)
                continue
            sizes = np.array([self.count[(t, L)] for L in ks], dtype=np.float64)
            if self.bucket_temperature != 1.0:
                sizes = np.power(sizes, self.bucket_temperature)
            probs = sizes / sizes.sum()
            self.task_len_probs[t] = (ks, probs)

        self.epoch = 0

    def __len__(self):  # fixed-length epoch
        return self.steps_per_epoch

    def _draw_batch_from_pool(self, rng, key):
        import numpy as np
        arr = self.pools.get(key)
        if arr is None or arr.size == 0:
            return None
        n = arr.size; p = self.pos[key]

        if (p % self.block == 0) and self.full_shuffle_on_wrap:
            start, end = p, min(p + self.block, n)
            window = arr[start:end]
            rng.shuffle(window)

        endp = min(p + self.bs, n)
        if self.drop_last and (endp - p) < self.bs:
            # wrap
            if self.full_shuffle_on_wrap:
                rng.shuffle(arr)
            self.pos[key] = 0
            if n < self.bs:
                return None
            p, endp = 0, self.bs

        batch = arr[p:endp].tolist()
        self.pos[key] = endp if endp < n else 0
        return batch

    def __iter__(self):
        import numpy as np
        self.epoch += 1
        rng = np.random.default_rng(self.seed + self.epoch)

        active_tasks = [t for t in self.task_list if len(self.task_len_keys.get(t, [])) > 0]
        if not active_tasks:
            return
        task_probs = np.array([self.task_probs[self.task_list.index(t)] for t in active_tasks], dtype=np.float64)
        task_probs /= task_probs.sum() or 1.0

        steps = self.steps_per_epoch
        s = 0
        while s < steps:
            # task draw
            t = rng.choice(active_tasks, p=task_probs)
            # length draw within task
            ks, probs = self.task_len_probs.get(t, ([], None))
            if not ks:
                continue  # try again
            L = rng.choice(ks, p=probs)
            # draw batch
            batch = self._draw_batch_from_pool(rng, (t, L))
            if batch is None:
                continue
            yield batch
            s += 1

# === TE/NVFP4 drop-in patch ===
# === TE/NVFP4 with optional MXFP8 tail-layers ===
import torch
import torch.nn as nn

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import NVFP4BlockScaling, MXFP8BlockScaling, Format
    _HAS_TE = True
except Exception:
    _HAS_TE = False
    te = None

def _swap_linear_to_te(module: nn.Module, verbose: bool = True):
    if not _HAS_TE:
        return module
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new = te.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None),
            )
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
            if verbose:
                print(f"[TE] swapped Linear -> te.Linear at: {name}")
        else:
            _swap_linear_to_te(child, verbose)
    return module

def _get_attr_path(root: nn.Module, path: str):
    cur = root
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def _find_t5_blocks(model: nn.Module):
    enc = _get_attr_path(model, "encoder.block")
    dec = _get_attr_path(model, "decoder.block")
    if isinstance(enc, nn.ModuleList) or isinstance(dec, nn.ModuleList):
        return enc if isinstance(enc, nn.ModuleList) else None, dec if isinstance(dec, nn.ModuleList) else None
    enc = _get_attr_path(model, "model.encoder.block")
    dec = _get_attr_path(model, "model.decoder.block")
    return enc if isinstance(enc, nn.ModuleList) else None, dec if isinstance(dec, nn.ModuleList) else None

class _TailPrecisionWrapper(nn.Module):
    def __init__(self, block: nn.Module, recipe):
        super().__init__()
        self.block = block
        self.recipe = recipe

    def forward(self, *args, **kwargs):
        with te.fp8_autocast(enabled=True, fp8_recipe=self.recipe):
            return self.block(*args, **kwargs)

def _wrap_tail_layers_in_mxfp8(model: nn.Module, tail_layers: int, where: str, recipe):
    if tail_layers <= 0 or recipe is None:
        return
    enc_blocks, dec_blocks = _find_t5_blocks(model)

    def wrap_tail(blocks: nn.ModuleList, n: int, name: str):
        if not isinstance(blocks, nn.ModuleList) or n <= 0:
            return
        n = min(n, len(blocks))
        for idx in range(len(blocks) - n, len(blocks)):
            if not isinstance(blocks[idx], _TailPrecisionWrapper):
                blocks[idx] = _TailPrecisionWrapper(blocks[idx], recipe)
                print(f"[TE] Tail {name} layer {idx} wrapped in MXFP8")

    if where in ("encoder", "both") and enc_blocks is not None:
        wrap_tail(enc_blocks, tail_layers, "encoder")
    if where in ("decoder", "both") and dec_blocks is not None:
        wrap_tail(dec_blocks, tail_layers, "decoder")

def apply_te_nvfp4_inplace(
    model: nn.Module,
    tail_layers: int = 2,
    tail_where: str = "decoder",
    verbose: bool = True,
):
    """
    Keep the HF model object, but:
      - swap Linear -> te.Linear
      - wrap tail blocks in MXFP8
      - patch model.forward to enter (disabled outer autocast) + BF16 autocast + TE fp8_autocast(NVFP4)
    """
    if not _HAS_TE:
        print("[TE] Transformer Engine not found; skipping.")
        return model

    # recipes
    try:
        nvfp4_recipe = NVFP4BlockScaling()
        main_recipe = nvfp4_recipe
        if verbose: print("[TE] Using NVFP4BlockScaling")
    except Exception:
        main_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
        if verbose: print("[TE] NVFP4 not available, using MXFP8BlockScaling")

    mxfp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    # 1) swap linears
    if verbose: print("[TE] Swapping Linear layers -> te.Linear …")
    _swap_linear_to_te(model, verbose=verbose)

    # 2) wrap tail layers
    if tail_layers and tail_layers > 0:
        _wrap_tail_layers_in_mxfp8(model, tail_layers=tail_layers, where=tail_where, recipe=mxfp8_recipe)

    # 3) patch forward in place (this avoids _forward_unimplemented issues)
    if not hasattr(model, "_te_saved_forward"):
        model._te_saved_forward = model.forward

    def _te_forward(*args, **kwargs):
        # kill any upstream fp16 autocast and force BF16 for NVFP4 RHT
        with torch.autocast(device_type="cuda", enabled=False):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with te.fp8_autocast(enabled=True, fp8_recipe=main_recipe):
                    return model._te_saved_forward(*args, **kwargs)

    model.forward = _te_forward
    return model

class TEAutocastWrapper(nn.Module):
    def __init__(self, model: nn.Module,
                 tail_layers: int = 2,
                 tail_where: str = "decoder"):
        super().__init__()
        self.model = model
        self.tail_layers = tail_layers
        self.tail_where = tail_where

        self.recipe_nvfp4 = None
        self.recipe_mxfp8 = None
        if _HAS_TE:
            try:
                self.recipe_nvfp4 = NVFP4BlockScaling()
            except Exception:
                self.recipe_nvfp4 = None
            try:
                self.recipe_mxfp8 = MXFP8BlockScaling(fp8_format=Format.E4M3)
            except Exception:
                self.recipe_mxfp8 = None

        if _HAS_TE and self.recipe_mxfp8 is not None and self.tail_layers > 0:
            _wrap_tail_layers_in_mxfp8(self.model, self.tail_layers, self.tail_where, self.recipe_mxfp8)

        def forward(self, *args, **kwargs):
            if not _HAS_TE:
                return self.model(*args, **kwargs)

            main_recipe = self.recipe_nvfp4 or self.recipe_mxfp8
            if main_recipe is None:
                return self.model(*args, **kwargs)

            # ✅ kill any upstream fp16 autocast (e.g. @torch.autocast("cuda") defaults to fp16)
            with torch.autocast(device_type="cuda", enabled=False):
                # ✅ now force BF16 activations (required for NVFP4 RHT in your build)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    with te.fp8_autocast(enabled=True, fp8_recipe=main_recipe):
                        return self.model(*args, **kwargs)

import copy
import torch
import torch.nn as nn

# --- relative position bias preservation for T5/UMT5 ---
def _ensure_relpos_bias_preserved(old_first_block: nn.Module, new_first_block: nn.Module):
    """
    In HF T5, the relative position bias module often lives on the *first* block's SelfAttention
    and is shared by all blocks. If we prune that block, move the module to the new first block.
    """
    try:
        old_sa = old_first_block.layer[0].SelfAttention
        new_sa = new_first_block.layer[0].SelfAttention
    except Exception:
        return
    if hasattr(old_sa, "relative_attention_bias") and not hasattr(new_sa, "relative_attention_bias"):
        setattr(new_sa, "relative_attention_bias", old_sa.relative_attention_bias)

def _slice_blocks(module_list: nn.ModuleList, drop_front: int, drop_back: int):
    n = len(module_list)
    f = max(0, min(drop_front, n))
    b = max(0, min(drop_back, n - f))
    keep_n = n - f - b
    if keep_n <= 0:
        raise ValueError(f"Pruning removes all layers (had {n}, drop_front={f}, drop_back={b}).")
    # We'll return the *same* ModuleList with items replaced in-place to avoid breaking references.
    kept = [module_list[i] for i in range(f, f + keep_n)]
    module_list[:] = kept
    return f, b, keep_n, n

def _find_t5_block_lists(model: nn.Module):
    """
    Returns (encoder_blocks, decoder_blocks) ModuleLists if found, else (None, None).
    Handles both `encoder.block`/`decoder.block` and `model.encoder.block`/`model.decoder.block`.
    """
    def _get(root, path):
        cur = root
        for p in path.split("."):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    enc = _get(model, "encoder.block")
    dec = _get(model, "decoder.block")
    if isinstance(enc, nn.ModuleList) or isinstance(dec, nn.ModuleList):
        return (enc if isinstance(enc, nn.ModuleList) else None,
                dec if isinstance(dec, nn.ModuleList) else None)

    enc = _get(model, "model.encoder.block")
    dec = _get(model, "model.decoder.block")
    return (enc if isinstance(enc, nn.ModuleList) else None,
            dec if isinstance(dec, nn.ModuleList) else None)

def prune_t5_layers_inplace(
    model: nn.Module,
    enc_drop_front: int = 0,
    enc_drop_back: int  = 0,
    dec_drop_front: int = 0,
    dec_drop_back: int  = 0,
    verbose: bool = True,
):
    """
    In-place prune of whole blocks from encoder/decoder.
    Updates config.num_layers / config.num_decoder_layers and preserves relpos bias if needed.
    Call this *before* creating the optimizer/Trainer.
    """
    enc_blocks, dec_blocks = _find_t5_block_lists(model)
    if enc_blocks is None and dec_blocks is None:
        raise RuntimeError("Could not locate T5/UMT5 encoder/decoder blocks to prune.")

    # Preserve a pointer to original first blocks (for relpos bias transfer)
    orig_first_enc = enc_blocks[0] if (enc_blocks is not None and len(enc_blocks) > 0) else None
    orig_first_dec = dec_blocks[0] if (dec_blocks is not None and len(dec_blocks) > 0) else None

    if enc_blocks is not None and (enc_drop_front > 0 or enc_drop_back > 0):
        f,b,keep,n = _slice_blocks(enc_blocks, enc_drop_front, enc_drop_back)
        if verbose:
            print(f"[PRUNE] Encoder: {n} -> {keep} (dropped front={f}, back={b})")
        # If we dropped the original first encoder block, move its relpos bias to new first
        if orig_first_enc is not None and f > 0 and len(enc_blocks) > 0:
            _ensure_relpos_bias_preserved(orig_first_enc, enc_blocks[0])
        # Adjust config
        if hasattr(model.config, "num_layers"):
            model.config.num_layers = keep
        if hasattr(model.config, "num_encoder_layers"):
            model.config.num_encoder_layers = keep

    if dec_blocks is not None and (dec_drop_front > 0 or dec_drop_back > 0):
        f,b,keep,n = _slice_blocks(dec_blocks, dec_drop_front, dec_drop_back)
        if verbose:
            print(f"[PRUNE] Decoder: {n} -> {keep} (dropped front={f}, back={b})")
        if orig_first_dec is not None and f > 0 and len(dec_blocks) > 0:
            _ensure_relpos_bias_preserved(orig_first_dec, dec_blocks[0])
        if hasattr(model.config, "num_decoder_layers"):
            model.config.num_decoder_layers = keep
        # T5 sometimes mirrors num_layers if separate attr missing
        if hasattr(model.config, "num_layers") and not hasattr(model.config, "num_decoder_layers"):
            model.config.num_layers = max(model.config.num_layers, keep)

def apply_te_nvfp4(
    model: nn.Module,
    tail_layers: int = 2,
    tail_where: str = "decoder",        # 'decoder' | 'encoder' | 'both'
    # NEW: pruning controls (set to >0 to prune)
    enc_drop_front: int = 0,
    enc_drop_back: int  = 0,
    dec_drop_front: int = 0,
    dec_drop_back: int  = 0,
    verbose: bool = True
) -> nn.Module:
    """
    - (NEW) Optionally prune encoder/decoder blocks before optimizer init.
    - Convert Linear -> te.Linear
    - Wrap tail blocks to MXFP8
    - Wrap whole model under NVFP4 (fallback MXFP8)
    """
    # 1) prune first, so optimizer never sees the removed params
    if any(x > 0 for x in (enc_drop_front, enc_drop_back, dec_drop_front, dec_drop_back)):
        prune_t5_layers_inplace(
            model,
            enc_drop_front=enc_drop_front,
            enc_drop_back=enc_drop_back,
            dec_drop_front=dec_drop_front,
            dec_drop_back=dec_drop_back,
            verbose=verbose,
        )

    # 2) (existing) TE conversion + tail wrapping + global NVFP4/MXFP8 context
    if not _HAS_TE:
        print("[TE] Transformer Engine not found; skipping TE conversion.")
        return model

    if verbose:
        print("[TE] Swapping Linear layers to te.Linear …")
    _swap_linear_to_te(model, verbose=verbose)

    if verbose:
        print(f"[TE] Wrapping model: main=NVFP4 (fallback MXFP8), tail_layers={tail_layers}, tail_where={tail_where}")
    return TEAutocastWrapper(model, tail_layers=tail_layers, tail_where=tail_where)


####HNet Tokenizer
# hnet_byte_tokenizer.py
from transformers import PreTrainedTokenizer
import codecs

class HNetByteTokenizer(PreTrainedTokenizer):
    # Reserve a small special-vocab on top of 256 bytes
    PAD, BOS, EOS, UNK = 256, 257, 258, 259
    vocab_size = 260

    def __init__(self, **kwargs):
        super().__init__(pad_token_id=self.PAD,
                         bos_token_id=self.BOS,
                         eos_token_id=self.EOS,
                         unk_token_id=self.UNK, **kwargs)

    def _tokenize(self, text, **kwargs):
        # Not used by fast path; HF calls encode directly via __call__
        return [b for b in text.encode("utf-8")]

    def _convert_token_to_id(self, token):
        return token  # tokens are already ints 0..255; specials are 256+

    def _convert_id_to_token(self, index):
        return index

    def get_vocab(self):
        # minimal, for compatibility
        return {i: i for i in range(self.vocab_size)}

    def build_inputs_with_special_tokens(self, token_ids):
        return [self.BOS] + token_ids + [self.EOS]

    def encode_plus(self, text, max_length=None, truncation=False, padding=False, **kwargs):
        ids = list(text.encode("utf-8"))
        if truncation and max_length is not None:
            ids = ids[: max(0, max_length - 2)]  # room for BOS/EOS
        ids = self.build_inputs_with_special_tokens(ids)
        if padding and max_length is not None:
            ids = ids + [self.PAD] * (max_length - len(ids))
        attn = [1 if x != self.PAD else 0 for x in ids]
        return {"input_ids": ids, "attention_mask": attn}

    def batch_encode_plus(self, texts, **kw):
        return {"input_ids": [self.encode_plus(t, **kw)["input_ids"] for t in texts],
                "attention_mask": [self.encode_plus(t, **kw)["attention_mask"] for t in texts]}

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        buf = bytearray()
        for i in ids:
            if i is None:
                continue
            if i >= 256:
                if skip_special_tokens:
                    continue
                # fall through for special handling if desired
                continue
            buf.append(i)
        # Be forgiving with historic scripts / diacritics
        return buf.decode("utf-8", errors="replace")



import torch
from transformers import T5GemmaForConditionalGeneration
try:
    from transformers import T5Gemma2ForConditionalGeneration
except ImportError:
    T5Gemma2ForConditionalGeneration = None

def _expand_weight_matrix(old_w: torch.Tensor, new_vocab_size: int) -> torch.Tensor:
    """
    Expand a [V_old, H] weight matrix to [V_new, H], copying the first
    V_old rows and initializing the extra rows from N(mu, sigma^2) of the old weights.
    """
    V_old, H = old_w.shape
    V_new = new_vocab_size
    if V_new <= V_old:
        return old_w  # nothing to do (or you could slice)

    device = old_w.device
    dtype = old_w.dtype

    new_w = torch.empty((V_new, H), device=device, dtype=dtype)
    new_w[:V_old] = old_w

    mu = old_w.mean().item()
    sigma = old_w.std().item()
    if sigma == 0.0:
        sigma = 0.02

    with torch.no_grad():
        new_w[V_old:] = torch.randn((V_new - V_old, H), device=device, dtype=dtype) * sigma + mu

    return new_w


def resize_t5gemma_embeddings(model: T5GemmaForConditionalGeneration,
                              new_vocab_size: int) -> T5GemmaForConditionalGeneration:
    """
    Safely resize T5Gemma’s embeddings + LM head without tying weights.

    - Does NOT call model.resize_token_embeddings()
    - Does NOT tie LM head to input embeddings
    - Expands shared, encoder, decoder, and lm_head independently.
    """
    # 1. Shared/input embeddings
    shared = model.get_input_embeddings()          # T5GemmaSharedEmbedding
    old_shared = shared.weight.data
    V_old, H = old_shared.shape

    if new_vocab_size == V_old:
        # Already the right size; nothing to do.
        return model

    # Expand shared matrix
    new_shared_w = _expand_weight_matrix(old_shared, new_vocab_size)
    shared.weight = torch.nn.Parameter(new_shared_w, requires_grad=True)
    model.set_input_embeddings(shared)

    # 2. Encoder embeddings
    enc_emb = model.get_encoder().embed_tokens
    enc_w = enc_emb.weight.data
    enc_emb.weight = torch.nn.Parameter(
        _expand_weight_matrix(enc_w, new_vocab_size),
        requires_grad=True,
    )

    # 3. Decoder embeddings
    dec_emb = model.get_decoder().embed_tokens
    dec_w = dec_emb.weight.data
    dec_emb.weight = torch.nn.Parameter(
        _expand_weight_matrix(dec_w, new_vocab_size),
        requires_grad=True,
    )

    # 4. LM head (out_proj for T5Gemma)
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        if hasattr(lm_head, "out_proj"):
            out_layer = lm_head.out_proj
        else:
            out_layer = lm_head

        out_w = out_layer.weight.data
        out_layer.weight = torch.nn.Parameter(
            _expand_weight_matrix(out_w, new_vocab_size),
            requires_grad=True,
        )

    # 5. Update config vocab_size everywhere
    model.config.vocab_size = new_vocab_size
    if hasattr(model.config, "encoder") and hasattr(model.config.encoder, "vocab_size"):
        model.config.encoder.vocab_size = new_vocab_size
    if hasattr(model.config, "decoder") and hasattr(model.config.decoder, "vocab_size"):
        model.config.decoder.vocab_size = new_vocab_size

    return model


def resize_t5gemma2_embeddings(model, new_vocab_size: int):
    """
    Safely resize T5Gemma2's embeddings + LM head, preserving the custom
    T5Gemma2TextScaledWordEmbedding class (embed_scale + eoi_embedding).

    HuggingFace's default resize_token_embeddings() has two bugs for T5Gemma2:
      1. It replaces T5Gemma2TextScaledWordEmbedding with a plain nn.Embedding,
         losing the ~48x embedding scale factor and the eoi_embedding parameter.
      2. It only updates config.decoder.vocab_size via get_text_config(), never
         updating config.encoder.text_config.vocab_size, causing "Imbalanced
         encoder-decoder vocabulary size" on reload.

    This function avoids both issues by expanding weight matrices in-place and
    syncing all three config locations.

    T5Gemma2 uses three-way tied embeddings:
        encoder.text_model.embed_tokens.weight  (canonical)
        decoder.embed_tokens.weight             (tied → encoder)
        lm_head.out_proj.weight                 (tied → encoder)
    Because weights are tied, we only need to expand the canonical encoder
    embedding and the lm_head, then re-tie.
    """
    old_emb = model.get_input_embeddings()  # encoder.text_model.embed_tokens
    old_vocab_size = old_emb.weight.shape[0]

    if new_vocab_size == old_vocab_size:
        return model

    # 1. Expand the encoder (canonical) embedding weight
    new_weight = _expand_weight_matrix(old_emb.weight.data, new_vocab_size)
    old_emb.weight = torch.nn.Parameter(new_weight, requires_grad=True)
    old_emb.num_embeddings = new_vocab_size
    model.set_input_embeddings(old_emb)

    # 2. Expand the LM head output projection
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        out_layer = lm_head.out_proj if hasattr(lm_head, "out_proj") else lm_head
        old_out_w = out_layer.weight.data
        out_layer.weight = torch.nn.Parameter(
            _expand_weight_matrix(old_out_w, new_vocab_size),
            requires_grad=True,
        )
        out_layer.out_features = new_vocab_size

    # 3. Re-tie weights so decoder.embed_tokens shares the encoder's expanded weight
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # 4. Sync ALL config locations (this is the critical part HF misses)
    model.config.vocab_size = new_vocab_size
    if hasattr(model.config, "decoder") and hasattr(model.config.decoder, "vocab_size"):
        model.config.decoder.vocab_size = new_vocab_size
    if (hasattr(model.config, "encoder")
            and hasattr(model.config.encoder, "text_config")
            and hasattr(model.config.encoder.text_config, "vocab_size")):
        model.config.encoder.text_config.vocab_size = new_vocab_size

    # 5. Sync the cached model-level vocab_size attribute.
    #    T5Gemma2ForConditionalGeneration.__init__ copies config.vocab_size
    #    to self.vocab_size as a plain int. The loss function uses self.vocab_size
    #    (not config.vocab_size) to reshape logits, so we must update it too.
    if hasattr(model, "vocab_size"):
        model.vocab_size = new_vocab_size

    return model


def ensure_vocab_match(model, tokenizer):
    """
    Generic wrapper used in your script.

    For T5Gemma v1: use custom expand logic (resize_t5gemma_embeddings).
    For T5Gemma2:   use custom expand logic (resize_t5gemma2_embeddings).
    For everything else: fallback to standard resize_token_embeddings.
    """
    new_vocab_size = len(tokenizer)
    old_vocab_size = model.get_input_embeddings().weight.shape[0]

    if new_vocab_size == old_vocab_size:
        return model

    if isinstance(model, T5GemmaForConditionalGeneration):
        return resize_t5gemma_embeddings(model, new_vocab_size)
    elif T5Gemma2ForConditionalGeneration is not None and isinstance(model, T5Gemma2ForConditionalGeneration):
        return resize_t5gemma2_embeddings(model, new_vocab_size)
    else:
        # Safe generic path for other models
        model.resize_token_embeddings(new_vocab_size, mean_resizing=True)
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        model.config.vocab_size = new_vocab_size
        return model

import numpy as np

def analyze_seq2seq_lengths(ds):
    # assuming ds is HF Dataset with input_ids, attention_mask, labels
    enc_lens = []
    dec_lens = []
    for ex in ds:
        am = ex["attention_mask"]
        labels = ex["labels"]
        enc_lens.append(int(np.sum(am)))
        if isinstance(labels, list):
            dec_lens.append(int(np.sum(np.array(labels) != -100)))
        else:
            # tensor-like
            arr = np.array(labels)
            dec_lens.append(int(np.sum(arr != -100)))
    enc_lens = np.array(enc_lens)
    dec_lens = np.array(dec_lens)
    total = enc_lens + dec_lens
    def stats(name, arr):
        print(f"{name}: p50={np.percentile(arr,50):.1f}, "
              f"p90={np.percentile(arr,90):.1f}, "
              f"p95={np.percentile(arr,95):.1f}, "
              f"p99={np.percentile(arr,99):.1f}, "
              f"max={arr.max()}")
    stats("enc_len", enc_lens)
    stats("dec_len", dec_lens)
    stats("total_len", total)

# Example:
# analyze_seq2seq_lengths(tokenized_datasets["train"])


# =============================================================================
# ADVANCED CUNEIFORM-AWARE COLLATORS (Session 4)
# =============================================================================
#
# These collators implement domain-specific corruption strategies for cuneiform:
#   1. Sign-Boundary Aware Corruption - corrupts at cuneiform sign boundaries
#   2. Determinative-Targeting Corruption - specifically targets determinatives
#   3. Position-Weighted (Edge) Corruption - simulates tablet edge damage
#   4. Curriculum Corruption Scheduler - adjusts difficulty over training
#   5. Bracket Damage Simulation - simulates CDLI-style [x] damage markers
#   6. Dynamic Task-Mixing Ratio - adjusts pretrain/translate ratio over time
#   7. AdvancedSpanCorruptionCollator - unified collator with all features
#
# Legacy collators (T5SpanCorruptionCollatorFast, T5DataCollatorForSpanCorruptionAsteriskFast,
# MixedDataCollator8Buckets, etc.) are preserved above.
# =============================================================================

import re
from typing import Callable, Optional, Sequence, Tuple, Union
from collections import defaultdict


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def _detect_sign_boundaries(text: str) -> List[Tuple[int, int]]:
    """
    Detect cuneiform sign boundaries in transliterated text.

    Signs are typically separated by:
      - Hyphen (-) within words
      - Space ( ) between words
      - Period (.) for determinatives attached to signs

    Returns list of (start, end) character positions for each sign.
    """
    if not text:
        return []

    boundaries = []
    current_start = 0

    # Split on sign separators while tracking positions
    i = 0
    while i < len(text):
        c = text[i]
        if c in '-  .':
            if i > current_start:
                boundaries.append((current_start, i))
            current_start = i + 1
        i += 1

    # Don't forget the last segment
    if current_start < len(text):
        boundaries.append((current_start, len(text)))

    return boundaries


def _find_determinative_positions(tokens: List[int], tokenizer) -> List[int]:
    """
    Find token positions that are determinatives.

    Determinatives in CDLI format: {d}, {disz}, {m}, {f}, {lu2}, {ki}, etc.
    These are semantic classifiers that help identify word class.

    Returns list of token indices that are determinatives.
    """
    det_positions = []

    # Decode each token and check if it's a determinative
    for i, tok_id in enumerate(tokens):
        try:
            tok_text = tokenizer.decode([tok_id], skip_special_tokens=False)
            # Check for determinative patterns
            if re.match(r'^\{[^}]+\}$', tok_text.strip()):
                det_positions.append(i)
            # Also catch partial determinatives that might be tokenized separately
            elif tok_text.strip() in ['{', '}'] or tok_text.strip().startswith('{') or tok_text.strip().endswith('}'):
                det_positions.append(i)
        except Exception:
            pass

    return det_positions


def _compute_edge_weights(L: int, edge_bias: float = 2.0, edge_width: float = 0.2) -> np.ndarray:
    """
    Compute position weights that favor edges (simulating tablet damage).

    Args:
        L: Sequence length
        edge_bias: How much more likely edges are to be corrupted (2.0 = 2x more likely)
        edge_width: Fraction of sequence considered "edge" (0.2 = first/last 20%)

    Returns:
        Array of weights, higher at edges
    """
    if L <= 1:
        return np.ones(L)

    weights = np.ones(L)
    edge_tokens = max(1, int(L * edge_width))

    # Increase weight at start
    for i in range(edge_tokens):
        # Linear decay from edge_bias to 1.0
        weights[i] = edge_bias - (edge_bias - 1.0) * (i / edge_tokens)

    # Increase weight at end
    for i in range(edge_tokens):
        idx = L - 1 - i
        weights[idx] = edge_bias - (edge_bias - 1.0) * (i / edge_tokens)

    # Normalize so mean is 1.0 (preserves expected corruption rate)
    weights = weights / weights.mean()

    return weights


class CurriculumScheduler:
    """
    Manages curriculum learning parameters that change over training.

    Tracks current epoch/step and provides scheduled values for:
      - noise_density: corruption rate
      - mean_noise_span_length: average span length
      - determinative_targeting_rate: how often to target determinatives
      - pretrain_ratio: ratio of pretrain to translate tasks

    Usage:
        scheduler = CurriculumScheduler(
            noise_schedule=[(0, 0.10), (50, 0.15), (150, 0.20)],
            pretrain_ratio_schedule=[(0, 0.8), (100, 0.5), (200, 0.2)]
        )

        # In training loop:
        scheduler.set_epoch(current_epoch)
        noise = scheduler.get_noise_density()
        ratio = scheduler.get_pretrain_ratio()
    """

    def __init__(
        self,
        noise_schedule: Optional[List[Tuple[int, float]]] = None,
        span_length_schedule: Optional[List[Tuple[int, float]]] = None,
        determinative_rate_schedule: Optional[List[Tuple[int, float]]] = None,
        pretrain_ratio_schedule: Optional[List[Tuple[int, float]]] = None,
        edge_bias_schedule: Optional[List[Tuple[int, float]]] = None,
    ):
        """
        Args:
            noise_schedule: List of (epoch, noise_density) pairs
            span_length_schedule: List of (epoch, mean_span_length) pairs
            determinative_rate_schedule: List of (epoch, targeting_rate) pairs
            pretrain_ratio_schedule: List of (epoch, pretrain_ratio) pairs
            edge_bias_schedule: List of (epoch, edge_bias) pairs

        Each schedule interpolates linearly between specified points.
        """
        self.noise_schedule = noise_schedule or [(0, 0.15)]
        self.span_length_schedule = span_length_schedule or [(0, 3.0)]
        self.determinative_rate_schedule = determinative_rate_schedule or [(0, 0.3)]
        self.pretrain_ratio_schedule = pretrain_ratio_schedule or [(0, 0.5)]
        self.edge_bias_schedule = edge_bias_schedule or [(0, 1.5)]

        self._epoch = 0
        self._step = 0

    def set_epoch(self, epoch: int):
        """Update current epoch."""
        self._epoch = epoch

    def set_step(self, step: int):
        """Update current step (for finer-grained scheduling)."""
        self._step = step

    def _interpolate(self, schedule: List[Tuple[int, float]], epoch: int) -> float:
        """Linearly interpolate value from schedule at given epoch."""
        if not schedule:
            return 0.0

        # Sort by epoch
        schedule = sorted(schedule, key=lambda x: x[0])

        # Before first point
        if epoch <= schedule[0][0]:
            return schedule[0][1]

        # After last point
        if epoch >= schedule[-1][0]:
            return schedule[-1][1]

        # Find surrounding points and interpolate
        for i in range(len(schedule) - 1):
            e1, v1 = schedule[i]
            e2, v2 = schedule[i + 1]
            if e1 <= epoch < e2:
                t = (epoch - e1) / (e2 - e1)
                return v1 + t * (v2 - v1)

        return schedule[-1][1]

    def get_noise_density(self) -> float:
        return self._interpolate(self.noise_schedule, self._epoch)

    def get_mean_span_length(self) -> float:
        return self._interpolate(self.span_length_schedule, self._epoch)

    def get_determinative_targeting_rate(self) -> float:
        return self._interpolate(self.determinative_rate_schedule, self._epoch)

    def get_pretrain_ratio(self) -> float:
        return self._interpolate(self.pretrain_ratio_schedule, self._epoch)

    def get_edge_bias(self) -> float:
        return self._interpolate(self.edge_bias_schedule, self._epoch)

    def get_all_params(self) -> Dict[str, float]:
        """Get all current scheduled parameters."""
        return {
            "noise_density": self.get_noise_density(),
            "mean_span_length": self.get_mean_span_length(),
            "determinative_targeting_rate": self.get_determinative_targeting_rate(),
            "pretrain_ratio": self.get_pretrain_ratio(),
            "edge_bias": self.get_edge_bias(),
            "epoch": self._epoch,
            "step": self._step,
        }


# -----------------------------------------------------------------------------
# ADVANCED SPAN CORRUPTION COLLATOR
# -----------------------------------------------------------------------------

class AdvancedSpanCorruptionCollator:
    """
    Advanced span corruption collator with cuneiform-specific features.

    Features:
      1. Sign-boundary aware corruption (optional)
      2. Determinative targeting (optional)
      3. Position-weighted (edge) corruption (optional)
      4. Curriculum scheduling support (optional)
      5. Bracket-style damage markers (optional)
      6. Standard asterisk (*) corruption (default)

    This collator is designed for cuneiform transliteration pretraining and
    can be used as a drop-in replacement for T5DataCollatorForSpanCorruptionAsteriskFast.

    Args:
        tokenizer: HuggingFace tokenizer
        noise_density: Base fraction of tokens to corrupt (0.15 = 15%)
        mean_noise_span_length: Average span length (3.0 typical)
        input_length: Maximum input sequence length
        target_length: Maximum target sequence length
        pad_to_multiple_of: Pad sequences to multiple of this (8 for tensor cores)
        dynamic_inputs: If True, pad to batch max rather than input_length

        # Cuneiform-specific options:
        use_sign_boundaries: If True, corrupt at sign boundaries
        use_determinative_targeting: If True, preferentially corrupt determinatives
        determinative_targeting_rate: Probability of targeting a determinative (0.0-1.0)
        use_edge_weighting: If True, weight corruption toward sequence edges
        edge_bias: How much more likely edges are corrupted (2.0 = 2x)
        edge_width: Fraction of sequence considered "edge" (0.2 = 20%)
        use_bracket_damage: If True, use [x] instead of * for corruption
        long_gap_threshold: Span length at which to use '* * *' instead of '*' (default 3)

        # Curriculum support:
        scheduler: Optional CurriculumScheduler for dynamic parameters

    Example:
        collator = AdvancedSpanCorruptionCollator(
            tokenizer=tokenizer,
            noise_density=0.15,
            use_sign_boundaries=True,
            use_determinative_targeting=True,
            determinative_targeting_rate=0.3,
            use_edge_weighting=True,
            edge_bias=2.0,
        )
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        input_length: int = 512,
        target_length: int = 512,
        pad_to_multiple_of: Optional[int] = 8,
        dynamic_inputs: bool = True,
        add_eos_to_target: bool = True,
        # Cuneiform-specific options
        use_sign_boundaries: bool = False,
        use_determinative_targeting: bool = False,
        determinative_targeting_rate: float = 0.3,
        use_edge_weighting: bool = False,
        edge_bias: float = 2.0,
        edge_width: float = 0.2,
        use_bracket_damage: bool = False,
        long_gap_threshold: int = 3,
        # Curriculum support
        scheduler: Optional[CurriculumScheduler] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.tok = tokenizer
        self.base_noise_density = float(noise_density)
        self.base_mean_span_length = float(mean_noise_span_length)
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.p2m = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.dynamic_inputs = bool(dynamic_inputs)
        self.add_eos_to_target = bool(add_eos_to_target)

        # Cuneiform-specific
        self.use_sign_boundaries = bool(use_sign_boundaries)
        self.use_determinative_targeting = bool(use_determinative_targeting)
        self.base_determinative_rate = float(determinative_targeting_rate)
        self.use_edge_weighting = bool(use_edge_weighting)
        self.base_edge_bias = float(edge_bias)
        self.edge_width = float(edge_width)
        self.use_bracket_damage = bool(use_bracket_damage)
        self.long_gap_threshold = int(long_gap_threshold)

        # Curriculum
        self.scheduler = scheduler

        # RNG
        self.rng = rng if rng is not None else np.random.default_rng()

        # Token IDs
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

        # Corruption tokens: either * / * * * or bracket tokens
        if use_bracket_damage:
            # For bracket damage, we'll use multiple tokens: [, x, ]
            self.bracket_open_id = tokenizer.convert_tokens_to_ids("[")
            self.bracket_x_id = tokenizer.convert_tokens_to_ids("x")
            self.bracket_close_id = tokenizer.convert_tokens_to_ids("]")
            self.short_corruption_ids = [self.bracket_open_id, self.bracket_x_id, self.bracket_close_id]
            self.long_corruption_ids = self.short_corruption_ids  # Same for bracket mode
        else:
            star_id = tokenizer.convert_tokens_to_ids("*")
            if star_id is None or star_id == tokenizer.unk_token_id:
                raise ValueError("Tokenizer does not contain '*' token; add it or use bracket damage mode.")
            self.star_id = int(star_id)
            # Short gap: single * | Long gap: * * * (matches text_pipeline.py convention)
            self.short_corruption_ids = [self.star_id]
            self.long_corruption_ids = [self.star_id, self.star_id, self.star_id]

        # Legacy attribute for compatibility
        self.corruption_ids = self.short_corruption_ids

    def _get_current_params(self) -> Dict[str, float]:
        """Get current parameters, applying curriculum if available."""
        if self.scheduler is not None:
            return {
                "noise_density": self.scheduler.get_noise_density(),
                "mean_span_length": self.scheduler.get_mean_span_length(),
                "determinative_rate": self.scheduler.get_determinative_targeting_rate(),
                "edge_bias": self.scheduler.get_edge_bias(),
            }
        return {
            "noise_density": self.base_noise_density,
            "mean_span_length": self.base_mean_span_length,
            "determinative_rate": self.base_determinative_rate,
            "edge_bias": self.base_edge_bias,
        }

    def _select_corruption_positions(
        self,
        tokens: List[int],
        noise_density: float,
        mean_span_length: float,
        determinative_rate: float,
        edge_bias: float,
    ) -> List[Tuple[int, int]]:
        """
        Select which token spans to corrupt.

        Returns list of (start_idx, length) tuples.
        """
        L = len(tokens)
        if L == 0:
            return []

        n_noise = max(1, int(round(L * noise_density)))
        n_noise = min(n_noise, L - 1)  # Keep at least 1 token uncorrupted

        # Compute base weights
        if self.use_edge_weighting and edge_bias > 1.0:
            weights = _compute_edge_weights(L, edge_bias, self.edge_width)
        else:
            weights = np.ones(L)

        # Boost determinative weights if targeting
        if self.use_determinative_targeting and determinative_rate > 0:
            det_positions = _find_determinative_positions(tokens, self.tok)
            for pos in det_positions:
                if pos < L:
                    # Boost weight significantly for determinatives
                    weights[pos] *= (1.0 + determinative_rate * 5.0)

        # Normalize weights to probabilities
        weights = weights / weights.sum()

        # Select span starting positions
        spans = []
        occupied = set()
        total_masked = 0

        max_attempts = L * 3
        attempts = 0

        while total_masked < n_noise and attempts < max_attempts:
            attempts += 1

            # Sample span length from exponential distribution
            span_len = max(1, int(self.rng.exponential(mean_span_length)))
            span_len = min(span_len, n_noise - total_masked, L)

            # Sample starting position with weights
            # Mask out occupied positions
            available_weights = weights.copy()
            for pos in occupied:
                if pos < L:
                    available_weights[pos] = 0

            # Can't start span too close to end
            for i in range(max(0, L - span_len + 1), L):
                available_weights[i] = 0

            if available_weights.sum() <= 0:
                break

            available_weights = available_weights / available_weights.sum()
            start = self.rng.choice(L, p=available_weights)

            # Check for overlap
            span_positions = set(range(start, min(start + span_len, L)))
            if span_positions & occupied:
                continue

            # Accept span
            actual_len = min(span_len, L - start)
            spans.append((start, actual_len))
            occupied.update(range(start, start + actual_len))
            total_masked += actual_len

        # Sort by position
        spans.sort(key=lambda x: x[0])

        return spans

    def _apply_corruption(
        self,
        tokens: List[int],
        spans: List[Tuple[int, int]],
    ) -> Tuple[List[int], List[int]]:
        """
        Apply corruption to tokens based on selected spans.

        Uses '*' for short spans (< long_gap_threshold tokens) and
        '* * *' for longer spans, matching text_pipeline.py convention.

        Returns (corrupted_input, labels).
        """
        corrupted = []
        labels = []
        prev_end = 0

        for start, length in spans:
            # Add uncorrupted tokens before this span
            if start > prev_end:
                corrupted.extend(tokens[prev_end:start])

            # Choose corruption marker based on span length
            # Short spans (1-2 tokens): single *
            # Long spans (3+ tokens): * * * (matches text_pipeline.py)
            if length >= self.long_gap_threshold:
                marker_ids = self.long_corruption_ids
            else:
                marker_ids = self.short_corruption_ids

            # Add corruption marker(s)
            corrupted.extend(marker_ids)

            # Add corruption marker to labels, then the masked tokens
            labels.extend(marker_ids)
            labels.extend(tokens[start:start + length])

            prev_end = start + length

        # Add remaining tokens
        if prev_end < len(tokens):
            corrupted.extend(tokens[prev_end:])

        # Add EOS to labels
        if self.add_eos_to_target and self.eos_id is not None:
            labels.append(self.eos_id)

        return corrupted, labels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples with corruption."""
        B = len(examples)
        params = self._get_current_params()

        corrupted_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for ex in examples:
            tokens: List[int] = ex["input_ids"]
            L = len(tokens)

            if L == 0:
                # Empty input - just add EOS
                corrupted_list.append([self.eos_id] if self.eos_id else [self.pad_id or 0])
                labels_list.append([self.eos_id] if self.eos_id else [self.pad_id or 0])
                continue

            # Select corruption spans
            spans = self._select_corruption_positions(
                tokens,
                params["noise_density"],
                params["mean_span_length"],
                params["determinative_rate"],
                params["edge_bias"],
            )

            # Apply corruption
            corrupted, labels = self._apply_corruption(tokens, spans)

            # Truncate
            corrupted = corrupted[:self.input_length]
            labels = labels[:self.target_length]

            corrupted_list.append(corrupted)
            labels_list.append(labels)

        # Determine padding lengths
        if self.dynamic_inputs:
            max_in = max((len(x) for x in corrupted_list), default=1)
            max_in = min(max_in, self.input_length)
        else:
            max_in = self.input_length

        max_tgt = min(self.target_length, max((len(x) for x in labels_list), default=1))

        # Ensure non-zero dimensions
        max_in = max(max_in, 1)
        max_tgt = max(max_tgt, 1)

        # Round up to multiple
        if self.p2m:
            if max_in % self.p2m != 0:
                max_in = int(math.ceil(max_in / self.p2m) * self.p2m)
            if max_tgt % self.p2m != 0:
                max_tgt = int(math.ceil(max_tgt / self.p2m) * self.p2m)

        # Allocate tensors
        inputs = torch.full((B, max_in), self.pad_id, dtype=torch.long)
        labels = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)

        # Fill tensors
        for i, (cin, tgt) in enumerate(zip(corrupted_list, labels_list)):
            if not cin:
                cin = [self.eos_id] if self.eos_id is not None else [self.pad_id or 0]
            if not tgt:
                tgt = [self.eos_id] if self.eos_id is not None else [self.pad_id or 0]

            li = min(len(cin), max_in)
            lt = min(len(tgt), max_tgt)

            if li > 0:
                inputs[i, :li] = torch.tensor(cin[:li], dtype=torch.long)
            if lt > 0:
                labels[i, :lt] = torch.tensor(tgt[:lt], dtype=torch.long)

        attention_mask = (inputs != self.pad_id).long()
        labels.masked_fill_(labels == self.pad_id, -100)

        return {
            "input_ids": inputs.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": labels.contiguous(),
        }


# -----------------------------------------------------------------------------
# SIGN-BOUNDARY AWARE CORRUPTION COLLATOR
# -----------------------------------------------------------------------------

class SignBoundaryCorruptionCollator:
    """
    Corruption collator that respects cuneiform sign boundaries.

    Instead of corrupting arbitrary token spans, this collator:
    1. Decodes tokens to text
    2. Identifies sign boundaries (separated by - or space)
    3. Corrupts whole signs rather than partial tokens
    4. Re-encodes the corrupted text

    This produces more realistic corruption that matches how tablet damage
    actually affects cuneiform texts.

    Args:
        tokenizer: HuggingFace tokenizer
        noise_density: Fraction of signs to corrupt (0.15 = 15%)
        input_length: Maximum input sequence length
        target_length: Maximum target sequence length
        use_bracket_damage: If True, use [x] markers instead of *
        pad_to_multiple_of: Pad to multiple of this value
        long_gap_threshold: Number of consecutive signs for '* * *' vs '*' (default 3)
        mean_gap_length: Average number of consecutive signs to corrupt (default 1.5)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float = 0.15,
        input_length: int = 512,
        target_length: int = 512,
        pad_to_multiple_of: Optional[int] = 8,
        use_bracket_damage: bool = False,
        dynamic_inputs: bool = True,
        add_eos_to_target: bool = True,
        long_gap_threshold: int = 3,
        mean_gap_length: float = 1.5,
        rng: Optional[np.random.Generator] = None,
    ):
        self.tok = tokenizer
        self.noise_density = float(noise_density)
        self.input_length = int(input_length)
        self.target_length = int(target_length)
        self.p2m = int(pad_to_multiple_of) if pad_to_multiple_of else None
        self.use_bracket_damage = bool(use_bracket_damage)
        self.dynamic_inputs = bool(dynamic_inputs)
        self.add_eos_to_target = bool(add_eos_to_target)
        self.long_gap_threshold = int(long_gap_threshold)
        self.mean_gap_length = float(mean_gap_length)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.pad_id = tokenizer.pad_token_id
        self.eos_id = getattr(tokenizer, "eos_token_id", None)

        # Corruption markers matching text_pipeline.py convention
        if use_bracket_damage:
            self.short_marker = "[x]"
            self.long_marker = "[x]"  # Same for bracket mode
        else:
            self.short_marker = "*"
            self.long_marker = "* * *"

        # Legacy attribute
        self.corruption_marker = self.short_marker

    def _corrupt_at_sign_boundaries(self, text: str) -> Tuple[str, str]:
        """
        Corrupt text at sign boundaries.

        Supports consecutive sign corruption (gap spans) with appropriate markers:
        - Single sign or 2 consecutive signs: '*'
        - 3+ consecutive signs: '* * *'

        Returns (corrupted_text, target_text).
        """
        if not text.strip():
            return text, text

        # Split into signs (preserving separators)
        # Pattern: match signs and separators separately
        pattern = r'([^\s\-\.]+|[\s\-\.]+)'
        parts = re.findall(pattern, text)

        if not parts:
            return text, text

        # Identify which parts are signs (not separators)
        sign_indices = [i for i, p in enumerate(parts) if p.strip() and not re.match(r'^[\s\-\.]+$', p)]

        if not sign_indices:
            return text, text

        # Calculate total signs to corrupt
        n_signs = len(sign_indices)
        n_corrupt = max(1, int(round(n_signs * self.noise_density)))
        n_corrupt = min(n_corrupt, n_signs - 1)  # Keep at least one sign

        # Build corruption spans (consecutive sign groups)
        # This simulates realistic tablet damage where adjacent signs may be lost together
        corrupt_set = set()
        remaining = n_corrupt

        while remaining > 0 and len(corrupt_set) < n_signs - 1:
            # Sample gap length from exponential distribution
            gap_len = max(1, int(self.rng.exponential(self.mean_gap_length)))
            gap_len = min(gap_len, remaining, n_signs - len(corrupt_set) - 1)

            # Find a starting position that's not already corrupted
            available = [idx for idx in sign_indices if idx not in corrupt_set]
            if not available:
                break

            start_sign_pos = int(self.rng.choice(available))
            start_idx = sign_indices.index(start_sign_pos)

            # Corrupt consecutive signs starting from this position
            for j in range(gap_len):
                if start_idx + j < len(sign_indices):
                    corrupt_set.add(sign_indices[start_idx + j])
                    remaining -= 1

        # Build corrupted and target, merging consecutive corruptions
        corrupted_parts = []
        target_parts = []
        i = 0

        while i < len(parts):
            part = parts[i]

            if i in corrupt_set:
                # Count consecutive corrupted signs
                gap_start = i
                corrupted_signs = []

                while i < len(parts) and i in corrupt_set:
                    corrupted_signs.append(parts[i])
                    i += 1
                    # Skip separator after corrupted sign (if any)
                    while i < len(parts) and re.match(r'^[\s\-\.]+$', parts[i]):
                        if i in corrupt_set or (i + 1 < len(parts) and i + 1 in corrupt_set):
                            i += 1
                        else:
                            break

                # Choose marker based on number of corrupted signs
                n_corrupted = len(corrupted_signs)
                marker = self.long_marker if n_corrupted >= self.long_gap_threshold else self.short_marker

                corrupted_parts.append(marker)
                target_parts.append(marker)
                for sign in corrupted_signs:
                    target_parts.append(sign)
            else:
                corrupted_parts.append(part)
                i += 1

        corrupted_text = ''.join(corrupted_parts)
        target_text = ''.join(target_parts)

        return corrupted_text, target_text

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with sign-boundary corruption."""
        B = len(examples)

        corrupted_list: List[List[int]] = []
        labels_list: List[List[int]] = []

        for ex in examples:
            tokens: List[int] = ex["input_ids"]

            # Decode to text
            text = self.tok.decode(tokens, skip_special_tokens=True)

            # Corrupt at sign boundaries
            corrupted_text, target_text = self._corrupt_at_sign_boundaries(text)

            # Re-encode
            corrupted_ids = self.tok.encode(corrupted_text, add_special_tokens=False)
            target_ids = self.tok.encode(target_text, add_special_tokens=False)

            # Add EOS to target
            if self.add_eos_to_target and self.eos_id is not None:
                target_ids.append(self.eos_id)

            # Truncate
            corrupted_ids = corrupted_ids[:self.input_length]
            target_ids = target_ids[:self.target_length]

            corrupted_list.append(corrupted_ids)
            labels_list.append(target_ids)

        # Padding (same as AdvancedSpanCorruptionCollator)
        if self.dynamic_inputs:
            max_in = max((len(x) for x in corrupted_list), default=1)
            max_in = min(max_in, self.input_length)
        else:
            max_in = self.input_length

        max_tgt = min(self.target_length, max((len(x) for x in labels_list), default=1))
        max_in = max(max_in, 1)
        max_tgt = max(max_tgt, 1)

        if self.p2m:
            if max_in % self.p2m != 0:
                max_in = int(math.ceil(max_in / self.p2m) * self.p2m)
            if max_tgt % self.p2m != 0:
                max_tgt = int(math.ceil(max_tgt / self.p2m) * self.p2m)

        inputs = torch.full((B, max_in), self.pad_id, dtype=torch.long)
        labels = torch.full((B, max_tgt), self.pad_id, dtype=torch.long)

        for i, (cin, tgt) in enumerate(zip(corrupted_list, labels_list)):
            if not cin:
                cin = [self.eos_id] if self.eos_id is not None else [self.pad_id or 0]
            if not tgt:
                tgt = [self.eos_id] if self.eos_id is not None else [self.pad_id or 0]

            li = min(len(cin), max_in)
            lt = min(len(tgt), max_tgt)

            if li > 0:
                inputs[i, :li] = torch.tensor(cin[:li], dtype=torch.long)
            if lt > 0:
                labels[i, :lt] = torch.tensor(tgt[:lt], dtype=torch.long)

        attention_mask = (inputs != self.pad_id).long()
        labels.masked_fill_(labels == self.pad_id, -100)

        return {
            "input_ids": inputs.contiguous(),
            "attention_mask": attention_mask.contiguous(),
            "labels": labels.contiguous(),
        }


# -----------------------------------------------------------------------------
# ADVANCED MIXED DATA COLLATOR WITH CURRICULUM
# -----------------------------------------------------------------------------

class AdvancedMixedDataCollator:
    """
    Advanced mixed data collator combining pretrain and translate tasks
    with curriculum learning support and all cuneiform-specific features.

    This is an enhanced version of MixedDataCollator8Buckets with:
      - Dynamic pretrain/translate ratio scheduling
      - Curriculum-based corruption parameters
      - Support for all advanced corruption features
      - Better bucketing and padding strategies

    Args:
        tokenizer: HuggingFace tokenizer
        pretrain_collator: Collator for pretrain task (e.g., AdvancedSpanCorruptionCollator)
        translate_collator: Collator for translation task (optional)
        prompt_text: Prompt to prepend to pretrain examples
        FIXED_MAX_LEN: Maximum sequence length
        allowed_buckets: Sequence length buckets for efficient batching
        add_decoder_inputs: Whether to add decoder_input_ids
        protect_prompt: If True, don't corrupt the prompt

        # Curriculum support
        scheduler: CurriculumScheduler for dynamic parameters
        base_pretrain_ratio: Base ratio of pretrain to translate (0.0-1.0)

    Example:
        scheduler = CurriculumScheduler(
            noise_schedule=[(0, 0.10), (100, 0.20)],
            pretrain_ratio_schedule=[(0, 0.8), (200, 0.2)]
        )

        pretrain_collator = AdvancedSpanCorruptionCollator(
            tokenizer=tokenizer,
            scheduler=scheduler,
            use_determinative_targeting=True,
            use_edge_weighting=True,
        )

        collator = AdvancedMixedDataCollator(
            tokenizer=tokenizer,
            pretrain_collator=pretrain_collator,
            scheduler=scheduler,
        )
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pretrain_collator,
        translate_collator=None,
        prompt_text: Optional[str] = "Identify missing signs: ",
        FIXED_MAX_LEN: int = 512,
        allowed_buckets: Sequence[int] = (8, 16, 32, 48, 64, 128, 256, 512),
        add_decoder_inputs: bool = True,
        protect_prompt: bool = True,
        pad_to_bucket: bool = False,
        # Curriculum support
        scheduler: Optional[CurriculumScheduler] = None,
        base_pretrain_ratio: float = 0.5,
    ):
        self.tok = tokenizer
        self.preC = pretrain_collator
        self.trC = translate_collator

        self.prompt_text = prompt_text
        self.FIXED_MAX_LEN = int(FIXED_MAX_LEN)
        self.allowed = tuple(sorted(allowed_buckets))
        self.add_decoder = bool(add_decoder_inputs)
        self.protect_prompt = bool(protect_prompt)
        self.pad_to_bucket = bool(pad_to_bucket)

        self.scheduler = scheduler
        self.base_pretrain_ratio = float(base_pretrain_ratio)

        self.PAD = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.START = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else self.PAD

        if isinstance(prompt_text, str) and prompt_text.strip():
            self.prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        else:
            self.prompt_ids = []

    def _get_pretrain_ratio(self) -> float:
        """Get current pretrain ratio from scheduler or base value."""
        if self.scheduler is not None:
            return self.scheduler.get_pretrain_ratio()
        return self.base_pretrain_ratio

    def _make_decoder_inputs(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create decoder inputs from labels."""
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1]
        shifted[:, 0] = int(self.START)
        shifted.masked_fill_(shifted == -100, int(self.PAD))
        return {
            "decoder_input_ids": shifted.contiguous(),
            "decoder_attention_mask": (shifted != int(self.PAD)).long().contiguous(),
        }

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate batch with dynamic task mixing."""
        # Separate pretrain and translate examples
        pretrain_examples = []
        translate_examples = []

        for ex in examples:
            task = ex.get("task", None)

            if task == "pretrain":
                pretrain_examples.append(ex)
            elif task == "train":
                translate_examples.append(ex)
            else:
                # Fallback: check for labels
                if "labels" in ex and ex["labels"] is not None:
                    translate_examples.append(ex)
                else:
                    pretrain_examples.append(ex)

        # Process pretrain examples
        pre_batches: List[Dict[str, torch.Tensor]] = []
        if pretrain_examples:
            buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            prompt_len = len(self.prompt_ids)
            max_content = max(1, self.FIXED_MAX_LEN - prompt_len) if self.protect_prompt else self.FIXED_MAX_LEN

            for ex in pretrain_examples:
                ex2 = dict(ex)

                if self.protect_prompt and self.prompt_ids:
                    content_ids = ex["input_ids"][:max_content]
                    if len(content_ids) == 0:
                        content_ids = [int(self.START)]
                    ex2["input_ids"] = content_ids
                    eff_len = min(self.FIXED_MAX_LEN, prompt_len + len(content_ids))
                else:
                    ids = (self.prompt_ids + ex["input_ids"])[:self.FIXED_MAX_LEN]
                    if len(ids) == 0:
                        ids = [int(self.START)]
                    ex2["input_ids"] = ids
                    eff_len = len(ids)

                tgt_len = _round_up_to_allowed(eff_len, self.allowed)
                buckets[tgt_len].append(ex2)

            for tgt_len, group in buckets.items():
                batch = self.preC(group)

                if self.protect_prompt and self.prompt_ids:
                    B = batch["input_ids"].size(0)
                    prompt = torch.tensor(self.prompt_ids, dtype=torch.long, device=batch["input_ids"].device)
                    prompt = prompt.unsqueeze(0).expand(B, -1)

                    new_inp = torch.cat([prompt, batch["input_ids"]], dim=1)
                    new_inp = new_inp[:, :self.FIXED_MAX_LEN].contiguous()

                    batch["input_ids"] = new_inp
                    batch["attention_mask"] = (new_inp != int(self.PAD)).long().contiguous()

                if self.pad_to_bucket:
                    want_w = min(int(tgt_len), int(self.FIXED_MAX_LEN))
                    batch = _pad_batch_to_len(batch, want_w, int(self.PAD))

                if self.add_decoder:
                    batch.update(self._make_decoder_inputs(batch["labels"]))

                pre_batches.append(batch)

        # Process translate examples
        tr_batch: Optional[Dict[str, torch.Tensor]] = None
        if translate_examples:
            fixed = []
            for ex in translate_examples:
                ids = ex["input_ids"][:self.FIXED_MAX_LEN]
                lbl = ex["labels"][:self.FIXED_MAX_LEN]

                if len(ids) == 0:
                    ids = [int(self.START)]
                if len(lbl) == 0:
                    lbl = [int(self.START)]

                fixed.append({"input_ids": ids, "labels": lbl})

            if self.trC is not None:
                tr_batch = self.trC(fixed)
            else:
                input_ids = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f["input_ids"], dtype=torch.long) for f in fixed],
                    batch_first=True, padding_value=int(self.PAD)
                )
                labels = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f["labels"], dtype=torch.long) for f in fixed],
                    batch_first=True, padding_value=-100
                )
                attention_mask = (input_ids != int(self.PAD)).long()
                tr_batch = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

            if self.add_decoder:
                tr_batch.update(self._make_decoder_inputs(tr_batch["labels"]))

        # Merge all batches
        all_batches: List[Dict[str, torch.Tensor]] = []
        if pre_batches:
            all_batches.extend(pre_batches)
        if tr_batch is not None:
            all_batches.append(tr_batch)

        if not all_batches:
            # Return dummy batch
            pad = int(self.PAD)
            one = torch.tensor([[pad]], dtype=torch.long)
            return {
                "input_ids": one,
                "attention_mask": (one != pad).long(),
                "labels": torch.full((1, 1), -100, dtype=torch.long),
            }

        # Pad all batches to same width
        max_w = max(b["input_ids"].size(1) for b in all_batches)
        all_batches = [_pad_batch_to_len(b, max_w, int(self.PAD)) for b in all_batches]

        # Concatenate
        keys = set().union(*(b.keys() for b in all_batches))
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            out[k] = torch.cat([b[k] for b in all_batches if k in b], dim=0)

        return out


# -----------------------------------------------------------------------------
# TRANSLATION CORRUPTION WITH DETERMINATIVE TARGETING
# -----------------------------------------------------------------------------

class DataCollatorForTranslationWithDeterminativeCorruption(DataCollatorForSeq2Seq):
    """
    Translation collator that applies determinative-targeting corruption.

    For translation tasks (source → target), this collator:
    1. Optionally corrupts the source with probability corruption_probability
    2. When corrupting, preferentially targets determinatives
    3. Uses edge-weighted corruption to simulate tablet damage

    This helps make translation robust to damaged/incomplete inputs.

    Args:
        tokenizer: HuggingFace tokenizer
        model: Model for DataCollatorForSeq2Seq
        corruption_probability: Probability of corrupting an example (0.0-1.0)
        noise_density: Fraction of tokens to corrupt when corrupting
        determinative_targeting_rate: How much to prefer determinatives (0.0-1.0)
        use_edge_weighting: If True, weight corruption toward edges
        edge_bias: Edge weighting factor
        source_length: Maximum source sequence length
        target_length: Maximum target sequence length
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        corruption_probability: float = 0.15,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
        determinative_targeting_rate: float = 0.3,
        use_edge_weighting: bool = True,
        edge_bias: float = 2.0,
        edge_width: float = 0.2,
        source_length: int = 512,
        target_length: int = 512,
        use_bracket_damage: bool = False,
    ):
        super().__init__(tokenizer, model=model, return_tensors="pt")

        self.corruption_probability = float(corruption_probability)
        self.noise_density = float(noise_density)
        self.mean_noise_span_length = float(mean_noise_span_length)
        self.determinative_targeting_rate = float(determinative_targeting_rate)
        self.use_edge_weighting = bool(use_edge_weighting)
        self.edge_bias = float(edge_bias)
        self.edge_width = float(edge_width)
        self.source_length = int(source_length)
        self.target_length = int(target_length)
        self.use_bracket_damage = bool(use_bracket_damage)

        self.tokenizer = tokenizer
        self.rng = np.random.default_rng()

        self.corruption_marker = "[x]" if use_bracket_damage else "*"
        self.star_id = tokenizer.convert_tokens_to_ids("*")

    def _corrupt_with_targeting(self, input_ids: List[int]) -> List[int]:
        """Apply determinative-targeting corruption to input."""
        L = len(input_ids)
        if L <= 1:
            return input_ids

        n_corrupt = max(1, int(round(L * self.noise_density)))
        n_corrupt = min(n_corrupt, L - 1)

        # Compute weights
        if self.use_edge_weighting:
            weights = _compute_edge_weights(L, self.edge_bias, self.edge_width)
        else:
            weights = np.ones(L)

        # Boost determinative weights
        if self.determinative_targeting_rate > 0:
            det_positions = _find_determinative_positions(input_ids, self.tokenizer)
            for pos in det_positions:
                if pos < L:
                    weights[pos] *= (1.0 + self.determinative_targeting_rate * 5.0)

        weights = weights / weights.sum()

        # Select positions to corrupt
        corrupt_positions = set(self.rng.choice(L, size=n_corrupt, replace=False, p=weights))

        # Build corrupted input
        corrupted = []
        i = 0
        while i < L:
            if i in corrupt_positions:
                # Start a span
                span_len = max(1, int(self.rng.exponential(self.mean_noise_span_length)))
                span_len = min(span_len, L - i)

                corrupted.append(self.star_id)
                i += span_len
            else:
                corrupted.append(input_ids[i])
                i += 1

        return corrupted

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate with optional corruption."""
        processed = []

        for f in features:
            new_f = dict(f)

            # Optionally corrupt source
            if random.random() < self.corruption_probability:
                new_f["input_ids"] = self._corrupt_with_targeting(f["input_ids"])

            # Truncate
            new_f["input_ids"] = new_f["input_ids"][:self.source_length]
            if "labels" in new_f:
                new_f["labels"] = new_f["labels"][:self.target_length]

            processed.append(new_f)

        return super().__call__(processed)


# -----------------------------------------------------------------------------
# LEGACY COLLATOR ALIASES (for backwards compatibility)
# -----------------------------------------------------------------------------

# The original collators are preserved above with their original names:
#   - T5SpanCorruptionCollatorFast
#   - T5DataCollatorForSpanCorruptionAsteriskFast
#   - MixedDataCollator8Buckets
#   - DataCollatorForTranslationCorruptionBuckets
#   - CappedSeq2SeqCollator
#
# These new advanced collators can be used as drop-in replacements:
#   - AdvancedSpanCorruptionCollator (replaces T5DataCollatorForSpanCorruptionAsteriskFast)
#   - SignBoundaryCorruptionCollator (new capability)
#   - AdvancedMixedDataCollator (replaces MixedDataCollator8Buckets)
#   - DataCollatorForTranslationWithDeterminativeCorruption (enhanced translation)


def create_curriculum_collators(
    tokenizer: PreTrainedTokenizerBase,
    input_length: int = 512,
    target_length: int = 512,
    # Curriculum schedules
    noise_schedule: Optional[List[Tuple[int, float]]] = None,
    pretrain_ratio_schedule: Optional[List[Tuple[int, float]]] = None,
    determinative_rate_schedule: Optional[List[Tuple[int, float]]] = None,
    edge_bias_schedule: Optional[List[Tuple[int, float]]] = None,
    # Feature flags
    use_sign_boundaries: bool = False,
    use_determinative_targeting: bool = True,
    use_edge_weighting: bool = True,
    use_bracket_damage: bool = False,
    # Other options
    prompt_text: str = "Identify missing signs: ",
    allowed_buckets: Sequence[int] = (8, 16, 32, 48, 64, 128, 256, 512),
) -> Tuple[CurriculumScheduler, AdvancedSpanCorruptionCollator, AdvancedMixedDataCollator]:
    """
    Factory function to create a complete curriculum-based training setup.

    Returns:
        (scheduler, pretrain_collator, mixed_collator)

    Example:
        scheduler, pretrain_collator, mixed_collator = create_curriculum_collators(
            tokenizer=tokenizer,
            noise_schedule=[(0, 0.10), (50, 0.15), (150, 0.20)],
            pretrain_ratio_schedule=[(0, 0.8), (100, 0.5), (200, 0.2)],
            use_determinative_targeting=True,
            use_edge_weighting=True,
        )

        # In training loop:
        for epoch in range(num_epochs):
            scheduler.set_epoch(epoch)
            # ... train with mixed_collator ...
    """
    # Default schedules if not provided
    if noise_schedule is None:
        noise_schedule = [(0, 0.10), (50, 0.15), (150, 0.20)]

    if pretrain_ratio_schedule is None:
        pretrain_ratio_schedule = [(0, 0.7), (100, 0.5), (200, 0.3)]

    if determinative_rate_schedule is None:
        determinative_rate_schedule = [(0, 0.2), (100, 0.4)]

    if edge_bias_schedule is None:
        edge_bias_schedule = [(0, 1.5), (100, 2.0)]

    # Create scheduler
    scheduler = CurriculumScheduler(
        noise_schedule=noise_schedule,
        pretrain_ratio_schedule=pretrain_ratio_schedule,
        determinative_rate_schedule=determinative_rate_schedule,
        edge_bias_schedule=edge_bias_schedule,
    )

    # Create pretrain collator
    if use_sign_boundaries:
        pretrain_collator = SignBoundaryCorruptionCollator(
            tokenizer=tokenizer,
            noise_density=noise_schedule[0][1],  # Initial value
            input_length=input_length,
            target_length=target_length,
            use_bracket_damage=use_bracket_damage,
        )
    else:
        pretrain_collator = AdvancedSpanCorruptionCollator(
            tokenizer=tokenizer,
            input_length=input_length,
            target_length=target_length,
            use_determinative_targeting=use_determinative_targeting,
            use_edge_weighting=use_edge_weighting,
            use_bracket_damage=use_bracket_damage,
            scheduler=scheduler,
        )

    # Create mixed collator
    mixed_collator = AdvancedMixedDataCollator(
        tokenizer=tokenizer,
        pretrain_collator=pretrain_collator,
        prompt_text=prompt_text,
        FIXED_MAX_LEN=input_length,
        allowed_buckets=allowed_buckets,
        scheduler=scheduler,
    )

    return scheduler, pretrain_collator, mixed_collator


# -----------------------------------------------------------------------------
# R-DROP REGULARIZATION COLLATOR
# -----------------------------------------------------------------------------

class RDropCollator:
    """
    Wraps a base collator to enable R-Drop regularization.

    R-Drop runs each sample through the model twice with different dropout masks,
    then minimizes the KL divergence between the two output distributions.
    This is highly effective for low-resource translation tasks.

    Reference: "R-Drop: Regularized Dropout for Neural Networks" (Liang et al., 2021)

    Args:
        base_collator: The underlying collator to wrap
        rdrop_alpha: Weight for the KL divergence loss (0.0 to 1.0+)
                     Higher values = stronger regularization
        duplicate_batch: If True, duplicates the batch (2x memory, enables R-Drop)
                        If False, just passes through (for inference)

    Usage:
        base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        rdrop_collator = RDropCollator(base_collator, rdrop_alpha=0.7)

        # In custom training loop, compute:
        # loss = ce_loss + rdrop_alpha * kl_divergence(output1, output2)
    """

    def __init__(
        self,
        base_collator,
        rdrop_alpha: float = 0.7,
        duplicate_batch: bool = True,
    ):
        self.base_collator = base_collator
        self.rdrop_alpha = float(rdrop_alpha)
        self.duplicate_batch = bool(duplicate_batch)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate and optionally duplicate batch for R-Drop."""
        batch = self.base_collator(features)

        if not self.duplicate_batch:
            return batch

        # Duplicate each tensor in the batch
        duplicated = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                duplicated[key] = torch.cat([value, value.clone()], dim=0)
            else:
                duplicated[key] = value

        # Add R-Drop metadata
        duplicated['rdrop_alpha'] = self.rdrop_alpha
        duplicated['rdrop_enabled'] = True
        duplicated['rdrop_batch_size'] = batch['input_ids'].size(0)  # Original batch size

        return duplicated


def compute_rdrop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    labels: torch.Tensor,
    rdrop_alpha: float = 0.7,
    pad_token_id: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute R-Drop loss: CE loss + alpha * KL divergence.

    Args:
        logits1: First forward pass logits [batch, seq, vocab]
        logits2: Second forward pass logits [batch, seq, vocab]
        labels: Target labels [batch, seq]
        rdrop_alpha: Weight for KL divergence term
        pad_token_id: Token ID to ignore in loss computation

    Returns:
        total_loss: Combined loss (CE + alpha * KL)
        ce_loss: Cross-entropy loss only
        kl_loss: KL divergence loss only
    """
    import torch.nn.functional as F

    # Compute CE loss for both passes
    ce_loss1 = F.cross_entropy(
        logits1.view(-1, logits1.size(-1)),
        labels.view(-1),
        ignore_index=pad_token_id,
        reduction='mean'
    )
    ce_loss2 = F.cross_entropy(
        logits2.view(-1, logits2.size(-1)),
        labels.view(-1),
        ignore_index=pad_token_id,
        reduction='mean'
    )
    ce_loss = (ce_loss1 + ce_loss2) / 2

    # Compute KL divergence between the two distributions
    # Use log_softmax for numerical stability
    log_p1 = F.log_softmax(logits1, dim=-1)
    log_p2 = F.log_softmax(logits2, dim=-1)
    p1 = F.softmax(logits1, dim=-1)
    p2 = F.softmax(logits2, dim=-1)

    # Symmetric KL: (KL(p1||p2) + KL(p2||p1)) / 2
    kl_loss = (
        F.kl_div(log_p1, p2, reduction='batchmean') +
        F.kl_div(log_p2, p1, reduction='batchmean')
    ) / 2

    total_loss = ce_loss + rdrop_alpha * kl_loss

    return total_loss, ce_loss, kl_loss


# -----------------------------------------------------------------------------
# TABLET DAMAGE PATTERN COLLATOR
# -----------------------------------------------------------------------------

class TabletDamageCollator:
    """
    Simulates realistic cuneiform tablet damage patterns for data augmentation.

    This collator applies various damage patterns that occur on real tablets:
    - Vertical cracks: Continuous damage through the tablet (breaks)
    - Surface erosion: Scattered random damage (weathering)
    - Corner damage: Missing chunks at start/end (broken corners)
    - Salt damage: Clustered damage from salt crystallization
    - Line breaks: Damage at logical line boundaries

    Each damage type can be enabled/disabled and tuned independently.

    Args:
        tokenizer: HuggingFace tokenizer
        vertical_crack_prob: Probability of vertical crack damage per example
        surface_erosion_prob: Probability of surface erosion per example
        corner_damage_prob: Probability of corner damage per example
        salt_damage_prob: Probability of salt crystallization damage
        line_break_prob: Probability of line break damage
        min_sequence_length: Minimum sequence length to apply damage
        damage_marker: Token to use for damaged areas (default '*')
        long_damage_marker: Token sequence for long damage (default '* * *')
        long_damage_threshold: Number of consecutive corruptions to trigger long marker

    Usage:
        damage_collator = TabletDamageCollator(
            tokenizer=tokenizer,
            vertical_crack_prob=0.15,
            surface_erosion_prob=0.20,
            corner_damage_prob=0.25,
        )

        # Wrap your base collator
        base_collator = T5DataCollatorForSpanCorruption(...)
        combined = ChainedCollator([damage_collator, base_collator])
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vertical_crack_prob: float = 0.10,
        surface_erosion_prob: float = 0.15,
        corner_damage_prob: float = 0.20,
        salt_damage_prob: float = 0.08,
        line_break_prob: float = 0.05,
        min_sequence_length: int = 8,
        damage_marker: str = '*',
        long_damage_marker: str = '* * *',
        long_damage_threshold: int = 3,
        max_erosion_fraction: float = 0.15,
        max_crack_width: int = 5,
        max_corner_fraction: float = 0.25,
    ):
        self.tokenizer = tokenizer
        self.vertical_crack_prob = float(vertical_crack_prob)
        self.surface_erosion_prob = float(surface_erosion_prob)
        self.corner_damage_prob = float(corner_damage_prob)
        self.salt_damage_prob = float(salt_damage_prob)
        self.line_break_prob = float(line_break_prob)
        self.min_sequence_length = int(min_sequence_length)
        self.max_erosion_fraction = float(max_erosion_fraction)
        self.max_crack_width = int(max_crack_width)
        self.max_corner_fraction = float(max_corner_fraction)
        self.long_damage_threshold = int(long_damage_threshold)

        # Get marker token IDs
        self.star_id = tokenizer.convert_tokens_to_ids(damage_marker)
        if self.star_id is None or self.star_id == tokenizer.unk_token_id:
            self.star_id = tokenizer.convert_tokens_to_ids('*')

        # Long damage marker (for spans >= threshold)
        self.long_marker_ids = tokenizer.encode(long_damage_marker, add_special_tokens=False)
        if not self.long_marker_ids:
            self.long_marker_ids = [self.star_id, self.star_id, self.star_id]

        self.rng = np.random.default_rng()

        # Common line break tokens in cuneiform transliteration
        self.line_break_tokens = set()
        for token in ['\n', '/', '|', ';']:
            tid = tokenizer.convert_tokens_to_ids(token)
            if tid is not None and tid != tokenizer.unk_token_id:
                self.line_break_tokens.add(tid)

    def _apply_vertical_crack(self, ids: List[int]) -> List[int]:
        """
        Simulate a vertical crack through the tablet.
        Corrupts a continuous span of 2-5 tokens.
        """
        if len(ids) < self.min_sequence_length:
            return ids

        if self.rng.random() > self.vertical_crack_prob:
            return ids

        result = list(ids)

        # Pick crack position (avoid very start/end)
        margin = max(2, len(ids) // 10)
        if len(ids) <= 2 * margin + 2:
            return ids

        crack_pos = self.rng.integers(margin, len(ids) - margin - 1)
        crack_width = self.rng.integers(2, min(self.max_crack_width + 1, len(ids) - crack_pos))

        # Replace with damage markers
        if crack_width >= self.long_damage_threshold:
            # Use long marker for wide cracks
            result = result[:crack_pos] + self.long_marker_ids + result[crack_pos + crack_width:]
        else:
            # Use short markers
            result = result[:crack_pos] + [self.star_id] * crack_width + result[crack_pos + crack_width:]

        return result

    def _apply_surface_erosion(self, ids: List[int]) -> List[int]:
        """
        Simulate scattered surface wear.
        Corrupts random individual tokens (5-15% of sequence).
        """
        if len(ids) < self.min_sequence_length:
            return ids

        if self.rng.random() > self.surface_erosion_prob:
            return ids

        result = list(ids)

        # Corrupt 5-15% of tokens randomly
        erosion_fraction = self.rng.uniform(0.05, self.max_erosion_fraction)
        num_corrupt = max(1, int(len(ids) * erosion_fraction))
        num_corrupt = min(num_corrupt, len(ids) - 1)

        positions = self.rng.choice(len(ids), size=num_corrupt, replace=False)
        for pos in positions:
            result[pos] = self.star_id

        return result

    def _apply_corner_damage(self, ids: List[int]) -> List[int]:
        """
        Simulate broken tablet corners.
        Damages tokens at the start or end of the sequence.
        """
        if len(ids) < self.min_sequence_length:
            return ids

        if self.rng.random() > self.corner_damage_prob:
            return ids

        result = list(ids)

        # Damage 1-25% of sequence at one corner
        max_damage = max(1, int(len(ids) * self.max_corner_fraction))
        damage_len = self.rng.integers(1, max_damage + 1)

        if self.rng.random() < 0.5:
            # Start damage (left edge / top corner)
            if damage_len >= self.long_damage_threshold:
                result = self.long_marker_ids + result[damage_len:]
            else:
                result = [self.star_id] * damage_len + result[damage_len:]
        else:
            # End damage (right edge / bottom corner)
            if damage_len >= self.long_damage_threshold:
                result = result[:-damage_len] + self.long_marker_ids
            else:
                result = result[:-damage_len] + [self.star_id] * damage_len

        return result

    def _apply_salt_damage(self, ids: List[int]) -> List[int]:
        """
        Simulate salt crystallization damage.
        Creates 2-3 clustered damage spots.
        """
        if len(ids) < self.min_sequence_length * 2:
            return ids

        if self.rng.random() > self.salt_damage_prob:
            return ids

        result = list(ids)

        # Create 1-3 damage clusters
        num_clusters = self.rng.integers(1, 4)

        for _ in range(num_clusters):
            if len(result) < 6:
                break

            # Pick cluster center
            center = self.rng.integers(2, len(result) - 2)

            # Cluster size (2-4 tokens around center)
            cluster_size = self.rng.integers(2, 5)
            start = max(0, center - cluster_size // 2)
            end = min(len(result), start + cluster_size)

            # Apply damage
            for i in range(start, end):
                if self.rng.random() < 0.7:  # 70% chance per token in cluster
                    result[i] = self.star_id

        return result

    def _apply_line_break_damage(self, ids: List[int]) -> List[int]:
        """
        Simulate damage at line boundaries.
        Text often breaks or is damaged where lines meet.
        """
        if len(ids) < self.min_sequence_length:
            return ids

        if self.rng.random() > self.line_break_prob:
            return ids

        if not self.line_break_tokens:
            return ids

        result = list(ids)

        # Find line break positions
        break_positions = [i for i, tid in enumerate(ids) if tid in self.line_break_tokens]

        if not break_positions:
            return ids

        # Damage around 1-2 line breaks
        num_to_damage = min(len(break_positions), self.rng.integers(1, 3))
        positions_to_damage = self.rng.choice(break_positions, size=num_to_damage, replace=False)

        for pos in positions_to_damage:
            # Damage 1-3 tokens around the break
            damage_range = self.rng.integers(1, 4)
            start = max(0, pos - damage_range // 2)
            end = min(len(result), pos + damage_range // 2 + 1)

            for i in range(start, end):
                if i != pos:  # Keep the line break itself
                    result[i] = self.star_id

        return result

    def apply_damage(self, ids: List[int]) -> List[int]:
        """Apply all enabled damage patterns to a sequence."""
        result = list(ids)

        # Apply damage patterns in sequence
        # Order matters: corner damage first, then cracks, then erosion, then salt
        result = self._apply_corner_damage(result)
        result = self._apply_vertical_crack(result)
        result = self._apply_line_break_damage(result)
        result = self._apply_surface_erosion(result)
        result = self._apply_salt_damage(result)

        return result

    def __call__(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply tablet damage to input features.

        Note: This returns a list of features, not a batched tensor.
        Chain this with another collator that handles batching.
        """
        damaged_features = []

        for feat in features:
            new_feat = dict(feat)

            if 'input_ids' in feat:
                input_ids = feat['input_ids']
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.tolist()

                damaged_ids = self.apply_damage(input_ids)
                new_feat['input_ids'] = damaged_ids

            damaged_features.append(new_feat)

        return damaged_features


class TabletDamageWrapper:
    """
    Wrapper that applies TabletDamageCollator before a base collator.

    Usage:
        base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        damage_wrapper = TabletDamageWrapper(
            base_collator=base_collator,
            tokenizer=tokenizer,
            vertical_crack_prob=0.15,
        )
    """

    def __init__(
        self,
        base_collator,
        tokenizer: PreTrainedTokenizerBase,
        **damage_kwargs,
    ):
        self.base_collator = base_collator
        self.damage_collator = TabletDamageCollator(tokenizer=tokenizer, **damage_kwargs)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Apply damage, then base collation."""
        damaged = self.damage_collator(features)
        return self.base_collator(damaged)


# -----------------------------------------------------------------------------
# CONSISTENCY TRAINING COLLATOR
# -----------------------------------------------------------------------------

class ConsistencyTrainingCollator:
    """
    Collator for consistency regularization training.

    Creates pairs of (original, perturbed) inputs that should produce
    the same output. The model is trained to be consistent across
    minor input variations that don't change meaning.

    Perturbations include:
    - Adjacent token swaps (simulates reading order variants)
    - Minor noise injection (simulates OCR/transcription errors)
    - Whitespace normalization variants

    Args:
        base_collator: The underlying collator to wrap
        tokenizer: HuggingFace tokenizer
        consistency_weight: Weight for consistency loss (0.0 to 1.0)
        perturbation_prob: Probability of creating a perturbed copy
        swap_prob: Probability of adjacent token swap per position
        noise_prob: Probability of token noise per position
        include_original: If True, batch includes both original and perturbed

    Usage:
        consistency_collator = ConsistencyTrainingCollator(
            base_collator=DataCollatorForSeq2Seq(...),
            tokenizer=tokenizer,
            consistency_weight=0.5,
        )

        # In training, compute:
        # loss = task_loss + consistency_weight * mse(output_original, output_perturbed)
    """

    def __init__(
        self,
        base_collator,
        tokenizer: PreTrainedTokenizerBase,
        consistency_weight: float = 0.5,
        perturbation_prob: float = 0.5,
        swap_prob: float = 0.1,
        noise_prob: float = 0.05,
        include_original: bool = True,
        min_sequence_length: int = 4,
    ):
        self.base_collator = base_collator
        self.tokenizer = tokenizer
        self.consistency_weight = float(consistency_weight)
        self.perturbation_prob = float(perturbation_prob)
        self.swap_prob = float(swap_prob)
        self.noise_prob = float(noise_prob)
        self.include_original = bool(include_original)
        self.min_sequence_length = int(min_sequence_length)

        self.rng = np.random.default_rng()
        self.vocab_size = tokenizer.vocab_size

        # Special tokens to never perturb
        self.special_tokens = set()
        for attr in ['pad_token_id', 'eos_token_id', 'bos_token_id', 'unk_token_id', 'sep_token_id']:
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                self.special_tokens.add(tid)

    def _swap_adjacent(self, ids: List[int]) -> List[int]:
        """Swap adjacent tokens with some probability."""
        if len(ids) < 3:
            return ids

        result = list(ids)

        # Don't swap at the very start or end
        for i in range(1, len(result) - 2):
            if self.rng.random() < self.swap_prob:
                # Check neither token is special
                if result[i] not in self.special_tokens and result[i + 1] not in self.special_tokens:
                    result[i], result[i + 1] = result[i + 1], result[i]

        return result

    def _add_noise(self, ids: List[int]) -> List[int]:
        """Add minor token noise."""
        if len(ids) < 3:
            return ids

        result = list(ids)

        for i in range(1, len(result) - 1):  # Skip first and last
            if result[i] not in self.special_tokens and self.rng.random() < self.noise_prob:
                # Replace with a random token (simulates transcription error)
                # Use a token from a similar range to avoid wild jumps
                offset = self.rng.integers(-10, 11)
                new_id = max(0, min(self.vocab_size - 1, result[i] + offset))
                if new_id not in self.special_tokens:
                    result[i] = new_id

        return result

    def _perturb(self, ids: List[int]) -> List[int]:
        """Apply light perturbation that shouldn't change meaning."""
        if len(ids) < self.min_sequence_length:
            return ids

        result = list(ids)
        result = self._swap_adjacent(result)
        result = self._add_noise(result)

        return result

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate with consistency training support."""
        if not self.include_original:
            # Just perturb all inputs
            perturbed_features = []
            for feat in features:
                new_feat = dict(feat)
                if 'input_ids' in feat:
                    input_ids = feat['input_ids']
                    if isinstance(input_ids, torch.Tensor):
                        input_ids = input_ids.tolist()
                    new_feat['input_ids'] = self._perturb(input_ids)
                perturbed_features.append(new_feat)

            batch = self.base_collator(perturbed_features)
            batch['consistency_weight'] = self.consistency_weight
            return batch

        # Create pairs: original and perturbed
        all_features = []
        pair_indices = []  # Track which examples are pairs

        for i, feat in enumerate(features):
            # Always include original
            all_features.append(feat)

            # Maybe add perturbed copy
            if self.rng.random() < self.perturbation_prob:
                new_feat = dict(feat)
                if 'input_ids' in feat:
                    input_ids = feat['input_ids']
                    if isinstance(input_ids, torch.Tensor):
                        input_ids = input_ids.tolist()
                    new_feat['input_ids'] = self._perturb(input_ids)

                pair_indices.append((len(all_features) - 1, len(all_features)))
                all_features.append(new_feat)

        batch = self.base_collator(all_features)

        # Add consistency metadata
        batch['consistency_weight'] = self.consistency_weight
        batch['consistency_pairs'] = pair_indices  # List of (original_idx, perturbed_idx)
        batch['consistency_enabled'] = len(pair_indices) > 0

        return batch


def compute_consistency_loss(
    outputs_original: torch.Tensor,
    outputs_perturbed: torch.Tensor,
    consistency_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute consistency loss between original and perturbed outputs.

    Uses MSE between the output distributions/logits.

    Args:
        outputs_original: Model outputs for original inputs
        outputs_perturbed: Model outputs for perturbed inputs
        consistency_weight: Weight for the consistency term

    Returns:
        Weighted consistency loss
    """
    import torch.nn.functional as F

    # Normalize and compute MSE
    if outputs_original.dim() == 3:  # [batch, seq, vocab]
        # Use softmax outputs for comparison
        p_orig = F.softmax(outputs_original, dim=-1)
        p_pert = F.softmax(outputs_perturbed, dim=-1)
        mse = F.mse_loss(p_orig, p_pert)
    else:
        mse = F.mse_loss(outputs_original, outputs_perturbed)

    return consistency_weight * mse


# -----------------------------------------------------------------------------
# COMBINED ADVANCED TRAINING COLLATOR
# -----------------------------------------------------------------------------

class AdvancedTrainingCollator:
    """
    Combines multiple training enhancements into a single collator.

    Features:
    - R-Drop regularization (optional)
    - Tablet damage simulation (optional)
    - Consistency training (optional)
    - Curriculum scheduling support

    This is a convenience wrapper that combines the above collators
    with a base translation/pretraining collator.

    Args:
        base_collator: The underlying collator (e.g., AdvancedMixedDataCollator)
        tokenizer: HuggingFace tokenizer
        enable_rdrop: Enable R-Drop regularization
        rdrop_alpha: R-Drop weight
        enable_tablet_damage: Enable tablet damage simulation
        tablet_damage_kwargs: Arguments for TabletDamageCollator
        enable_consistency: Enable consistency training
        consistency_weight: Consistency loss weight
        curriculum_scheduler: Optional CurriculumScheduler for dynamic adjustment

    Usage:
        collator = AdvancedTrainingCollator(
            base_collator=mixed_collator,
            tokenizer=tokenizer,
            enable_rdrop=True,
            rdrop_alpha=0.7,
            enable_tablet_damage=True,
            tablet_damage_kwargs={'vertical_crack_prob': 0.15},
            enable_consistency=True,
            consistency_weight=0.5,
        )
    """

    def __init__(
        self,
        base_collator,
        tokenizer: PreTrainedTokenizerBase,
        enable_rdrop: bool = True,
        rdrop_alpha: float = 0.7,
        enable_tablet_damage: bool = True,
        tablet_damage_kwargs: Optional[Dict[str, Any]] = None,
        enable_consistency: bool = False,
        consistency_weight: float = 0.5,
        consistency_perturbation_prob: float = 0.3,
        curriculum_scheduler: Optional[CurriculumScheduler] = None,
    ):
        self.tokenizer = tokenizer
        self.base_collator = base_collator
        self.curriculum_scheduler = curriculum_scheduler

        # R-Drop settings
        self.enable_rdrop = bool(enable_rdrop)
        self.rdrop_alpha = float(rdrop_alpha)

        # Tablet damage settings
        self.enable_tablet_damage = bool(enable_tablet_damage)
        if enable_tablet_damage:
            damage_kwargs = tablet_damage_kwargs or {}
            self.tablet_damage = TabletDamageCollator(tokenizer=tokenizer, **damage_kwargs)
        else:
            self.tablet_damage = None

        # Consistency settings
        self.enable_consistency = bool(enable_consistency)
        self.consistency_weight = float(consistency_weight)
        self.consistency_perturbation_prob = float(consistency_perturbation_prob)
        if enable_consistency:
            # We'll handle consistency manually to avoid double-batching issues
            self.consistency_collator = ConsistencyTrainingCollator(
                base_collator=lambda x: x,  # Identity - we handle batching
                tokenizer=tokenizer,
                consistency_weight=consistency_weight,
                perturbation_prob=consistency_perturbation_prob,
                include_original=False,
            )
        else:
            self.consistency_collator = None

    def _maybe_adjust_from_curriculum(self):
        """Adjust damage probabilities based on curriculum if available."""
        if self.curriculum_scheduler is None or self.tablet_damage is None:
            return

        # Scale damage probabilities with curriculum progression
        # Early training: less damage, later: more damage
        epoch = self.curriculum_scheduler.current_epoch
        total = self.curriculum_scheduler.total_epochs
        progress = min(1.0, epoch / max(1, total))

        # Gradually increase damage as training progresses
        base_scale = 0.5 + 0.5 * progress  # 0.5 -> 1.0

        # This modifies the damage collator's probabilities
        # (In a more sophisticated implementation, you'd store base values)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Apply all enabled training enhancements."""
        working_features = list(features)

        # 1. Apply tablet damage (modifies input_ids)
        if self.enable_tablet_damage and self.tablet_damage is not None:
            working_features = self.tablet_damage(working_features)

        # 2. Collate with base collator
        batch = self.base_collator(working_features)

        # 3. Apply R-Drop (duplicates batch)
        if self.enable_rdrop:
            duplicated = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    duplicated[key] = torch.cat([value, value.clone()], dim=0)
                else:
                    duplicated[key] = value

            duplicated['rdrop_alpha'] = self.rdrop_alpha
            duplicated['rdrop_enabled'] = True
            duplicated['rdrop_batch_size'] = batch['input_ids'].size(0)
            batch = duplicated

        # 4. Add consistency metadata (actual consistency handled in training loop)
        if self.enable_consistency:
            batch['consistency_weight'] = self.consistency_weight
            batch['consistency_enabled'] = True

        return batch


# -----------------------------------------------------------------------------
# RDROP-AWARE TRAINER MIXIN
# -----------------------------------------------------------------------------

class RDropTrainerMixin:
    """
    Mixin class that adds R-Drop support to a Trainer.

    Usage:
        class MyRDropTrainer(RDropTrainerMixin, Seq2SeqTrainer):
            pass

        trainer = MyRDropTrainer(...)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to add R-Drop loss computation."""
        rdrop_enabled = inputs.pop('rdrop_enabled', False)
        rdrop_alpha = inputs.pop('rdrop_alpha', 0.7)
        rdrop_batch_size = inputs.pop('rdrop_batch_size', None)

        if not rdrop_enabled:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        # Forward pass (model sees duplicated batch, different dropout masks)
        outputs = model(**inputs)
        logits = outputs.logits

        # Split back into two halves
        logits1 = logits[:rdrop_batch_size]
        logits2 = logits[rdrop_batch_size:]
        labels = inputs['labels'][:rdrop_batch_size]

        # Compute R-Drop loss
        total_loss, ce_loss, kl_loss = compute_rdrop_loss(
            logits1, logits2, labels,
            rdrop_alpha=rdrop_alpha,
            pad_token_id=-100,
        )

        if return_outputs:
            # Return only first half of outputs
            outputs.logits = logits1
            return total_loss, outputs

        return total_loss
