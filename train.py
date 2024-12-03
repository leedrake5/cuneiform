###Launch without memory restrictions on a Mac
#PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

###Launch with torch distributed launch
#python -m torch.distributed.launch --nproc_per_node=2 train.py

###Using Accelerate
#/home/bly/anaconda3/envs/torch/bin/accelerate config
#accelerate launch train.py


import sys, os, datetime, pwd
user_directory = pwd.getpwuid(os.getuid()).pw_dir
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
import torch
import random
import glob
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import DatasetDict, Dataset
from evaluate import load as load_metric
import accelerate
from accelerate import Accelerator

import numpy as np

import re
import requests
import unicodedata

from transformers import Seq2SeqTrainer,Seq2SeqTrainingArguments, EarlyStoppingCallback, BertTokenizer,MT5ForConditionalGeneration
from transformers.data.data_collator import DataCollatorForSeq2Seq,default_data_collator
import pandas as pd
import math,os
import numpy as np
from tqdm import tqdm
import torch

import os
os.chdir(os.path.join(user_directory, 'GitHub', 'cuneiform'))


import logging
logger = logging.getLogger('pytorch_lightning.utilities.rank_zero')
class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        return 'available:' not in record.getMessage()

logger.addFilter(IgnorePLFilter())

source_langs = set(["akk"])
target_langs = set(["en"])

def get_finetune_model_id(model_id):
    model_dir = f"../results/{model_id}"
    checkpoints = [(os.path.abspath(x), int(os.path.split(x)[1].split("-")[1])) for x in glob.glob(f"{model_dir}/checkpoint-*")]
    checkpoints = sorted(checkpoints, key=lambda x: x[1])[-1]
    return checkpoints[0]

#os.environ["WANDB_NOTEBOOK_NAME"] = "TrainTranslator.ipynb"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

base_model_id = "google/mt5-large"
finetune_model_id = None
# finetune_model_id = get_finetune_model_id("t5-base-p-akksux-en-20220722-173018")

#model_max_length = 512
#batch_size = 8 if os.path.basename(base_model_id).startswith("t5-base") else 128


is_bi = False
use_paragraphs = True
use_lines = True
is_finetune = finetune_model_id is not None and len(finetune_model_id) > 1

date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
flags = ""
suffix = ""
if is_bi:
    flags += "-bi"

if use_paragraphs:
    flags += "-p"

if use_lines:
    flags += "-l"

if is_finetune:
    flags += "-f"
    suffix += f"-{os.path.basename(os.path.split(finetune_model_id)[0])}-{os.path.basename(finetune_model_id)}"

model_id = f"{os.path.basename(base_model_id)}{flags}-{''.join(sorted(list(source_langs)))}-{''.join(sorted(list(target_langs)))}-{date_id}{suffix}"
model_id


device = torch.device("cuda" if torch.cuda.is_available() else "mps")


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString_en(s, use_prefix=False, task="Translate", target="cuneiform", type="simple", language="Akkadian"):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    s = s.strip()
    if use_prefix:
        if task=="Translate":
            if target=="cuneiform":
                return 'Translate English to ' + language + ' cuneiform: ' + s
            elif target=="transliteration":
                if type == "simple":
                    return 'Translate English to simple ' + language + ' transliteration: ' + s
                elif type == "group":
                    return 'Translate English to grouped ' + language + ' transliteration: ' + s
                elif type == "origional":
                    return 'Translate English to complex ' + language + ' transliteration: ' + s
    else:
        return s


# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate(s, use_prefix=True, type="simple", language="Akkadian"):
    if type == "simple":
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    elif type == "origional":
        s = unicodeToAscii(s.lower().strip())
        s = s.translate(str.maketrans({'[': None, ']': None, '-': ' '}))
    normalized_string = s.strip()
    if use_prefix:
        if type == "simple":
            return 'Transliterate ' + language + ' cuneiform to simple Latin characters: ' + normalized_string
        elif type == "origional":
            return 'Transliterate ' + language + ' cuneiform to complex Latin characters: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_rev_transliterate(s, use_prefix=True, type="simple", language="Akkadian"):
    if type == "simple":
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    elif type == "origional":
        s = unicodeToAscii(s.lower().strip())
        s = s.translate(str.maketrans({'[': None, ']': None, '-': ' '}))
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
    normalized_string = s.strip()
    if use_prefix:
        if type == "simple" :
            return 'Convert simple transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
        elif type == "group":
            return 'Convert grouped transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
        elif type == "origional" :
            return 'Convert complex transliterated Latin characters to ' + language + ' cuneiform: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate_translate(s, use_prefix=True, task="Translate", type="simple", language="Akkadian"):
    if type=="simple":
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    elif type=="origional":
        s = unicodeToAscii(s.lower().strip())
        s = s.translate(str.maketrans({'[': None, ']': None, '-': ' '}))
    elif type == "group":
        s = unicodeToAscii(s.lower().strip())
    normalized_string = s.strip()
    if use_prefix:
        if task == "Translate":
            if type == "simple":
                return 'Translate simple ' + language + ' transliteration to English: ' + normalized_string
            elif type == "origional":
                return 'Translate complex ' + language + ' transliteration to English: ' + normalized_string
            elif type == "group":
                return 'Translate grouped ' + language + ' transliteration to English: ' + normalized_string
        elif task == "Group":
            if type == "simple":
                return 'Group simple ' + language + ' transliteration into likely words: ' + normalized_string
            elif type == "origional":
                return 'Group complex ' + language + ' transliteration into likely words: ' + normalized_string
    else:
        return normalized_string

# Lowercase, trim, and remove non-letter characters
def normalizeString_cuneiform_transliterate_minimal(s, use_prefix=True, language="Akkadian"):
    s = unicodeToAscii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    normalized_string = s.strip()
    if use_prefix:
        return 'Translate ' + language + ' grouped transliteration to English: ' + normalized_string
    else:
        return normalized_string

def normalizeString_cuneiform(s, use_prefix=True, task="Translate", type="simple", language="Akkadian"):
    # Optional: Remove unwanted modern characters, if any (adjust regex as needed)
    # s = re.sub(r'[^\u12000-\u123FF\u12400-\u1247F]+', '', s)  # Adjust Unicode ranges to cuneiform and related characters
    # Split each character/sign into separate entries
    # This assumes each character in the string is a distinct sign, no need to join with spaces if already separated
    normalized_string = ' '.join(s)  # This joins every character with a space, treating each as a separate token
    # Add the prefix if use_prefix is True
    if use_prefix:
        if task == "Translate":
            return 'Translate ' + language + ' cuneiform to English: ' + normalized_string
        elif task == "Transliterate":
            if type == "simple":
                return 'Transliterate ' + language + ' cuneiform to simple Latin characters: ' + normalized_string
            elif type == "group":
                return 'Transliterate ' + language + ' cuneiform to grouped Latin characters: ' + normalized_string
            elif type == "origional":
                return 'Transliterate ' + language + ' cuneiform to complex Latin characters: ' + normalized_string
    else:
        return normalized_string

def read_and_process_file(file_path):
    # Check if the file_path is a URL
    if file_path.startswith('http://') or file_path.startswith('https://'):
        # Fetch the content from the URL
        response = requests.get(file_path)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        lines = response.text.strip().split('\n')
    else:
        # Open the local file and read the lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.read().strip().split('\n')
    # Replace ". . . " with "*" in each line
    processed_lines = [re.sub(r'\s*\.\s*\.\s*\.\s*', '*', line) for line in lines]
    return processed_lines

def convert(lst):
   res_dict = {}
   for i in range(0, len(lst), 2):
       res_dict[lst[i]] = lst[i + 1]
   return res_dict

###For Pretrain
def trim_singles(pairs, max_length1, max_length2, max_length_threshold, min_length_threshold):
    # Filter out pairs where either element exceeds the word count limit
    max_filtered_pairs = [
        pair for pair in pairs
        if len(pair[0].split()) <= max_length_threshold
    ]
    min_filtered_pairs = [
        pair for pair in max_filtered_pairs
        if len(pair[0].split()) >= min_length_threshold
    ]
    # Trim each element in the pair to the maximum character length
    trimmed_pairs = [
        (s1[:max_length1]) for s1 in min_filtered_pairs
    ]
    return trimmed_pairs

###For Train
def trim_pairs(pairs, max_length1, max_length2, max_length_threshold, min_length_threshold):
    # Filter out pairs where either element exceeds the word count limit
    max_filtered_pairs = [
        pair for pair in pairs
        if len(pair[0].split()) <= max_length_threshold and len(pair[1].split()) <= (max_length_threshold - 5)
    ]
    min_filtered_pairs = [
        pair for pair in max_filtered_pairs
        if len(pair[0].split()) >= min_length_threshold and len(pair[1].split()) >= min_length_threshold
    ]
    # Trim each element in the pair to the maximum character length
    trimmed_pairs = [
        (s1[:max_length1], s2[:max_length2]) for s1, s2 in min_filtered_pairs
    ]
    return trimmed_pairs


def readLangsTrain(max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=True):
    print("Reading lines...")
    ##############
    ###Akkadian###
    ##############
    # Read the file and split into lines
    akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_train.cu'))
    akk_transcription_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_train.tr'))
    akk_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_train.en'))
    akk_cuneiform_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_test.cu'))
    akk_transcription_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_test.tr'))
    akk_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_test.en'))
    akk_cuneiform_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_valid.cu'))
    akk_transcription_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_valid.tr'))
    akk_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_valid.en'))
    # Split every line into pairs and normalize
    ###Translate from Akkadian cuneiform to English
    akk_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
    akk_test_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
    akk_val_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Translate", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
    if debug == True:
        print(f"", {akk_train_pairs_cuneiform_translate[1][0]}, " -> ", {akk_train_pairs_cuneiform_translate[1][1]})
    ###Translate from simple transliterated Akkadian to English
    akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_train[l], use_prefix=True, task="Translate", type="simple", language="unknown language"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_test[l], use_prefix=True, task="Translate", type="simple", language="unknown language"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_val[l], use_prefix=True, task="Translate", type="simple", language="unknown language"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
    if debug == True:
        print(f"", {akk_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_simple_transliterated_translate[1][1]})
    ###Translate from origional transliterated Akkadian to English
    akk_train_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_train[l], use_prefix=True, task="Translate", type="origional", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_test[l], use_prefix=True, task="Translate", type="origional", language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_val[l], use_prefix=True, task="Translate", type="origional", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
    if debug == True:
        print(f"", {akk_train_pairs_origional_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_origional_transliterated_translate[1][1]})
    ###Translate from grouped transliterated Akkadian to English
    akk_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transcription_train[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transcription_test[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transcription_val[l], use_prefix=True, language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
    if debug == True:
        print(f"", {akk_train_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_train_pairs_group_transliterated_translate[1][1]})
    ###Transliterate from Akkadian Cuenfirom to simple Latin characters
    akk_train_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="simple", language="poor Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
    akk_test_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="simple", language="poor Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
    akk_val_pairs_transliterated_simple = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="poor Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
    if debug == True:
        print(f"", {akk_train_pairs_transliterate_simple[1][0]}, " -> ", {akk_train_pairs_transliterate_simple[1][1]})
    ###Transliterate from Akkadian Cuenfirom to complex Latin characters
    akk_train_pairs_transliterate_origional = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_train[l], use_prefix=False, type="origional")] for l in range(len(akk_cuneiform_train))]
    akk_test_pairs_transliterate_origional = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_test[l], use_prefix=False, type="origional")] for l in range(len(akk_cuneiform_test))]
    akk_val_pairs_transliterated_origional = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_val[l], use_prefix=False, type="origional")] for l in range(len(akk_cuneiform_val))]
    if debug == True:
        print(f"", {akk_train_pairs_transliterate_origional[1][0]}, " -> ", {akk_train_pairs_transliterate_origional[1][1]})
    ###Transliterate from Akkadian Cuenfirom to grouped Latin characters
    akk_train_pairs_transliterate_group = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_train[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_train))]
    akk_test_pairs_transliterate_group = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_test[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_test))]
    akk_val_pairs_transliterated_group = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_val[l], use_prefix=False, type="group")] for l in range(len(akk_cuneiform_val))]
    if debug == True:
        print(f"", {akk_train_pairs_transliterate_group[1][0]}, " -> ", {akk_train_pairs_transliterate_group[1][1]})
    ###Group simple transliterated Akkadian into words
    akk_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_train[l], use_prefix=True, task="Group", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
    akk_test_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_test[l], use_prefix=True, task="Group", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
    akk_val_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_val[l], use_prefix=True, task="Group", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
    if debug == True:
        print(f"", {akk_train_pairs_group_simple_transliterate[1][0]}, " -> ", {akk_train_pairs_group_simple_transliterate[1][1]})
    ###Group complex transliterated Akkadian into words
    akk_train_pairs_group_origional_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_train[l], use_prefix=True, task="Group", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
    akk_test_pairs_group_origional_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_test[l], use_prefix=True, task="Group", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
    akk_val_pairs_group_origional_transliterate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_val[l], use_prefix=True, task="Group", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
    if debug == True:
        print(f"", {akk_train_pairs_group_origional_transliterate[1][0]}, " -> ", {akk_train_pairs_group_origional_transliterate[1][1]})
    ###Translate from English to cuneiform Akkadian
    akk_train_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
    akk_test_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
    akk_val_rev_pairs_cuneiform_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="cuneiform", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False), ] for l in range(len(akk_cuneiform_val))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_cuneiform_translate[1][0]}, " -> ", {akk_train_rev_pairs_cuneiform_translate[1][1]})
    ###Translate from English to simple transliterated Akkadian
    akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="unknown language"), normalizeString_cuneiform_transliterate(akk_transcription_train[l], use_prefix=False, type="simple")] for l in range(len(akk_en_train))]
    akk_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="unknown language"), normalizeString_cuneiform_transliterate(akk_transcription_test[l], use_prefix=False, type="simple")] for l in range(len(akk_en_test))]
    akk_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="unknown language"), normalizeString_cuneiform_transliterate(akk_transcription_val[l], use_prefix=False, type="simple")] for l in range(len(akk_en_val))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_simple_transliterated_translate[1][1]})
    ###Translate from English to complex transliterated Akkadian
    akk_train_rev_pairs_origional_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_train[l], use_prefix=False, type="origional")] for l in range(len(akk_en_train))]
    akk_test_rev_pairs_origional_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_test[l], use_prefix=False, type="origional")] for l in range(len(akk_en_test))]
    akk_val_rev_pairs_origional_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="origional", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_val[l], use_prefix=False, type="origional")] for l in range(len(akk_en_val))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_origional_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_origional_transliterated_translate[1][1]})
    ###Translate from English to grouped transliterated Akkadian
    akk_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_train[l], use_prefix=False)] for l in range(len(akk_en_train))]
    akk_test_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_test[l], use_prefix=False)] for l in range(len(akk_en_test))]
    akk_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Akkadian"), normalizeString_cuneiform_transliterate_minimal(akk_transcription_val[l], use_prefix=False)] for l in range(len(akk_en_val))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {akk_train_rev_pairs_group_transliterated_translate[1][1]})
    ###Convert from simple transliterated Akkadian to cuneiform
    akk_train_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_train[l], use_prefix=True, type="simple", language="poor Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_test[l], use_prefix=True, type="simple", language="poor Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_val[l], use_prefix=True, type="simple", language="poor Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_transliterate_simple[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_simple[1][1]})
    ###Convert from grouped transliterated Akkadian to cuneiform
    akk_train_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_train[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_test[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_rev_pairs_transliterate_group = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_val[l], use_prefix=True, type="group", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_transliterate_group[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_group[1][1]})
    ###Convert from complex transliterated Akkadian to cuneiform
    akk_train_rev_pairs_transliterate_origional = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_train[l], use_prefix=True, type="origional", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_rev_pairs_transliterate_origional = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_test[l], use_prefix=True, type="origional", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_rev_pairs_transliterate_origional = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_val[l], use_prefix=True, type="origional", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_transliterate_origional[1][0]}, " -> ", {akk_train_rev_pairs_transliterate_origional[1][1]})
    ##############
    ###Sumerian###
    ##############
    # Read the file and split into lines
    sux_transcription_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_train.tr'))
    sux_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_train.en'))
    sux_transcription_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_test.tr'))
    sux_en_test = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_test.en'))
    sux_transcription_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_valid.tr'))
    sux_en_val = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_valid.en'))
    # Split every line into pairs and normalize
    ##Translate from simple transliterated Sumerian to English
    sux_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_train[l], use_prefix=True, task="Translate", type="simple", language="unknown language"), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_transcription_train))]
    sux_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_test[l], use_prefix=True, task="Translate", type="simple", language="unknown language"), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_transcription_test))]
    sux_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_val[l], use_prefix=True, task="Translate", type="simple", language="unknown language"), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_transcription_val))]
    if debug == True:
        print(f"", {sux_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {sux_train_pairs_simple_transliterated_translate[1][1]})
    ###Translate from origional transliterated Sumerian to English
    sux_train_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_train[l], use_prefix=True, task="Translate", type="origional", language="Sumerian"), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_transcription_train))]
    sux_test_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_test[l], use_prefix=True, task="Translate", type="origional", language="Sumerian"), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_transcription_test))]
    sux_val_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_val[l], use_prefix=True, task="Translate", type="origional", language="Sumerian"), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_transcription_val))]
    if debug == True:
        print(f"", {sux_train_pairs_origional_transliterated_translate[1][0]}, " -> ", {sux_train_pairs_origional_transliterated_translate[1][1]})
    ###Translate from grouped transliterated Sumerian to English
    sux_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transcription_train[l], use_prefix=True, language="Sumerian"), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_transcription_train))]
    sux_test_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transcription_test[l], use_prefix=True, language="Sumerian"), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_transcription_test))]
    sux_val_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transcription_val[l], use_prefix=True, language="Sumerian"), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_transcription_val))]
    if debug == True:
        print(f"", {sux_train_pairs_group_transliterated_translate[1][0]}, " -> ", {sux_train_pairs_group_transliterated_translate[1][1]})
    ###Group simple transliterated Sumerian into words
    sux_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_train[l], use_prefix=True, task="Group", type="simple", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_train[l], use_prefix=False)] for l in range(len(sux_transcription_train))]
    sux_test_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_test[l], use_prefix=True, task="Group", type="simple", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_test[l], use_prefix=False)] for l in range(len(sux_transcription_test))]
    sux_val_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_val[l], use_prefix=True, task="Group", type="simple", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_val[l], use_prefix=False)] for l in range(len(sux_transcription_val))]
    if debug == True:
        print(f"", {sux_train_pairs_group_simple_transliterate[1][0]}, " -> ", {sux_train_pairs_group_simple_transliterate[1][1]})
    ###Group complex transliterated Sumerian into words
    sux_train_pairs_group_origional_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_train[l], use_prefix=True, task="Group", type="origional", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_train[l], use_prefix=False)] for l in range(len(sux_transcription_train))]
    sux_test_pairs_group_origional_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_test[l], use_prefix=True, task="Group", type="origional", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_test[l], use_prefix=False)] for l in range(len(sux_transcription_test))]
    sux_val_pairs_group_origional_transliterate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_val[l], use_prefix=True, task="Group", type="origional", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_val[l], use_prefix=False)] for l in range(len(sux_transcription_val))]
    if debug == True:
        print(f"", {sux_train_pairs_group_origional_transliterate[1][0]}, " -> ", {sux_train_pairs_group_origional_transliterate[1][1]})
    ###Translate from English to simple transliterated Sumerian
    sux_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="unknown language"), normalizeString_cuneiform_transliterate(sux_transcription_train[l], use_prefix=False, type="simple")] for l in range(len(sux_en_train))]
    sux_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="unknown language"), normalizeString_cuneiform_transliterate(sux_transcription_test[l], use_prefix=False, type="simple")] for l in range(len(sux_en_test))]
    sux_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="unknown language"), normalizeString_cuneiform_transliterate(sux_transcription_val[l], use_prefix=False, type="simple")] for l in range(len(sux_en_val))]
    if debug == True:
        print(f"", {sux_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {sux_train_rev_pairs_simple_transliterated_translate[1][1]})
    ###Translate from English to complex transliterated Sumerian
    sux_train_rev_pairs_origional_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="origional", language="Sumerian"), normalizeString_cuneiform_transliterate(sux_transcription_train[l], use_prefix=False, type="origional")] for l in range(len(sux_en_train))]
    sux_test_rev_pairs_origional_transliterated_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="origional", language="Sumerian"), normalizeString_cuneiform_transliterate(sux_transcription_test[l], use_prefix=False, type="origional")] for l in range(len(sux_en_test))]
    sux_val_rev_pairs_origional_transliterated_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="origional", language="Sumerian"), normalizeString_cuneiform_transliterate(sux_transcription_val[l], use_prefix=False, type="origional")] for l in range(len(sux_en_val))]
    if debug == True:
        print(f"", {sux_train_rev_pairs_origional_transliterated_translate[1][0]}, " -> ", {sux_train_rev_pairs_origional_transliterated_translate[1][1]})
    ###Translate from English to grouped transliterated Sumerian
    sux_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_train[l], use_prefix=False)] for l in range(len(sux_en_train))]
    sux_test_rev_pairs_group_transliterated_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_test[l], use_prefix=False)] for l in range(len(sux_en_test))]
    sux_val_rev_pairs_group_transliterated_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Sumerian"), normalizeString_cuneiform_transliterate_minimal(sux_transcription_val[l], use_prefix=False)] for l in range(len(sux_en_val))]
    if debug == True:
        print(f"", {sux_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {sux_train_rev_pairs_group_transliterated_translate[1][1]})
    ###############
    ###Elamite###
    ###############
    # Read the file and split into lines
    elx_transcription_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'e_train.tr'))
    elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'e_train.en'))
    # Split every line into pairs and normalize
    ##Translate from simple transliterated Elamite to English
    elx_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transcription_train[l], use_prefix=True, task="Translate", type="simple", language="unknown language"), normalizeString_en(elx_en_train[l], use_prefix=False)] for l in range(len(elx_transcription_train))]
    if debug == True:
        print(f"", {elx_train_pairs_simple_transliterated_translate[1][0]}, " -> ", {elx_train_pairs_simple_transliterated_translate[1][1]})
    ###Translate from origional transliterated Elamite to English
    elx_train_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transcription_train[l], use_prefix=True, task="Translate", type="origional", language="Elamite"), normalizeString_en(elx_en_train[l], use_prefix=False)] for l in range(len(elx_transcription_train))]
    if debug == True:
        print(f"", {elx_train_pairs_origional_transliterated_translate[1][0]}, " -> ", {elx_train_pairs_origional_transliterated_translate[1][1]})
    ###Translate from grouped transliterated Elamite to English
    elx_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(elx_transcription_train[l], use_prefix=True, language="Elamite"), normalizeString_en(elx_en_train[l], use_prefix=False)] for l in range(len(elx_transcription_train))]
    if debug == True:
       print(f"", {elx_train_pairs_group_transliterated_translate[1][0]}, " -> ", {elx_train_pairs_group_transliterated_translate[1][1]})
    ###Group simple transliterated Elamite into words
    elx_train_pairs_group_simple_transliterate = [[normalizeString_cuneiform_transliterate_translate(elx_transcription_train[l], use_prefix=True, task="Group", type="simple", language="Elamite"), normalizeString_cuneiform_transliterate_minimal(elx_transcription_train[l], use_prefix=False)] for l in range(len(elx_transcription_train))]
    if debug == True:
        print(f"", {elx_train_pairs_group_simple_transliterate[1][0]}, " -> ", {elx_train_pairs_group_simple_transliterate[1][1]})
    ###Group complex transliterated Elamite into words
    elx_train_pairs_group_origional_transliterate = [[normalizeString_cuneiform_transliterate_translate(elx_transcription_train[l], use_prefix=True, task="Group", type="origional", language="Elamite"), normalizeString_cuneiform_transliterate_minimal(elx_transcription_train[l], use_prefix=False)] for l in range(len(elx_transcription_train))]
    if debug == True:
        print(f"", {elx_train_pairs_group_origional_transliterate[1][0]}, " -> ", {elx_train_pairs_group_origional_transliterate[1][1]})
    ###Translate from English to simple transliterated Elamite
    elx_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="unknown language"), normalizeString_cuneiform_transliterate(elx_transcription_train[l], use_prefix=False, type="simple")] for l in range(len(elx_en_train))]
    if debug == True:
        print(f"", {elx_train_rev_pairs_simple_transliterated_translate[1][0]}, " -> ", {elx_train_rev_pairs_simple_transliterated_translate[1][1]})
    ###Translate from English to complex transliterated Elamite
    elx_train_rev_pairs_origional_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="origional", language="Elamite"), normalizeString_cuneiform_transliterate(elx_transcription_train[l], use_prefix=False, type="origional")] for l in range(len(elx_en_train))]
    if debug == True:
        print(f"", {elx_train_rev_pairs_origional_transliterated_translate[1][0]}, " -> ", {elx_train_rev_pairs_origional_transliterated_translate[1][1]})
    ###Translate from English to grouped transliterated Elamite
    elx_train_rev_pairs_group_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="group", language="Elamite"), normalizeString_cuneiform_transliterate_minimal(elx_transcription_train[l], use_prefix=False)] for l in range(len(elx_en_train))]
    if debug == True:
        print(f"", {elx_train_rev_pairs_group_transliterated_translate[1][0]}, " -> ", {elx_train_rev_pairs_group_transliterated_translate[1][1]})
    ###Merge all data sets
    #train_pairs = akk_train_pairs_cuneiform_translate + akk_train_pairs_simple_transliterated_translate + akk_train_pairs_origional_transliterated_translate + akk_train_pairs_group_transliterated_translate + akk_train_pairs_transliterate_simple + akk_train_pairs_transliterate_origional + akk_train_pairs_transliterate_group + akk_train_pairs_group_simple_transliterate + akk_train_pairs_group_origional_transliterate + akk_train_rev_pairs_cuneiform_translate + akk_train_rev_pairs_simple_transliterated_translate + akk_train_rev_pairs_origional_transliterated_translate + akk_train_rev_pairs_group_transliterated_translate + akk_train_rev_pairs_transliterate_simple + akk_train_rev_pairs_transliterate_group + akk_train_rev_pairs_transliterate_origional + sux_train_pairs_simple_transliterated_translate + sux_train_pairs_origional_transliterated_translate + sux_train_pairs_group_transliterated_translate + sux_train_pairs_group_simple_transliterate + sux_train_rev_pairs_simple_transliterated_translate + sux_train_rev_pairs_origional_transliterated_translate + sux_train_rev_pairs_group_transliterated_translate + elx_train_pairs_simple_transliterated_translate + elx_train_pairs_origional_transliterated_translate + elx_train_pairs_group_transliterated_translate + elx_train_pairs_group_simple_transliterate + elx_train_rev_pairs_simple_transliterated_translate + elx_train_rev_pairs_origional_transliterated_translate + elx_train_rev_pairs_group_transliterated_translate
    #test_pairs = akk_test_pairs_cuneiform_translate + akk_test_pairs_simple_transliterated_translate + akk_test_pairs_origional_transliterated_translate + akk_test_pairs_group_transliterated_translate + akk_test_pairs_transliterate_simple + akk_test_pairs_transliterate_origional + akk_test_pairs_transliterate_group + akk_test_pairs_group_simple_transliterate + akk_test_pairs_group_origional_transliterate + akk_test_rev_pairs_cuneiform_translate + akk_test_rev_pairs_simple_transliterated_translate + akk_test_rev_pairs_origional_transliterated_translate + akk_test_rev_pairs_group_transliterated_translate + akk_test_rev_pairs_transliterate_simple + akk_test_rev_pairs_transliterate_group + akk_test_rev_pairs_transliterate_origional + sux_test_pairs_simple_transliterated_translate + sux_test_pairs_origional_transliterated_translate + sux_test_pairs_group_transliterated_translate + sux_test_pairs_group_simple_transliterate + sux_test_rev_pairs_simple_transliterated_translate + sux_test_rev_pairs_origional_transliterated_translate + sux_test_rev_pairs_group_transliterated_translate
    #val_pairs = akk_val_pairs_cuneiform_translate + akk_val_pairs_simple_transliterated_translate + akk_val_pairs_origional_transliterated_translate + akk_val_pairs_group_transliterated_translate + akk_val_pairs_transliterated_simple + akk_val_pairs_transliterated_origional + akk_val_pairs_transliterated_group + akk_val_pairs_group_simple_transliterate + akk_val_pairs_group_origional_transliterate + akk_val_rev_pairs_cuneiform_translate + akk_val_rev_pairs_simple_transliterated_translate + akk_val_rev_pairs_origional_transliterated_translate + akk_val_rev_pairs_group_transliterated_translate + akk_val_rev_pairs_transliterate_simple + akk_val_rev_pairs_transliterate_group + akk_val_rev_pairs_transliterate_origional + sux_val_pairs_simple_transliterated_translate + sux_val_pairs_origional_transliterated_translate + sux_val_pairs_group_transliterated_translate + sux_val_pairs_group_simple_transliterate + sux_val_rev_pairs_simple_transliterated_translate + sux_val_rev_pairs_origional_transliterated_translate + sux_val_rev_pairs_group_transliterated_translate
    ###All but grouped
    train_pairs = akk_train_pairs_cuneiform_translate + akk_train_pairs_simple_transliterated_translate + akk_train_pairs_origional_transliterated_translate + akk_train_pairs_transliterate_simple + akk_train_pairs_transliterate_origional + akk_train_rev_pairs_cuneiform_translate + akk_train_rev_pairs_simple_transliterated_translate + akk_train_rev_pairs_origional_transliterated_translate + akk_train_rev_pairs_transliterate_simple + akk_train_rev_pairs_transliterate_origional + sux_train_pairs_simple_transliterated_translate + sux_train_pairs_origional_transliterated_translate + sux_train_rev_pairs_simple_transliterated_translate + sux_train_rev_pairs_origional_transliterated_translate + elx_train_pairs_simple_transliterated_translate + elx_train_pairs_origional_transliterated_translate + elx_train_rev_pairs_simple_transliterated_translate + elx_train_rev_pairs_origional_transliterated_translate
    test_pairs = akk_test_pairs_cuneiform_translate + akk_test_pairs_simple_transliterated_translate + akk_test_pairs_origional_transliterated_translate + akk_test_pairs_transliterate_simple + akk_test_pairs_transliterate_origional + akk_test_rev_pairs_cuneiform_translate + akk_test_rev_pairs_simple_transliterated_translate + akk_test_rev_pairs_origional_transliterated_translate + akk_test_rev_pairs_transliterate_simple + akk_test_rev_pairs_transliterate_origional + sux_test_pairs_simple_transliterated_translate + sux_test_pairs_origional_transliterated_translate + sux_test_rev_pairs_simple_transliterated_translate + sux_test_rev_pairs_origional_transliterated_translate
    val_pairs = akk_val_pairs_cuneiform_translate + akk_val_pairs_simple_transliterated_translate + akk_val_pairs_origional_transliterated_translate + akk_val_pairs_transliterated_simple + akk_val_pairs_transliterated_origional + akk_val_rev_pairs_cuneiform_translate + akk_val_rev_pairs_simple_transliterated_translate + akk_val_rev_pairs_origional_transliterated_translate + akk_val_rev_pairs_transliterate_simple + akk_val_rev_pairs_transliterate_origional + sux_val_pairs_simple_transliterated_translate + sux_val_pairs_origional_transliterated_translate + sux_val_rev_pairs_simple_transliterated_translate + sux_val_rev_pairs_origional_transliterated_translate
    pairs = train_pairs + test_pairs + val_pairs
    #print("Examples:")
    #print(f"", {train_pairs_cuneiform_translate[1][0]}, " -> ", {train_pairs_cuneiform_translate[1][1]})
    #print(f"", {train_pairs_transliterate_origional[1][0]}, " -> ", {train_pairs_transliterate_origional[1][1]})
    #print(f"", {train_pairs_transliterate_group[1][0]}, " -> ", {train_pairs_transliterate_group[1][1]})
    #print(f"", {train_pairs_group_transliterated_translate[1][0]}, " -> ", {train_pairs_group_transliterated_translate[1][1]})
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_pairs(train_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("training set trimmed")
    val_pairs = trim_pairs(val_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("val set trimmed")
    test_pairs = trim_pairs(test_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("test set trimmed")
    pairs = train_pairs + test_pairs + val_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0].split()))
    max_length_pair_1 = max(pairs, key=lambda pair: len(pair[1].split()))
    print("Largest number of words in pair[0]:")
    print(f"Word Count: {len(max_length_pair_0[0].split())}, Content: {max_length_pair_0[0]}")
    print("Largest number of words in pair[1]:")
    print(f"Word Count: {len(max_length_pair_1[1].split())}, Content: {max_length_pair_1[1]}")
    mean_length_pair_0 = sum(len(pair[0].split()) for pair in pairs) / len(pairs)
    mean_length_pair_1 = sum(len(pair[1].split()) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in Akkadian: {mean_length_pair_0:.2f}")
    print(f"Mean number of tokens in English: {mean_length_pair_1:.2f}")
    return train_pairs, val_pairs, test_pairs, pairs


def readLangsPreTrain(max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=False):
    print("Reading lines...")
    ##############
    ###Akkadian###
    ##############
    # Read the file and split into lines
    akk_cuneiform_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_train.cu'))
    akk_transcription_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_train.tr'))
    akk_transcription_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_u_train.tr'))
    akk_transcription_train = akk_transcription_train + akk_transcription_u_train
    akk_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'a_train.en'))
    # Split every line into pairs and normalize
    ###Translate from Akkadian cuneiform to English
    akk_train_pairs_cuneiform_translate = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False, task="Translate", language="Akkadian")] for l in range(len(akk_cuneiform_train))]
    if debug == True:
        print(f"", {akk_train_pairs_cuneiform_translate[1][0]})
    ###Translate from simple transliterated Akkadian to English
    akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_train[l], use_prefix=False, task="Translate", type="simple", language="Akkadian")] for l in range(len(akk_transcription_train))]
    if debug == True:
        print(f"", {akk_train_pairs_simple_transliterated_translate[1][0]})
    ###Translate from origional transliterated Akkadian to English
    akk_train_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_train[l], use_prefix=False, task="Translate", type="origional", language="Akkadian")] for l in range(len(akk_transcription_train))]
    if debug == True:
        print(f"", {akk_train_pairs_origional_transliterated_translate[1][0]})
    ###Translate from grouped transliterated Akkadian to English
    #akk_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(akk_transcription_train[l], use_prefix=False, language="Akkadian")] for l in range(len(akk_transcription_train))]
    #if debug == True:
        #print(f"", {akk_train_pairs_group_transliterated_translate[1][0]})
    ###Translate from English to simple transliterated Akkadian
    akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Akkadian")] for l in range(len(akk_en_train))]
    if debug == True:
        print(f"", {akk_train_rev_pairs_simple_transliterated_translate[1][0]})
    ##############
    ###Sumerian###
    ##############
    # Read the file and split into lines
    sux_transcription_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_train.tr'))
    sux_transcription_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_u_train.tr'))
    sux_transcription_train = sux_transcription_train + sux_transcription_u_train
    sux_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 's_train.en'))
    # Split every line into pairs and normalize
    ##Translate from simple transliterated Sumerian to English
    sux_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_train[l], use_prefix=False, task="Translate", type="simple", language="Sumerian")] for l in range(len(sux_transcription_train))]
    if debug == True:
        print(f"", {sux_train_pairs_simple_transliterated_translate[1][0]})
    ###Translate from origional transliterated Sumerian to English
    sux_train_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_train[l], use_prefix=False, task="Translate", type="origional", language="Sumerian")] for l in range(len(sux_transcription_train))]
    if debug == True:
        print(f"", {sux_train_pairs_origional_transliterated_translate[1][0]})
    ###Translate from grouped transliterated Sumerian to English
    #sux_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(sux_transcription_train[l], use_prefix=False, language="Sumerian")] for l in range(len(sux_transcription_train))]
    #if debug == True:
        #print(f"", {sux_train_pairs_group_transliterated_translate[1][0]})
    ###Translate from English to simple transliterated Sumerian
    sux_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Sumerian")] for l in range(len(sux_en_train))]
    if debug == True:
        print(f"", {sux_train_rev_pairs_simple_transliterated_translate[1][0]})
    ###############
    ###Elamite###
    ###############
    # Read the file and split into lines
    elx_transcription_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'e_train.tr'))
    elx_transcription_u_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'e_u_train.tr'))
    elx_transcription_train = elx_transcription_train + elx_transcription_u_train
    elx_en_train = read_and_process_file(os.path.join(user_directory, 'GitHub', 'cuneiform', 'data', 'e_train.en'))
    # Split every line into pairs and normalize
    ##Translate from simple transliterated Elamite to English
    elx_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transcription_train[l], use_prefix=False, task="Translate", type="simple", language="Elamite")] for l in range(len(elx_transcription_train))]
    if debug == True:
        print(f"", {elx_train_pairs_simple_transliterated_translate[1][0]})
    ###Translate from origional transliterated Elamite to English
    elx_train_pairs_origional_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transcription_train[l], use_prefix=False, task="Translate", type="origional", language="Elamite")] for l in range(len(elx_transcription_train))]
    if debug == True:
        print(f"", {elx_train_pairs_origional_transliterated_translate[1][0]})
    ###Translate from grouped transliterated Elamite to English
    #elx_train_pairs_group_transliterated_translate = [[normalizeString_cuneiform_transliterate_minimal(elx_transcription_train[l], use_prefix=False, language="Elamite")] for l in range(len(elx_transcription_train))]
    #if debug == True:
       #print(f"", {elx_train_pairs_group_transliterated_translate[1][0]})
    ###Translate from English to simple transliterated Elamite
    elx_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="simple", language="Elamite")] for l in range(len(elx_en_train))]
    if debug == True:
        print(f"", {elx_train_rev_pairs_simple_transliterated_translate[1][0]})
    ###Translate from English to complex transliterated Elamite
    elx_train_rev_pairs_origional_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=False, task="Translate", target="transliteration", type="origional", language="Elamite")] for l in range(len(elx_en_train))]
    if debug == True:
        print(f"", {elx_train_rev_pairs_origional_transliterated_translate[1][0]})
    ###Merge all data sets
    #train_pairs = akk_train_pairs_cuneiform_translate + akk_train_pairs_simple_transliterated_translate + akk_train_pairs_origional_transliterated_translate + akk_train_pairs_group_transliterated_translate + akk_train_rev_pairs_simple_transliterated_translate +  sux_train_pairs_simple_transliterated_translate + sux_train_pairs_origional_transliterated_translate + sux_train_pairs_group_transliterated_translate + sux_train_rev_pairs_simple_transliterated_translate +  elx_train_pairs_simple_transliterated_translate + elx_train_pairs_origional_transliterated_translate + elx_train_pairs_group_transliterated_translate + elx_train_rev_pairs_origional_transliterated_translate
    train_pairs = akk_train_pairs_cuneiform_translate + akk_train_pairs_simple_transliterated_translate + akk_train_pairs_origional_transliterated_translate +  akk_train_rev_pairs_simple_transliterated_translate +  sux_train_pairs_simple_transliterated_translate + sux_train_pairs_origional_transliterated_translate +  sux_train_rev_pairs_simple_transliterated_translate +  elx_train_pairs_simple_transliterated_translate + elx_train_pairs_origional_transliterated_translate +  elx_train_rev_pairs_origional_transliterated_translate
    pairs = train_pairs
    #print("Examples:")
    #print(f"", {train_pairs_cuneiform_translate[1][0]}, " -> ", {train_pairs_cuneiform_translate[1][1]})
    #print(f"", {train_pairs_transliterate_origional[1][0]}, " -> ", {train_pairs_transliterate_origional[1][1]})
    #print(f"", {train_pairs_transliterate_group[1][0]}, " -> ", {train_pairs_transliterate_group[1][1]})
    #print(f"", {train_pairs_group_transliterated_translate[1][0]}, " -> ", {train_pairs_group_transliterated_translate[1][1]})
    print(f"Total pairs imported: {len(pairs)}")
    train_pairs = trim_singles(train_pairs, max_length_akk, max_length_en, max_length_threshold, min_length_threshold)
    if debug == True:
        print("training set trimmed")
    pairs = train_pairs
    print(f"Total pairs filtered: {len(pairs)}")
    max_length_pair_0 = max(pairs, key=lambda pair: len(pair[0].split()))
    print("Largest number of words in pair[0]:")
    print(f"Word Count: {len(max_length_pair_0[0].split())}, Content: {max_length_pair_0[0]}")
    mean_length_pair_0 = sum(len(pair[0].split()) for pair in pairs) / len(pairs)
    print(f"Mean number of tokens in Akkadian: {mean_length_pair_0:.2f}")
    return train_pairs, pairs

max_length = 60
min_length = 1
cuneiform_pad = max_length
cuneiform_to_english_pad = int(max_length*0.62)
transliterated_to_english_pad = int(max_length*0.71)
save_directory = os.path.join(user_directory, 'GitHub', 'results', 'AKK-SUX-ELX_mT5Large')

if not os.path.exists(os.path.join(save_directory)):
    os.makedirs(os.path.join(save_directory))

if os.path.exists(os.path.join(save_directory, 'data', 'u_tokenized_datasets','dataset_dict.json')):
    print(f"Pretraining data present, skipping")
else:
    # Read your data
    u_train_pairs, u_pairs = readLangsPreTrain(5000, 5000, max_length_threshold=max_length, min_length_threshold= 3)

    # Specify the directory to save the tokenizer

    ###PreTraining
    # Create data dictionaries
    u_train_data_dict = {"akk": [pair[0] for pair in u_train_pairs]}


    # Create datasets
    u_train_dataset = Dataset.from_dict(u_train_data_dict)
    u_train_test_dataset = u_train_dataset.train_test_split(test_size=0.10)
    u_train_dataset = u_train_test_dataset['train']
    u_test_dataset = u_train_test_dataset['test']

    #Create Test Datasets
    #train_dataset = Dataset.from_dict(train_data_dict).select(range(1000))

    u_translations = DatasetDict({"train": u_train_dataset, "test": u_test_dataset})

if os.path.exists(os.path.join(save_directory, 'data', 'tokenized_datasets','dataset_dict.json')):
    print(f"Training data present, skipping")
else:
    ###Training
    ###Generate Data
    train_pairs, val_pairs, test_pairs, pairs = readLangsTrain(5000, 5000, max_length_threshold=max_length, min_length_threshold= min_length)

    # Create data dictionaries
    train_data_dict = {"akk": [pair[0] for pair in train_pairs], "en": [pair[1] for pair in train_pairs]}
    val_data_dict = {"akk": [pair[0] for pair in val_pairs], "en": [pair[1] for pair in val_pairs]}
    test_data_dict = {"akk": [pair[0] for pair in test_pairs], "en": [pair[1] for pair in test_pairs]}

    # Create datasets
    train_dataset = Dataset.from_dict(train_data_dict)
    val_dataset = Dataset.from_dict(val_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)

    translations = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})

# Load the tokenizer and model
if os.path.exists(os.path.join(save_directory, 'base', 'tokenizer.json')):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(save_directory, 'base'), use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_directory, 'base'))
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(finetune_model_id if is_finetune else base_model_id)
    # Extract unique characters for new tokens
    def extract_unique_characters(dataset, column):
        unique_chars = set()
        for example in dataset:
            unique_chars.update(set(example[column]))
        return unique_chars

    # Extract unique tokens from English dataset
    def extract_unique_tokens(dataset, column):
        unique_tokens = set()
        for example in dataset:
            tokens = example[column].split()  # Adjust token splitting if necessary
            unique_tokens.update(tokens)
        return unique_tokens

    ###PreTrain Tokens
    u_train_unique_tokens_akk = extract_unique_tokens(u_translations['train'], 'akk')
    u_test_unique_tokens_akk = extract_unique_tokens(u_translations['test'], 'akk')

    u_unique_tokens_akk = u_train_unique_tokens_akk.union(u_test_unique_tokens_akk)
    print(f'Total pre-training data tokens: {len(u_unique_tokens_akk)}')


    # Get current tokenizer vocabulary
    u_current_vocab = set(tokenizer.get_vocab().keys())
    print(f'Total current tokens: {len(u_current_vocab)}')

    # Find new tokens that are not in the current vocabulary
    u_new_tokens_akk = u_unique_tokens_akk - u_current_vocab

    # Create a single set that includes all unique tokens from both sets, excluding duplicates
    u_unique_new_tokens = u_new_tokens_akk
    print(f'Total added new pre-training tokens: {len(u_unique_new_tokens)}')

    # Add new tokens to the tokenizer
    if u_unique_new_tokens:
        tokenizer.add_tokens(list(u_new_tokens_akk))

    ###Train Tokens
    train_unique_tokens_akk = extract_unique_tokens(translations['train'], 'akk')
    val_unique_tokens_akk = extract_unique_tokens(translations['val'], 'akk')
    test_unique_tokens_akk = extract_unique_tokens(translations['test'], 'akk')

    unique_tokens_akk = train_unique_tokens_akk.union(val_unique_tokens_akk).union(test_unique_tokens_akk)


    train_unique_tokens_en = extract_unique_tokens(translations['train'], 'en')
    val_unique_tokens_en = extract_unique_tokens(translations['val'], 'en')
    test_unique_tokens_en = extract_unique_tokens(translations['test'], 'en')

    unique_tokens_en = train_unique_tokens_en.union(val_unique_tokens_en).union(test_unique_tokens_en)

    unique_data_tokens = unique_tokens_akk.symmetric_difference(unique_tokens_en)
    print(f'Total training data tokens: {len(unique_data_tokens)}')

    # Get current tokenizer vocabulary
    current_vocab = set(tokenizer.get_vocab().keys())
    print(f'Total current tokens: {len(current_vocab)}')

    # Find new tokens that are not in the current vocabulary
    new_tokens_akk = unique_tokens_akk - current_vocab
    new_tokens_en = unique_tokens_en - current_vocab

    # Create a single set that includes all unique tokens from both sets, excluding duplicates
    unique_new_tokens = new_tokens_akk.symmetric_difference(new_tokens_en)

    print(f'Total added new training tokens: {len(unique_new_tokens)}')

    # Add new tokens to the tokenizer
    if unique_new_tokens:
        tokenizer.add_tokens(list(new_tokens_akk))

    final_vocab = set(tokenizer.get_vocab().keys())
    print(f'Total final tokens: {len(final_vocab)}')

    # Resize model token embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))


if os.path.exists(os.path.join(save_directory, 'base', 'tokenizer.json')):
    print(f"Base model and tokenizer present, skipping")
else:
    if not os.path.exists(os.path.join(save_directory, 'base')):
        os.makedirs(os.path.join(save_directory, 'base'))

    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=os.path.join(save_directory, 'base'),  # Same as save_directory
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        # Optionally add train_dataset and eval_dataset if you have them
    )

    # Save the model and tokenizer
    trainer.save_model()

from datasets import load_from_disk

if os.path.exists(os.path.join(save_directory, 'data', 'u_tokenized_datasets','dataset_dict.json')):
    # Load the tokenized datasets from disk
    u_tokenized_datasets = load_from_disk(os.path.join(save_directory, 'data', 'u_tokenized_datasets'))
else:
    # Function to tokenize the examples
    def u_tokenize_function(batch):
        # Compute the lengths of the 'akk' texts
        lengths = [len(text) for text in batch['akk']]
        
        # Tokenize the 'akk' texts without 'return_special_tokens_mask'
        tokenized_output = tokenizer(
            batch['akk'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        # Add labels
        tokenized_output['labels'] = tokenized_output['input_ids'].copy()
        # Add the 'length' to the tokenized output
        tokenized_output['length'] = lengths
        return tokenized_output

    # Tokenize the datasets
    u_tokenized_datasets = u_translations.map(
        u_tokenize_function,
        batched=True
    )
    

    u_tokenized_datasets.save_to_disk(os.path.join(save_directory, 'data', 'u_tokenized_datasets'))

if os.path.exists(os.path.join(save_directory, 'data', 'tokenized_datasets','dataset_dict.json')):
    # Load the tokenized datasets from disk
    tokenized_datasets = load_from_disk(os.path.join(save_directory, 'data', 'tokenized_datasets'))
else:
    # Function to tokenize the examples
    def tokenize_function(batch):
        # Compute the lengths of the 'akk' texts
        lengths = [len(text) for text in batch['akk']]
        # Tokenize the 'akk' texts
        tokenized_inputs = tokenizer(
            batch['akk'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
        # Tokenize the 'en' texts
        with tokenizer.as_target_tokenizer():
            tokenized_targets = tokenizer(
                batch['en'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        # Add labels
        tokenized_inputs["labels"] = tokenized_targets["input_ids"]
        # Add the 'length' to the tokenized inputs
        tokenized_inputs['length'] = lengths
        return tokenized_inputs

    # Tokenize the datasets
    tokenized_datasets = translations.map(
        tokenize_function,
        batched=True
    )

    if not os.path.exists(os.path.join(save_directory, 'data')):
        os.makedirs(os.path.join(save_directory, 'data'))

    tokenized_datasets.save_to_disk(os.path.join(save_directory, 'data', 'tokenized_datasets'))


import random
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
)
from datasets import Dataset
import torch
from typing import Dict, List

class T5DataCollatorForSpanCorruption:
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
    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Tokenize the inputs
        batch_input_ids = [example['input_ids'] for example in examples]
        batch = self._prepare_batch(batch_input_ids)
        return batch
    def _prepare_batch(self, batch_input_ids: List[List[int]]) -> Dict[str, torch.Tensor]:
        batch_input = []
        batch_labels = []
        for input_ids in batch_input_ids:
            processed_input_ids, labels = self._corrupt_input(input_ids)
            batch_input.append(processed_input_ids)
            batch_labels.append(labels)
        # Pad sequences
        batch_input = self.tokenizer.pad({'input_ids': batch_input}, return_tensors='pt')
        batch_labels = self.tokenizer.pad({'input_ids': batch_labels}, return_tensors='pt')
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        batch_labels['input_ids'][batch_labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        # Prepare final batch dict
        batch = {
            'input_ids': batch_input['input_ids'],
            'attention_mask': batch_input['attention_mask'],
            'labels': batch_labels['input_ids'],
        }
        return batch
    def _corrupt_input(self, input_ids: List[int]) -> (List[int], List[int]):
        num_tokens = len(input_ids)
        num_to_mask = max(1, int(round(num_tokens * self.noise_density)))
        # Decide mask lengths
        lengths = []
        while sum(lengths) < num_to_mask:
            lengths.append(
                min(
                    max(int(random.expovariate(1 / self.mean_noise_span_length)), 1),
                    num_tokens - sum(lengths),
                )
            )
        # Select starting positions for the masks
        span_starts = []
        used_positions = set()
        max_attempts = 10  # Maximum number of attempts to find a valid span start
        for length in lengths:
            attempts = 0
            while attempts < max_attempts:
                start = random.randint(0, num_tokens - length)
                positions = set(range(start, start + length))
                if not positions & used_positions:
                    span_starts.append(start)
                    used_positions.update(positions)
                    break
                attempts += 1
            else:
                # If no valid position is found, skip this span
                continue
    # Rest of the method remains the same
        # Sort spans by starting position
        spans = sorted(zip(span_starts, lengths))
        # Create the corrupted input and labels
        corrupted_input_ids = []
        labels = []
        prev_end = 0
        sentinel_token_id = self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        current_sentinel = 0
        for start, length in spans:
            # Add non-masked tokens to corrupted input
            corrupted_input_ids.extend(input_ids[prev_end:start])
            # Add sentinel token to corrupted input
            corrupted_input_ids.append(self.tokenizer.convert_tokens_to_ids(f'<extra_id_{current_sentinel}>'))
            # Add sentinel token to labels
            labels.append(self.tokenizer.convert_tokens_to_ids(f'<extra_id_{current_sentinel}>'))
            # Add masked tokens to labels
            labels.extend(input_ids[start:start + length])
            # Update positions
            prev_end = start + length
            current_sentinel += 1
        # Add the remaining non-masked tokens
        corrupted_input_ids.extend(input_ids[prev_end:])
        if prev_end < num_tokens:
            corrupted_input_ids.append(self.tokenizer.eos_token_id)
        # Add EOS token to labels
        labels.append(self.tokenizer.eos_token_id)
        return corrupted_input_ids, labels

###Pre-train data collator
u_data_collator = T5DataCollatorForSpanCorruption(
    tokenizer=tokenizer,
    noise_density=0.15,
    mean_noise_span_length=3.0,
    input_length=max_length,
    target_length=max_length,
)

###Checkpoint
def find_latest_checkpoint(checkpoint_dir):
    # List all files in the checkpoint directory
    checkpoints = os.listdir(checkpoint_dir)
    # Filter and sort the files based on modification time in descending order
    checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
    # Return the newest checkpoint file
    if checkpoints:
        return os.path.join(checkpoint_dir, checkpoints[0])
    return None

####################
###First Pretrain###
####################
if os.path.exists(os.path.join(save_directory, 'pretrain_1','tokenizer_config.json')):
    # Load the tokenized datasets from disk
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_directory, 'pretrain_1'))
else:
    if not os.path.exists(os.path.join(save_directory, 'pretrain_1')):
        os.makedirs(os.path.join(save_directory, 'pretrain_1'))
    
    last_checkpoint = find_latest_checkpoint(os.path.join(save_directory, 'pretrain_1'))
    if last_checkpoint:
        print(f"Loading from {last_checkpoint}")
        # Load your model from the latest checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    else:
        print("No checkpoint found, starting from scratch")
        
    # PreTraining arguments
    pre_training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_directory, 'pretrain_1'),
        learning_rate=5e-4,
        lr_scheduler_type="linear",
        warmup_steps=2000,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy ="epoch",
        #gradient_accumulation_steps=2,
        #gradient_checkpointing=True,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=200,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        logging_steps=500,
        #save_steps=10000,
        #optim="adamw_8bit",
        #max_grad_norm=0.75,
        #adam_epsilon=1e-8,
        #adam_beta1=0.9,
        #adam_beta2=0.999,
        #deepspeed=os.path.join(user_directory, 'GitHub', 'cuneiform', 'ds_config_bf16.json'),
    )

    # Initialize the PreTrainer
    pre_trainer = Seq2SeqTrainer(
        model=model,
        args=pre_training_args,
        train_dataset=u_tokenized_datasets["train"],
        eval_dataset=u_tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=u_data_collator,
        callbacks=[EarlyStoppingCallback(5, 0.0)],
        #optimizers=(adam_bnb_optim, None),
        #optimizers=(optimizer, lr_scheduler),
    )

    pre_trainer.train()


#################
###First Train###
#################
# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

if os.path.exists(os.path.join(save_directory, 'train_1','tokenizer_config.json')):
    # Load the tokenized datasets from disk
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_directory, 'train_1'))
else:
    if not os.path.exists(os.path.join(save_directory, 'train_1')):
        os.makedirs(os.path.join(save_directory, 'train_1'))
        
    last_checkpoint = find_latest_checkpoint(os.path.join(save_directory, 'train_1'))
    if last_checkpoint:
        print(f"Loading from {last_checkpoint}")
        # Load your model from the latest checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    else:
        print("No checkpoint found, starting from scratch")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_directory, 'train_1'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        lr_scheduler_type="linear",
        warmup_steps=2000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        #gradient_accumulation_steps=2,
        #gradient_checkpointing=True,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=200,
        fp16=False,
        bf16=True,
        #save_steps=25000,
        #eval_steps=25000,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        logging_steps=500,
        #optim="adamw_hf",
        #max_grad_norm=0.75,
        #adam_epsilon=1e-8,
        #adam_beta1=0.9,
        #adam_beta2=0.999,
        #deepspeed=os.path.join(user_directory, 'GitHub', 'cuneiform', 'ds_config_bf16.json'),
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(5, 0.0)],
        #optimizers=(adam_bnb_optim, None),
        #optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()

#####################
###Second Pretrain###
#####################
from torch.optim.lr_scheduler import LambdaLR

def get_inverse_linear_scheduler(optimizer, num_warmup_steps=10000):
    def lr_lambda(current_step):
        k = num_warmup_steps
        n = current_step + 1  # Adding 1 because step count starts from 0
        if n <= k:
            return 1.0  # Learning rate multiplier is 1 during warm-up
        else:
            return float(k) / float(n)  # Learning rate decays as k/n
    return LambdaLR(optimizer, lr_lambda)

from torch.optim import AdamW

# Assuming k = 10000
k = 10000
initial_lr = 1.0 / k  # Set initial learning rate to 1/k

optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)

scheduler = get_inverse_linear_scheduler(optimizer, num_warmup_steps=k)

if os.path.exists(os.path.join(save_directory, 'pretrain_2','tokenizer_config.json')):
    # Load the tokenized datasets from disk
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_directory, 'pretrain_2'))
else:
    if not os.path.exists(os.path.join(save_directory, 'pretrain_2')):
        os.makedirs(os.path.join(save_directory, 'pretrain_2'))
    
    last_checkpoint = find_latest_checkpoint(os.path.join(save_directory, 'pretrain_2'))
    if last_checkpoint:
        print(f"Loading from {last_checkpoint}")
        # Load your model from the latest checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    else:
        print("No checkpoint found, starting from scratch")

    # PreTraining arguments
    pre_training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_directory, 'pretrain_2'),
        #learning_rate=5e-4,
        lr_scheduler_type="linear",
        #warmup_steps=2000,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy ="epoch",
        #gradient_accumulation_steps=2,
        #gradient_checkpointing=True,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=200,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        fp16=False,
        bf16=True,
        logging_steps=500,
        #save_steps=10000,
        #optim="adamw_8bit",
        #max_grad_norm=0.75,
        #adam_epsilon=1e-8,
        #adam_beta1=0.9,
        #adam_beta2=0.999,
        #deepspeed=os.path.join(user_directory, 'GitHub', 'cuneiform', 'ds_config_bf16.json'),
    )

    # Initialize the PreTrainer
    pre_trainer = Seq2SeqTrainer(
        model=model,
        args=pre_training_args,
        train_dataset=u_tokenized_datasets["train"],
        eval_dataset=u_tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=u_data_collator,
        callbacks=[EarlyStoppingCallback(5, 0.0)],
        optimizers=(optimizer, scheduler),
        )

    pre_trainer.train()

    ####Push it
    pre_trainer.push_to_hub()

##################
###Second Train###
##################
if os.path.exists(os.path.join(save_directory, 'train_2','tokenizer_config.json')):
    # Load the tokenized datasets from disk
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(save_directory, 'train_2'))
else:
    if not os.path.exists(os.path.join(save_directory, 'train_2')):
        os.makedirs(os.path.join(save_directory, 'train_2'))

    last_checkpoint = find_latest_checkpoint(os.path.join(save_directory, 'train_2'))
    if last_checkpoint:
        print(f"Loading from {last_checkpoint}")
        # Load your model from the latest checkpoint
        model = AutoModelForSeq2SeqLM.from_pretrained(last_checkpoint)
    else:
        print("No checkpoint found, starting from scratch")

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_directory, 'train_2'),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        #learning_rate=5e-6,
        #lr_scheduler_type="linear",
        #warmup_steps=2000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        #gradient_accumulation_steps=2,
        #gradient_checkpointing=True,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=200,
        fp16=False,
        bf16=True,
        #save_steps=25000,
        #eval_steps=25000,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        greater_is_better=False,
        logging_steps=500,
        #optim="adamw_hf",
        #max_grad_norm=0.75,
        #adam_epsilon=1e-8,
        #adam_beta1=0.9,
        #adam_beta2=0.999,
        #deepspeed=os.path.join(user_directory, 'GitHub', 'cuneiform', 'ds_config_bf16.json'),
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(5, 0.0)],
        optimizers=(optimizer, scheduler),
    )

    trainer.train()

    ####Push it
    trainer.push_to_hub()

print("All done")
