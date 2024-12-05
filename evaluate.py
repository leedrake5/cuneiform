prediction_length = 60

import sys, os, datetime, pwd
user_directory = pwd.getpwuid(os.getuid()).pw_dir
import json
import torch
import random
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import DatasetDict, Dataset
import accelerate
import sacrebleu
import pandas as pd
from datasets import DatasetDict, Dataset


import re
import requests
import unicodedata

import os

import gc

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

base_model_id = "google-t5/t5-base"
finetune_model_id = None
# finetune_model_id = get_finetune_model_id("t5-base-p-akksux-en-20220722-173018")

#model_max_length = 512
#batch_size = 8 if os.path.basename(base_model_id).startswith("t5-base") else 128

batch_size=1

num_train_epochs = 10

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


def readLangs(max_length_akk=5000, max_length_en=5000, max_length_threshold=100, min_length_threshold=50, debug=False):
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
    akk_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_train[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_test[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(akk_transcription_val[l], use_prefix=True, task="Translate", type="simple", language="Akkadian"), normalizeString_en(akk_en_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
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
    akk_train_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_train[l], use_prefix=False)] for l in range(len(akk_cuneiform_train))]
    akk_test_pairs_transliterate_simple = [[normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_test[l], use_prefix=False)] for l in range(len(akk_cuneiform_test))]
    akk_val_pairs_transliterated_simple = [[normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=True, task="Transliterate", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_val[l], use_prefix=False)] for l in range(len(akk_cuneiform_val))]
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
    akk_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_train[l], use_prefix=False, type="simple")] for l in range(len(akk_en_train))]
    akk_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_test[l], use_prefix=False, type="simple")] for l in range(len(akk_en_test))]
    akk_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(akk_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Akkadian"), normalizeString_cuneiform_transliterate(akk_transcription_val[l], use_prefix=False, type="simple")] for l in range(len(akk_en_val))]
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
    akk_train_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_train[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_train[l], use_prefix=False)] for l in range(len(akk_transcription_train))]
    akk_test_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_test[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_test[l], use_prefix=False)] for l in range(len(akk_transcription_test))]
    akk_val_rev_pairs_transliterate_simple = [[normalizeString_cuneiform_rev_transliterate(akk_transcription_val[l], use_prefix=True, type="simple", language="Akkadian"), normalizeString_cuneiform(akk_cuneiform_val[l], use_prefix=False)] for l in range(len(akk_transcription_val))]
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
    sux_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_train[l], use_prefix=True, task="Translate", type="simple", language="Sumerian"), normalizeString_en(sux_en_train[l], use_prefix=False)] for l in range(len(sux_transcription_train))]
    sux_test_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_test[l], use_prefix=True, task="Translate", type="simple", language="Sumerian"), normalizeString_en(sux_en_test[l], use_prefix=False)] for l in range(len(sux_transcription_test))]
    sux_val_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(sux_transcription_val[l], use_prefix=True, task="Translate", type="simple", language="Sumerian"), normalizeString_en(sux_en_val[l], use_prefix=False)] for l in range(len(sux_transcription_val))]
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
    sux_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Sumerian"), normalizeString_cuneiform_transliterate(sux_transcription_train[l], use_prefix=False, type="simple")] for l in range(len(sux_en_train))]
    sux_test_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_test[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Sumerian"), normalizeString_cuneiform_transliterate(sux_transcription_test[l], use_prefix=False, type="simple")] for l in range(len(sux_en_test))]
    sux_val_rev_pairs_simple_transliterated_translate = [[normalizeString_en(sux_en_val[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Sumerian"), normalizeString_cuneiform_transliterate(sux_transcription_val[l], use_prefix=False, type="simple")] for l in range(len(sux_en_val))]
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
    elx_train_pairs_simple_transliterated_translate = [[normalizeString_cuneiform_transliterate_translate(elx_transcription_train[l], use_prefix=True, task="Translate", type="simple", language="Elamite"), normalizeString_en(elx_en_train[l], use_prefix=False)] for l in range(len(elx_transcription_train))]
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
    elx_train_rev_pairs_simple_transliterated_translate = [[normalizeString_en(elx_en_train[l], use_prefix=True, task="Translate", target="transliteration", type="simple", language="Elamite"), normalizeString_cuneiform_transliterate(elx_transcription_train[l], use_prefix=False, type="simple")] for l in range(len(elx_en_train))]
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
    train_pairs = akk_train_pairs_cuneiform_translate + akk_train_pairs_simple_transliterated_translate + akk_train_pairs_origional_transliterated_translate + akk_train_pairs_group_transliterated_translate + akk_train_pairs_transliterate_simple + akk_train_pairs_transliterate_origional + akk_train_pairs_transliterate_group + akk_train_pairs_group_simple_transliterate + akk_train_pairs_group_origional_transliterate + akk_train_rev_pairs_cuneiform_translate + akk_train_rev_pairs_simple_transliterated_translate + akk_train_rev_pairs_origional_transliterated_translate + akk_train_rev_pairs_group_transliterated_translate + akk_train_rev_pairs_transliterate_simple + akk_train_rev_pairs_transliterate_group + akk_train_rev_pairs_transliterate_origional + sux_train_pairs_simple_transliterated_translate + sux_train_pairs_origional_transliterated_translate + sux_train_pairs_group_transliterated_translate + sux_train_pairs_group_simple_transliterate + sux_train_rev_pairs_simple_transliterated_translate + sux_train_rev_pairs_origional_transliterated_translate + sux_train_rev_pairs_group_transliterated_translate + elx_train_pairs_simple_transliterated_translate + elx_train_pairs_origional_transliterated_translate + elx_train_pairs_group_transliterated_translate + elx_train_pairs_group_simple_transliterate + elx_train_rev_pairs_simple_transliterated_translate + elx_train_rev_pairs_origional_transliterated_translate + elx_train_rev_pairs_group_transliterated_translate
    test_pairs = akk_test_pairs_cuneiform_translate + akk_test_pairs_simple_transliterated_translate + akk_test_pairs_origional_transliterated_translate + akk_test_pairs_group_transliterated_translate + akk_test_pairs_transliterate_simple + akk_test_pairs_transliterate_origional + akk_test_pairs_transliterate_group + akk_test_pairs_group_simple_transliterate + akk_test_pairs_group_origional_transliterate + akk_test_rev_pairs_cuneiform_translate + akk_test_rev_pairs_simple_transliterated_translate + akk_test_rev_pairs_origional_transliterated_translate + akk_test_rev_pairs_group_transliterated_translate + akk_test_rev_pairs_transliterate_simple + akk_test_rev_pairs_transliterate_group + akk_test_rev_pairs_transliterate_origional + sux_test_pairs_simple_transliterated_translate + sux_test_pairs_origional_transliterated_translate + sux_test_pairs_group_transliterated_translate + sux_test_pairs_group_simple_transliterate + sux_test_rev_pairs_simple_transliterated_translate + sux_test_rev_pairs_origional_transliterated_translate + sux_test_rev_pairs_group_transliterated_translate
    val_pairs = akk_val_pairs_cuneiform_translate + akk_val_pairs_simple_transliterated_translate + akk_val_pairs_origional_transliterated_translate + akk_val_pairs_group_transliterated_translate + akk_val_pairs_transliterated_simple + akk_val_pairs_transliterated_origional + akk_val_pairs_transliterated_group + akk_val_pairs_group_simple_transliterate + akk_val_pairs_group_origional_transliterate + akk_val_rev_pairs_cuneiform_translate + akk_val_rev_pairs_simple_transliterated_translate + akk_val_rev_pairs_origional_transliterated_translate + akk_val_rev_pairs_group_transliterated_translate + akk_val_rev_pairs_transliterate_simple + akk_val_rev_pairs_transliterate_group + akk_val_rev_pairs_transliterate_origional + sux_val_pairs_simple_transliterated_translate + sux_val_pairs_origional_transliterated_translate + sux_val_pairs_group_transliterated_translate + sux_val_pairs_group_simple_transliterate + sux_val_rev_pairs_simple_transliterated_translate + sux_val_rev_pairs_origional_transliterated_translate + sux_val_rev_pairs_group_transliterated_translate
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
    return train_pairs, val_pairs, test_pairs, pairs, akk_train_pairs_cuneiform_translate , akk_train_pairs_simple_transliterated_translate , akk_train_pairs_origional_transliterated_translate , akk_train_pairs_group_transliterated_translate , akk_train_pairs_transliterate_simple , akk_train_pairs_transliterate_origional , akk_train_pairs_transliterate_group , akk_train_pairs_group_simple_transliterate , akk_train_pairs_group_origional_transliterate , akk_train_rev_pairs_cuneiform_translate , akk_train_rev_pairs_simple_transliterated_translate , akk_train_rev_pairs_origional_transliterated_translate , akk_train_rev_pairs_group_transliterated_translate , akk_train_rev_pairs_transliterate_simple , akk_train_rev_pairs_transliterate_group , akk_train_rev_pairs_transliterate_origional , sux_train_pairs_simple_transliterated_translate , sux_train_pairs_origional_transliterated_translate , sux_train_pairs_group_transliterated_translate , sux_train_pairs_group_simple_transliterate , sux_train_rev_pairs_simple_transliterated_translate , sux_train_rev_pairs_origional_transliterated_translate , sux_train_rev_pairs_group_transliterated_translate , elx_train_pairs_simple_transliterated_translate , elx_train_pairs_origional_transliterated_translate , elx_train_pairs_group_transliterated_translate , elx_train_pairs_group_simple_transliterate , elx_train_rev_pairs_simple_transliterated_translate , elx_train_rev_pairs_origional_transliterated_translate , elx_train_rev_pairs_group_transliterated_translate , akk_test_pairs_cuneiform_translate , akk_test_pairs_simple_transliterated_translate , akk_test_pairs_origional_transliterated_translate , akk_test_pairs_group_transliterated_translate , akk_test_pairs_transliterate_simple , akk_test_pairs_transliterate_origional , akk_test_pairs_transliterate_group , akk_test_pairs_group_simple_transliterate , akk_test_pairs_group_origional_transliterate , akk_test_rev_pairs_cuneiform_translate , akk_test_rev_pairs_simple_transliterated_translate , akk_test_rev_pairs_origional_transliterated_translate , akk_test_rev_pairs_group_transliterated_translate , akk_test_rev_pairs_transliterate_simple , akk_test_rev_pairs_transliterate_group , akk_test_rev_pairs_transliterate_origional , sux_test_pairs_simple_transliterated_translate , sux_test_pairs_origional_transliterated_translate , sux_test_pairs_group_transliterated_translate , sux_test_pairs_group_simple_transliterate , sux_test_rev_pairs_simple_transliterated_translate , sux_test_rev_pairs_origional_transliterated_translate , sux_test_rev_pairs_group_transliterated_translate , akk_val_pairs_cuneiform_translate , akk_val_pairs_simple_transliterated_translate , akk_val_pairs_origional_transliterated_translate , akk_val_pairs_group_transliterated_translate , akk_val_pairs_transliterated_simple , akk_val_pairs_transliterated_origional , akk_val_pairs_transliterated_group , akk_val_pairs_group_simple_transliterate , akk_val_pairs_group_origional_transliterate , akk_val_rev_pairs_cuneiform_translate , akk_val_rev_pairs_simple_transliterated_translate , akk_val_rev_pairs_origional_transliterated_translate , akk_val_rev_pairs_group_transliterated_translate , akk_val_rev_pairs_transliterate_simple , akk_val_rev_pairs_transliterate_group , akk_val_rev_pairs_transliterate_origional , sux_val_pairs_simple_transliterated_translate , sux_val_pairs_origional_transliterated_translate , sux_val_pairs_group_transliterated_translate , sux_val_pairs_group_simple_transliterate , sux_val_rev_pairs_simple_transliterated_translate , sux_val_rev_pairs_origional_transliterated_translate , sux_val_rev_pairs_group_transliterated_translate

# Read your data
max_length = 60
min_length = 1
cuneiform_pad = max_length
cuneiform_to_english_pad = int(max_length*0.62)
transliterated_to_english_pad = int(max_length*0.71)
train_pairs, val_pairs, test_pairs, pairs, akk_train_pairs_cuneiform_translate , akk_train_pairs_simple_transliterated_translate , akk_train_pairs_origional_transliterated_translate , akk_train_pairs_group_transliterated_translate , akk_train_pairs_transliterate_simple , akk_train_pairs_transliterate_origional , akk_train_pairs_transliterate_group , akk_train_pairs_group_simple_transliterate , akk_train_pairs_group_origional_transliterate , akk_train_rev_pairs_cuneiform_translate , akk_train_rev_pairs_simple_transliterated_translate , akk_train_rev_pairs_origional_transliterated_translate , akk_train_rev_pairs_group_transliterated_translate , akk_train_rev_pairs_transliterate_simple , akk_train_rev_pairs_transliterate_group , akk_train_rev_pairs_transliterate_origional , sux_train_pairs_simple_transliterated_translate , sux_train_pairs_origional_transliterated_translate , sux_train_pairs_group_transliterated_translate , sux_train_pairs_group_simple_transliterate , sux_train_rev_pairs_simple_transliterated_translate , sux_train_rev_pairs_origional_transliterated_translate , sux_train_rev_pairs_group_transliterated_translate , elx_train_pairs_simple_transliterated_translate , elx_train_pairs_origional_transliterated_translate , elx_train_pairs_group_transliterated_translate , elx_train_pairs_group_simple_transliterate , elx_train_rev_pairs_simple_transliterated_translate , elx_train_rev_pairs_origional_transliterated_translate , elx_train_rev_pairs_group_transliterated_translate , akk_test_pairs_cuneiform_translate , akk_test_pairs_simple_transliterated_translate , akk_test_pairs_origional_transliterated_translate , akk_test_pairs_group_transliterated_translate , akk_test_pairs_transliterate_simple , akk_test_pairs_transliterate_origional , akk_test_pairs_transliterate_group , akk_test_pairs_group_simple_transliterate , akk_test_pairs_group_origional_transliterate , akk_test_rev_pairs_cuneiform_translate , akk_test_rev_pairs_simple_transliterated_translate , akk_test_rev_pairs_origional_transliterated_translate , akk_test_rev_pairs_group_transliterated_translate , akk_test_rev_pairs_transliterate_simple , akk_test_rev_pairs_transliterate_group , akk_test_rev_pairs_transliterate_origional , sux_test_pairs_simple_transliterated_translate , sux_test_pairs_origional_transliterated_translate , sux_test_pairs_group_transliterated_translate , sux_test_pairs_group_simple_transliterate , sux_test_rev_pairs_simple_transliterated_translate , sux_test_rev_pairs_origional_transliterated_translate , sux_test_rev_pairs_group_transliterated_translate , akk_val_pairs_cuneiform_translate , akk_val_pairs_simple_transliterated_translate , akk_val_pairs_origional_transliterated_translate , akk_val_pairs_group_transliterated_translate , akk_val_pairs_transliterated_simple , akk_val_pairs_transliterated_origional , akk_val_pairs_transliterated_group , akk_val_pairs_group_simple_transliterate , akk_val_pairs_group_origional_transliterate , akk_val_rev_pairs_cuneiform_translate , akk_val_rev_pairs_simple_transliterated_translate , akk_val_rev_pairs_origional_transliterated_translate , akk_val_rev_pairs_group_transliterated_translate , akk_val_rev_pairs_transliterate_simple , akk_val_rev_pairs_transliterate_group , akk_val_rev_pairs_transliterate_origional , sux_val_pairs_simple_transliterated_translate , sux_val_pairs_origional_transliterated_translate , sux_val_pairs_group_transliterated_translate , sux_val_pairs_group_simple_transliterate , sux_val_rev_pairs_simple_transliterated_translate , sux_val_rev_pairs_origional_transliterated_translate , sux_val_rev_pairs_group_transliterated_translate = readLangs(5000, 5000, max_length_threshold=max_length, min_length_threshold= min_length)

#akk_u_pairs_simple_transliterated_translate, akk_u_pairs_origional_transliterated_translate, akk_u_pairs_group_transliterated_translate, sux_u_pairs_simple_transliterated_translate, sux_u_pairs_origional_transliterated_translate, sux_u_pairs_group_transliterated_translate, elx_u_pairs_simple_transliterated_translate, elx_u_pairs_origional_transliterated_translate, elx_u_pairs_group_transliterated_translate = readLangsUnknown(5000, 5000, max_length_threshold=max_length, min_length_threshold= min_length)



# Create data dictionaries
train_data_dict = {"input": [pair[0] for pair in train_pairs], "target": [pair[1] for pair in train_pairs]}
val_data_dict = {"input": [pair[0] for pair in val_pairs], "target": [pair[1] for pair in val_pairs]}
test_data_dict = {"input": [pair[0] for pair in test_pairs], "target": [pair[1] for pair in test_pairs]}

# Create datasets
train_dataset = Dataset.from_dict(train_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)

#Testing Datasets
#train_dataset = Dataset.from_dict(train_data_dict).select(range(1000))
#val_dataset = Dataset.from_dict(val_data_dict).select(range(1000))
#test_dataset = Dataset.from_dict(test_data_dict).select(range(1000))



translations = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})


model_path = os.path.join(user_directory, 'GitHub', 'results', 't5-base-p-l-akk-en-20241110-154709', 'checkpoint-864360')
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)


# Create data dictionaries
train_data_dict = {"input": [pair[0] for pair in train_pairs], "target": [pair[1] for pair in train_pairs]}
val_data_dict = {"input": [pair[0] for pair in val_pairs], "target": [pair[1] for pair in val_pairs]}
test_data_dict = {"input": [pair[0] for pair in test_pairs], "target": [pair[1] for pair in test_pairs]}

# Create datasets
train_dataset = Dataset.from_dict(train_data_dict)
val_dataset = Dataset.from_dict(val_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)


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

train_unique_chars_akk = extract_unique_tokens(translations['train'], 'input')
val_unique_chars_akk = extract_unique_tokens(translations['val'], 'input')
test_unique_chars_akk = extract_unique_tokens(translations['test'], 'input')

unique_chars_akk = train_unique_chars_akk.union(val_unique_chars_akk).union(test_unique_chars_akk)


train_unique_tokens_en = extract_unique_tokens(translations['train'], 'target')
val_unique_tokens_en = extract_unique_tokens(translations['val'], 'target')
test_unique_tokens_en = extract_unique_tokens(translations['test'], 'target')

unique_tokens_en = train_unique_tokens_en.union(val_unique_tokens_en).union(test_unique_tokens_en)

# Get current tokenizer vocabulary
current_vocab = set(tokenizer.get_vocab().keys())

# Find new tokens that are not in the current vocabulary
new_tokens_akk = unique_chars_akk - current_vocab
new_tokens_en = unique_tokens_en - current_vocab

# Create a single set that includes all unique tokens from both sets, excluding duplicates
unique_new_tokens = new_tokens_akk.symmetric_difference(new_tokens_en)

# Add new tokens to the tokenizer
#if unique_new_tokens:
#    tokenizer.add_tokens(list(new_tokens_akk))



# Function to tokenize the examples
def tokenize_function(example):
    tokenized_inputs = tokenizer(example['akk'], padding="max_length", truncation=True, max_length=max_length)
    with tokenizer.as_target_tokenizer():
        tokenized_targets = tokenizer(example['en'], padding="max_length", truncation=True, max_length=max_length)
    tokenized_inputs["labels"] = tokenized_targets["input_ids"]
    return tokenized_inputs

# Define the data collator
import numpy as np
from datasets import load_metric
metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Ensure all token IDs are valid
    valid_token_ids = set(tokenizer.get_vocab().values())
    preds = np.where(np.isin(preds, list(valid_token_ids)), preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    collected = gc.collect()
    return result

# Define the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def langEval(model, tokenizer, batch_size, train_pairs, test_pairs, val_pairs, set):
    # Create data dictionaries
    train_data_dict = {"akk": [pair[0] for pair in train_pairs], "en": [pair[1] for pair in train_pairs]}
    val_data_dict = {"akk": [pair[0] for pair in val_pairs], "en": [pair[1] for pair in val_pairs]}
    test_data_dict = {"akk": [pair[0] for pair in test_pairs], "en": [pair[1] for pair in test_pairs]}
    # Create datasets
    train_dataset = Dataset.from_dict(train_data_dict)
    val_dataset = Dataset.from_dict(val_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)
    translations = DatasetDict({"train": train_dataset, "val": val_dataset, "test": test_dataset})
    # Tokenize the datasets
    tokenized_datasets = DatasetDict({
        "train": translations["train"].map(tokenize_function, batched=True),
        "val": translations["val"].map(tokenize_function, batched=True),
        "test": translations["test"].map(tokenize_function, batched=True),
    })
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        learning_rate=2*2e-5,
        per_device_train_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        fp16=False,
        save_steps=500,
        eval_strategy="no",
        #use_mps_device=True,
    )
    if set == "all":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        train_preds = trainer.predict(tokenized_datasets["train"], max_length=prediction_length)
        print(train_preds[2])
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        test_preds = trainer.predict(tokenized_datasets["test"], max_length=prediction_length)
        print(test_preds[2])
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        val_preds = trainer.predict(tokenized_datasets["val"], max_length=prediction_length)
        print(val_preds[2])
        return train_preds, test_preds, val_preds
    elif set == "train":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        train_preds = trainer.predict(tokenized_datasets["train"], max_length=prediction_length)
        print(train_preds[2])
        return train_preds
    elif set == "test":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        test_preds = trainer.predict(tokenized_datasets["test"], max_length=prediction_length)
        print(test_preds[2])
        return test_preds
    elif set == "val":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        val_preds = trainer.predict(tokenized_datasets["val"], max_length=prediction_length)
        print(val_preds[2])
        return val_preds
    elif set =="just_vals":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        test_preds = trainer.predict(tokenized_datasets["test"], max_length=prediction_length)
        print(test_preds[2])
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["val"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        val_preds = trainer.predict(tokenized_datasets["val"], max_length=prediction_length)
        print(val_preds[2])
        return test_preds, val_preds
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % (collected))

def lang_compare(model, tokenizer, batch_size, akk_pairs, sux_pairs, elx_pairs, set):
    # Create data dictionaries
    akk_data_dict = {"akk": [pair[0] for pair in akk_pairs], "en": [pair[1] for pair in akk_pairs]}
    sux_data_dict = {"akk": [pair[0] for pair in sux_pairs], "en": [pair[1] for pair in sux_pairs]}
    elx_data_dict = {"akk": [pair[0] for pair in elx_pairs], "en": [pair[1] for pair in elx_pairs]}
    # Create datasets
    akk_dataset = Dataset.from_dict(akk_data_dict)
    sux_dataset = Dataset.from_dict(sux_data_dict)
    elx_dataset = Dataset.from_dict(elx_data_dict)
    translations = DatasetDict({"akk": akk_dataset, "sux": sux_dataset, "elx": elx_dataset})
    # Tokenize the datasets
    tokenized_datasets = DatasetDict({
        "akk": translations["akk"].map(tokenize_function, batched=True),
        "sux": translations["sux"].map(tokenize_function, batched=True),
        "elx": translations["elx"].map(tokenize_function, batched=True),
    })
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        evaluation_strategy="no",
        learning_rate=2*2e-5,
        per_device_train_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        fp16=False,
        save_steps=500,
        #use_mps_device=True,
    )
    if set == "akk-sux-elx":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["akk"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        akk_preds = trainer.predict(tokenized_datasets["akk"], max_length=prediction_length)
        print(akk_preds[2])
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["sux"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        sux_preds = trainer.predict(tokenized_datasets["sux"], max_length=prediction_length)
        print(sux_preds[2])
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["elx"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        elx_preds = trainer.predict(tokenized_datasets["elx"], max_length=prediction_length)
        print(elx_preds[2])
        return akk_preds, sux_preds, elx_preds
    elif set == "akk-sux":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["akk"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        akk_preds = trainer.predict(tokenized_datasets["akk"], max_length=prediction_length)
        print(akk_preds[2])
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["sux"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        sux_preds = trainer.predict(tokenized_datasets["sux"], max_length=prediction_length)
        print(sux_preds[2])
        return akk_preds, sux_preds
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % (collected))

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

import os
os.environ["WANDB_DISABLED"] = "true"


from transformers.pipelines.text2text_generation import Text2TextGenerationPipeline
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import string
import re
import pandas as pd

def normalize_spaces(text):
    # Replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', text).strip()

import nltk
from nltk.corpus import words

# Download the list of English words if not already downloaded
nltk.download('words', quiet=True)

def correct_split_words(text, tokenizer):
    # Get a set of English words for quick lookup
    english_words = set(words.words())
    # Optionally add custom words relevant to your domain
    custom_words = {'namri', 'assur', 'karkar', 'kassie'}
    english_words.update(custom_words)
    tokenizer_words = set(tokenizer.get_vocab())
    english_words.update(tokenizer_words)
    # Tokenize the text into words
    tokens = text.split()
    i = 0
    corrected_tokens = []
    while i < len(tokens):
        found = False
        # Determine the maximum number of tokens to attempt to merge
        max_n = min(5, len(tokens) - i)
        # Try to merge tokens starting with the longest possible sequence
        for n in range(max_n, 1, -1):
            merged_token = ''.join(tokens[i:i+n])
            if merged_token.lower() in english_words:
                # Found a valid word by merging tokens
                corrected_tokens.append(merged_token)
                i += n  # Skip the merged tokens
                found = True
                break
        if not found:
            # If no valid merge found, add the current token
            corrected_tokens.append(tokens[i])
            i += 1
    corrected_text = ' '.join(corrected_tokens)
    return corrected_text

def remove_prompt(text):
    # Split the text at the first occurrence of ':'
    parts = text.split(':', 1)
    if len(parts) > 1:
        # Return the part after the colon, stripping any leading/trailing whitespace
        return parts[1].strip()
    else:
        # If no colon is found, return the original text
        return text.strip()

def translate_cuneiform_set(pairs, set='val', name="translated_cuneiform", check_prompt="Translate English to complex Akkadian transliteration: "):
    if not os.path.exists(os.path.join(model_path, 'evaluation')):
        os.makedirs(os.path.join(model_path, 'evaluation'))
    filepath = os.path.join(model_path, 'evaluation', f'{name}_{set}_set.csv')
    if os.path.exists(filepath):
        result_df = pd.read_csv(filepath)
        print("     Results found, skipping")
    else:
        set_data_dict = {"akk": [pair[0] for pair in pairs], "en": [pair[1] for pair in pairs]}
        # Create datasets
        set_dataset = Dataset.from_dict(set_data_dict)
        translations = DatasetDict({"set": set_dataset})
        results = []
        # Initialize the pipeline outside the loop
        pipe = Text2TextGenerationPipeline(
            model=model, tokenizer=tokenizer, device='mps'
        )
        # Get the total number of examples
        tokenized_datasets = DatasetDict({
            "set": translations["set"].map(tokenize_function, batched=True),
        })
        total_examples = len(tokenized_datasets["set"])
        for test_number in tqdm(range(total_examples), desc="Translating"):
            example = tokenized_datasets["set"][test_number]
            original = example['akk']
            reference = example['en']
            if len(reference.strip()) < 2:
                #print(f"Skipping entry {test_number}: Reference is too short or empty.")
                continue
            # Generate predictions
            predicted_texts = pipe(
                original,
                max_length=max_length,
                truncation=True,
                #skip_special_tokens=True,
                #clean_up_tokenization_spaces=True
            )
            # Extract generated text and normalize spaces
            predicted_text = predicted_texts[0]['generated_text'].strip()
            # Apply the correction function
            corrected_predicted_text = correct_split_words(predicted_text, tokenizer)
            # return to origional data
            #rev_predicted_text_prediction = pipe(
            #    check_prompt + corrected_predicted_text,
            #    max_length=max_length+6,
            #    truncation=True,
                #skip_special_tokens=True,
                #clean_up_tokenization_spaces=True
            #)
            #rev_predicted_text =  rev_predicted_text_prediction[0]['generated_text'].strip()
            # Calculate BLEU-4 score with error handling
            try:
                if not predicted_text.strip():
                    bleu_score = 0.0
                else:
                    bleu = sacrebleu.sentence_bleu(
                        predicted_text, [reference], smooth_method='floor', smooth_value=0.1
                    )
                    adjusted_bleu = sacrebleu.sentence_bleu(
                        corrected_predicted_text, [reference], smooth_method='floor', smooth_value=0.1
                    )
                    #rev_bleu = sacrebleu.sentence_bleu(
                    #    rev_predicted_text, [remove_prompt(original)], smooth_method='floor', smooth_value=0.1
                    #)
                    bleu_score = bleu.score
                    adjusted_bleu_score = adjusted_bleu.score
                    #rev_bleu_score = rev_bleu.score
            except EOFError as e:
                print(f"Error processing entry {test_number}: {e}")
                bleu_score = 0.0
                adjusted_bleu_score = 0.0
            # Append the results
            results.append({
                "Cuneiform": original,
                "Reference": reference,
                "Predicted": predicted_text,
                "BLEU-4": bleu_score,
                "Adjusted Predicted": corrected_predicted_text,
                "Adjusted BLEU-4": adjusted_bleu_score
                #"Reverse Translate": rev_predicted_text,
                #"Reverse BLEU-4": rev_bleu_score
            })
        # Convert the list of dictionaries into a DataFrame
        result_df = pd.DataFrame(results)
        # Define the full path for the CSV file
        csv_file_path = filepath
        # Save the DataFrame to a CSV file in the specified directory
        result_df.to_csv(csv_file_path, index=False)
    return result_df
    collected = gc.collect()
    print("Garbage collector: collected %d objects." % (collected))

# Call the function

################
###Validation###
################
print("Generating Validation Akkadian cuneiform translations")
try:
    akk_val_pairs_cuneiform_translate_results = translate_cuneiform_set(akk_val_pairs_cuneiform_translate, 'val', 'akk_val_pairs_cuneiform_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian origional transliterated translations")
try:
    akk_val_pairs_origional_transliterated_translate_results = translate_cuneiform_set(akk_val_pairs_origional_transliterated_translate, 'val', 'akk_val_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian simple transliterated translations")
try:
    akk_val_pairs_simple_transliterated_translate_results = translate_cuneiform_set(akk_val_pairs_simple_transliterated_translate, 'val', 'akk_val_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian grouped transliterated translations")
try:
    akk_val_pairs_group_transliterated_translate_results = translate_cuneiform_set(akk_val_pairs_group_transliterated_translate, 'val', 'akk_val_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
    
print("Generating Validation Akkadian cuneiform origional transliterations")
try:
    akk_val_pairs_transliterated_origional_results = translate_cuneiform_set(akk_val_pairs_transliterated_origional, 'val', 'akk_val_pairs_transliterated_origional')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian cuneiform simple transliterations")
try:
    akk_val_pairs_transliterated_simple_results = translate_cuneiform_set(akk_val_pairs_transliterated_simple, 'val', 'akk_val_pairs_transliterated_simple')
except NameError as e:
    print("An error occurred:", str(e))


print("Generating Validation Akkadian cuneiform origional from transliterations")
try:
    akk_val_rev_pairs_transliterate_origional_results = translate_cuneiform_set(akk_val_rev_pairs_transliterate_origional, 'val', 'akk_val_rev_pairs_transliterate_origional')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian cuneiform simple from transliterations")
try:
    akk_val_rev_pairs_transliterate_simple_results = translate_cuneiform_set(akk_val_rev_pairs_transliterate_simple, 'val', 'akk_val_rev_pairs_transliterate_simple')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Validation Sumerian origional transliterated translations")
try:
    sux_val_pairs_origional_transliterated_translate_results = translate_cuneiform_set(sux_val_pairs_origional_transliterated_translate, 'val', 'sux_val_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Sumerian simple transliterated translations")
try:
    sux_val_pairs_simple_transliterated_translate_results = translate_cuneiform_set(sux_val_pairs_simple_transliterated_translate, 'val', 'sux_val_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian grouped transliterated translations")
try:
    sux_val_pairs_group_transliterated_translate_results = translate_cuneiform_set(sux_val_pairs_group_transliterated_translate, 'val', 'sux_val_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Validation Akkadian english-cuneiform translations")
try:
    akk_val_rev_pairs_cuneiform_translate_results = translate_cuneiform_set(akk_val_rev_pairs_cuneiform_translate, 'val', 'akk_val_rev_pairs_cuneiform_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian english-origional transliterated translations")
try:
    akk_val_rev_pairs_origional_transliterated_translate_results = translate_cuneiform_set(akk_val_rev_pairs_origional_transliterated_translate, 'val', 'akk_val_rev_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian english-simple transliterated translations")
try:
    akk_val_rev_pairs_simple_transliterated_translate_results = translate_cuneiform_set(akk_val_rev_pairs_simple_transliterated_translate, 'val', 'akk_val_rev_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian english-grouped transliterated translations")
try:
    akk_val_rev_pairs_group_transliterated_translate_results = translate_cuneiform_set(akk_val_rev_pairs_group_transliterated_translate, 'val', 'akk_val_rev_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Validation Sumerian english-origional transliterated translations")
try:
    sux_val_rev_pairs_origional_transliterated_translate_results = translate_cuneiform_set(sux_val_rev_pairs_origional_transliterated_translate, 'val', 'sux_val_rev_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Sumerian english-simple transliterated translations")
try:
    sux_val_rev_pairs_simple_transliterated_translate_results = translate_cuneiform_set(sux_val_rev_pairs_simple_transliterated_translate, 'val', 'sux_val_rev_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Validation Akkadian english-grouped transliterated translations")
try:
    sux_val_rev_pairs_group_transliterated_translate_results = translate_cuneiform_set(sux_val_rev_pairs_group_transliterated_translate, 'val', 'sux_val_rev_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

##########
###Test###
##########
print("Generating Test Akkadian cuneiform translations")
try:
    akk_test_pairs_cuneiform_translate_results = translate_cuneiform_set(akk_test_pairs_cuneiform_translate, 'test', 'akk_val_pairs_cuneiform_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian origional transliterated translations")
try:
    akk_test_pairs_origional_transliterated_translate_results = translate_cuneiform_set(akk_test_pairs_origional_transliterated_translate, 'test', 'akk_test_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian simple transliterated translations")
try:
    akk_test_pairs_simple_transliterated_translate_results = translate_cuneiform_set(akk_test_pairs_simple_transliterated_translate, 'test', 'akk_test_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian grouped transliterated translations")
try:
    akk_test_pairs_group_transliterated_translate_results = translate_cuneiform_set(akk_test_pairs_group_transliterated_translate, 'test', 'akk_test_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Test Akkadian cuneiform origional transliterations")
try:
    akk_test_pairs_transliterate_origional_results = translate_cuneiform_set(akk_test_pairs_transliterate_origional, 'test', 'akk_test_pairs_transliterate_origional')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian cuneiform simple transliterations")
try:
    akk_test_pairs_transliterate_simple_results = translate_cuneiform_set(akk_test_pairs_transliterate_simple, 'test', 'akk_test_pairs_transliterate_simple')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Test Akkadian cuneiform origional from transliterations")
try:
    akk_test_rev_pairs_transliterate_origional_results = translate_cuneiform_set(akk_val_rev_pairs_transliterate_origional, 'test', 'akk_test_rev_pairs_transliterate_origional')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian cuneiform simple from transliterations")
try:
    akk_test_rev_pairs_transliterate_simple_results = translate_cuneiform_set(akk_testrev_pairs_transliterate_simple, 'test', 'akk_test_rev_pairs_transliterate_simple')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Test Sumerian origional transliterated translations")
try:
    sux_test_pairs_origional_transliterated_translate_results = translate_cuneiform_set(sux_test_pairs_origional_transliterated_translate, 'test', 'sux_test_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Sumerian simple transliterated translations")
try:
    sux_test_pairs_simple_transliterated_translate_results = translate_cuneiform_set(sux_test_pairs_simple_transliterated_translate, 'test', 'sux_test_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian grouped transliterated translations")
try:
    sux_test_pairs_group_transliterated_translate_results = translate_cuneiform_set(sux_test_pairs_group_transliterated_translate, 'test', 'sux_test_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Test Akkadian english-cuneiform translations")
try:
    akk_test_rev_pairs_cuneiform_translate_results = translate_cuneiform_set(akk_test_rev_pairs_cuneiform_translate, 'test', 'akk_test_rev_pairs_cuneiform_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian english-origional transliterated translations")
try:
    akk_test_rev_pairs_origional_transliterated_translate_results = translate_cuneiform_set(akk_test_rev_pairs_origional_transliterated_translate, 'test', 'akk_test_rev_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian english-simple transliterated translations")
try:
    akk_test_rev_pairs_simple_transliterated_translate_results = translate_cuneiform_set(akk_test_rev_pairs_simple_transliterated_translate, 'test', 'akk_test_rev_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian english-grouped transliterated translations")
try:
    akk_test_rev_pairs_group_transliterated_translate_results = translate_cuneiform_set(akk_test_rev_pairs_group_transliterated_translate, 'val', 'akk_test_rev_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Test Sumerian english-origional transliterated translations")
try:
    sux_test_rev_pairs_origional_transliterated_translate_results = translate_cuneiform_set(sux_test_rev_pairs_origional_transliterated_translate, 'test', 'sux_test_rev_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Sumerian english-simple transliterated translations")
try:
    sux_test_rev_pairs_simple_transliterated_translate_results = translate_cuneiform_set(sux_test_rev_pairs_simple_transliterated_translate, 'test', 'sux_test_rev_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Test Akkadian english-grouped transliterated translations")
try:
    sux_test_rev_pairs_group_transliterated_translate_results = translate_cuneiform_set(sux_test_rev_pairs_group_transliterated_translate, 'test', 'sux_test_rev_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

###########
###Train###
###########
print("Generating Train Akkadian cuneiform translations")
try:
    akk_train_pairs_cuneiform_translate_results = translate_cuneiform_set(akk_train_pairs_cuneiform_translate, 'train', 'akk_train_pairs_cuneiform_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian origional transliterated translations")
try:
    akk_train_pairs_origional_transliterated_translate_results = translate_cuneiform_set(akk_train_pairs_origional_transliterated_translate, 'train', 'akk_train_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian simple transliterated translations")
try:
    akk_train_pairs_simple_transliterated_translate_results = translate_cuneiform_set(akk_train_pairs_simple_transliterated_translate, 'train', 'akk_train_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian grouped transliterated translations")
try:
    akk_train_pairs_group_transliterated_translate_results = translate_cuneiform_set(akk_train_pairs_group_transliterated_translate, 'train', 'akk_train_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Train Akkadian cuneiform origional transliterations")
try:
    akk_train_pairs_transliterated_origional_results = translate_cuneiform_set(akk_train_pairs_transliterated_origional, 'train', 'akk_train_pairs_transliterated_origional')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian cuneiform simple transliterations")
try:
    akk_train_pairs_transliterated_simple_results = translate_cuneiform_set(akk_train_pairs_transliterated_simple, 'val', 'akk_train_pairs_transliterated_simple')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Train Akkadian cuneiform origional from transliterations")
try:
    akk_train_rev_pairs_transliterate_origional_results = translate_cuneiform_set(akk_train_rev_pairs_transliterate_origional, 'train', 'akk_train_rev_pairs_transliterate_origional')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian cuneiform simple from transliterations")
try:
    akk_train_rev_pairs_transliterate_simple_results = translate_cuneiform_set(akk_train_rev_pairs_transliterate_simple, 'val', 'akk_val_rev_pairs_transliterate_simple')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Train Sumerian origional transliterated translations")
try:
    sux_train_pairs_origional_transliterated_translate_results = translate_cuneiform_set(sux_train_pairs_origional_transliterated_translate, 'train', 'sux_val_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Sumerian simple transliterated translations")
try:
    sux_train_pairs_simple_transliterated_translate_results = translate_cuneiform_set(sux_train_pairs_simple_transliterated_translate, 'train', 'sux_train_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian grouped transliterated translations")
try:
    sux_train_pairs_group_transliterated_translate_results = translate_cuneiform_set(sux_train_pairs_group_transliterated_translate, 'train', 'sux_train_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Train Akkadian english-cuneiform translations")
try:
    akk_train_rev_pairs_cuneiform_translate_results = translate_cuneiform_set(akk_train_rev_pairs_cuneiform_translate, 'train', 'akk_train_rev_pairs_cuneiform_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian english-origional transliterated translations")
try:
    akk_train_rev_pairs_origional_transliterated_translate_results = translate_cuneiform_set(akk_train_rev_pairs_origional_transliterated_translate, 'train', 'akk_train_rev_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian english-simple transliterated translations")
try:
    akk_train_rev_pairs_simple_transliterated_translate_results = translate_cuneiform_set(akk_train_rev_pairs_simple_transliterated_translate, 'train', 'akk_train_rev_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian english-grouped transliterated translations")
try:
    akk_train_rev_pairs_group_transliterated_translate_results = translate_cuneiform_set(akk_train_rev_pairs_group_transliterated_translate, 'train', 'akk_train_rev_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Train Sumerian english-origional transliterated translations")
try:
    sux_train_rev_pairs_origional_transliterated_translate_results = translate_cuneiform_set(sux_train_rev_pairs_origional_transliterated_translate, 'train', 'sux_train_rev_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Sumerian english-simple transliterated translations")
try:
    sux_train_rev_pairs_simple_transliterated_translate_results = translate_cuneiform_set(sux_train_rev_pairs_simple_transliterated_translate, 'train', 'sux_train_rev_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Akkadian english-grouped transliterated translations")
try:
    sux_train_rev_pairs_group_transliterated_translate_results = translate_cuneiform_set(sux_train_rev_pairs_group_transliterated_translate, 'train', 'sux_train_rev_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Train Elamite origional transliterated translations")
try:
    elx_train_pairs_origional_transliterated_translate_results = translate_cuneiform_set(elx_train_pairs_origional_transliterated_translate, 'train', 'elx_val_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Elamite simple transliterated translations")
try:
    elx_train_pairs_simple_transliterated_translate_results = translate_cuneiform_set(elx_train_pairs_simple_transliterated_translate, 'train', 'elx_train_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Elamite grouped transliterated translations")
try:
    elx_train_pairs_group_transliterated_translate_results = translate_cuneiform_set(elx_train_pairs_group_transliterated_translate, 'train', 'elx_train_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))

print("Generating Train Elamite english-origional transliterated translations")
try:
    elx_train_rev_pairs_origional_transliterated_translate_results = translate_cuneiform_set(elx_train_rev_pairs_origional_transliterated_translate, 'train', 'elx_train_rev_pairs_origional_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Elamite english-simple transliterated translations")
try:
    elx_train_rev_pairs_simple_transliterated_translate_results = translate_cuneiform_set(elx_train_rev_pairs_simple_transliterated_translate, 'train', 'elx_train_rev_pairs_simple_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
print("Generating Train Elamite english-grouped transliterated translations")
try:
    elx_train_rev_pairs_group_transliterated_translate_results = translate_cuneiform_set(elx_train_rev_pairs_group_transliterated_translate, 'train', 'elx_train_rev_pairs_group_transliterated_translate')
except NameError as e:
    print("An error occurred:", str(e))
