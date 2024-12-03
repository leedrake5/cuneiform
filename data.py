import sys, os, datetime
import json
import torch
import random
import glob
from tqdm.notebook import tqdm
# from transformers import AutoTokenizer
# from transformers import DataCollatorForSeq2Seq
# from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import TranslationPipeline
from datasets import load_dataset, Dataset

os.chdir("/Users/lee/GitHub/CuneiformTranslators/tools")
import cdli
import languages


os.environ["TOKENIZERS_PARALLELISM"] = "false"

source_langs = set(["akk", "sux", "elx", "arc", "peo", "uga", "grc"])

# target_langs = set(["en", "it", "es", "fr", "de"])
target_langs = set(["en"])



paragraphs = True

test_publication_ids = set(["P393923"])

output_path = "../data/dataset_index.json"


publications = cdli.get_atf()

len(publications), "publications"

def target_ok(target_text):
    if len(target_text) == 0:
        return False
    if len(set(target_text.replace(" ", ""))) < 2:
        return False
    return True
    

def test_target_ok(text):
    ok = target_ok(text)
    print(ok, repr(text))

test_target_ok("")
test_target_ok(" ")
test_target_ok("xx xxx x")
test_target_ok(".. . .. ")
test_target_ok("Hi")

def pub_has_text(pub):
    for a in pub.text_areas:
        for l in a.lines:
            if len(l.text) > 0 and "en" in l.languages and target_ok(l.languages["en"]):
                return True
    return False

akk_pubs = [x for x in publications if x.language == "akk" and pub_has_text(x)]
print(len(akk_pubs), "akk pubs")

sux_pubs = [x for x in publications if x.language == "sux" and pub_has_text(x)]
print(len(sux_pubs), "sux pubs")

elx_pubs = [x for x in publications if x.language == "elx" and pub_has_text(x)]
print(len(elx_pubs), "elx pubs")

arc_pubs = [x for x in publications if x.language == "arc" and pub_has_text(x)]
print(len(arc_pubs), "arc pubs")

uga_pubs = [x for x in publications if x.language == "uga" and pub_has_text(x)]
print(len(uga_pubs), "uga pubs")

peo_pubs = [x for x in publications if x.language == "peo" and pub_has_text(x)]
print(len(peo_pubs), "peo pubs")

grc_pubs = [x for x in publications if x.language == "grc" and pub_has_text(x)]
print(len(grc_pubs), "grc pubs")


def train_test_split(pubs, test_split=5, val_split=5):
    r = list(pubs)
    random.shuffle(r)
    n = len(r)
    train = []
    test = []
    val = []
    for p in r:
        # Generate a random number to decide where to place the publication
        rnd = random.uniform(0, 100)
        # Assuming 'test_publication_ids' is predefined and contains specific publications to be in the test
        if rnd < test_split or p.id in test_publication_ids:
            test.append(p.id)
        elif rnd < test_split + val_split:
            val.append(p.id)
        else:
            train.append(p.id)
    ntrain = len(train)
    ntest = len(test)
    nval = len(val)
    print(ntrain, "train")
    print(ntest, "test")
    print(nval, "val")
    return {"train": sorted(train), "test": sorted(test), "val": sorted(val)}
    
    
def train_test_split_single(pubs, test_split=5, val_split=5):
    r = list(pubs)
    random.shuffle(r)
    n = len(r)
    train = []
    test = []
    val = []
    for p in r:
        # Generate a random number to decide where to place the publication
        rnd = random.uniform(0, 100)
        # Assuming 'test_publication_ids' is predefined and contains specific publications to be in the test
        if rnd < test_split or p.id in test_publication_ids:
            train.append(p.id)
        elif rnd < test_split + val_split:
                        train.append(p.id)
        else:
            train.append(p.id)
    ntrain = len(train)
    ntest = len(test)
    nval = len(val)
    print(ntrain, "train")
    print(ntest, "test")
    print(nval, "val")
    return {"train": sorted(train), "test": sorted(test), "val": sorted(val)}

print("akk")
akk_split = train_test_split(akk_pubs)
print("sux")
sux_split = train_test_split(sux_pubs)
print("elx")
elx_split = train_test_split_single(elx_pubs)
print("arc")
arc_split = train_test_split(arc_pubs)
print("peo")
peo_split = train_test_split(peo_pubs)
print("uga")
uga_split = train_test_split(uga_pubs)
print("grc")
grc_split = train_test_split(grc_pubs)

output_obj = { "akk": akk_split, "sux": sux_split, "elx": elx_split, "arc": arc_split, "uga": uga_split, "peo": peo_split, "grc": grc_split }
output_obj.keys()

output_json = json.dumps(output_obj)
output_json[:100]

with open(output_path, "wt") as f:
    f.write(output_json)

dataset_index = output_obj
source_langs = set(["sux"])
target_langs = set(["en"])



def get_prefix(src_lang, tgt_lang):
    s = languages.all_languages[src_lang]
    t = languages.all_languages[tgt_lang]
    return f""

def get_pubs_targets(dataset, s='elx', t='en'):
    new_sourceandtargets = []
    added_sources = set()
    def add_line_ranges(area, b, e):
    #                     print("-"*50)
        ls = " ".join([x.text for x in area.lines[b:e]])
        ls = " ".join(ls.split(" "))
        prefixed_ls = st_prefix + ls
        if prefixed_ls in added_sources:
            return
        lt = " ".join([(x.languages[t] if t in x.languages else "") for x in area.lines[b:e]])
        lt = " ".join(lt.split(" "))
        lt = languages.replace_unsupported_en(lt)
        if not target_ok(lt):
            return
    #                     print(ls)
    #                     print(lt)
        added_sources.add(prefixed_ls)
        new_sourceandtargets.append((prefixed_ls, lt))
        if is_bi:
            new_sourceandtargets.append((ts_prefix + lt, ls))
    for s in source_langs:
        pub_index = dataset_index[s][dataset]
        for t in target_langs:
            print("Preparing", s, "to", t)
            st_prefix = get_prefix(s, t)
            ts_prefix = get_prefix(t, s)
            for pub in tqdm([p for p in publications if p.language==s and p.id in pub_index]):
                for area in pub.text_areas:
                    if not any(x for x in area.lines if t in x.languages):
                        continue
                    use_paragraphs=False
                    use_lines=True
                    if use_paragraphs:
                        paragraphs = area.lines_to_paragraphs(s)
                        line_ranges = []
                        for p in paragraphs:
                            wlines = wrap_paragraph(p, area.lines, s, t)
                            line_ranges.extend(wlines)
        #                 print("="*50, len(area.lines))
                        for b, e in line_ranges:
                            add_line_ranges(area, b, e)
                    if use_lines:
                        for i, _ in enumerate(area.lines):
                            add_line_ranges(area, i, i + 1)
    random.shuffle(new_sourceandtargets)
    return Dataset.from_dict({"source": [x[0] for x in new_sourceandtargets], "target": [x[1] for x in new_sourceandtargets]})

is_bi = False
use_paragraphs = True
use_lines = True


date_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
flags = ""
suffix = ""
if is_bi:
    flags += "-bi"

if use_paragraphs:
    flags += "-p"

if use_lines:
    flags += "-l"


train_dataset = get_pubs_targets("train")
test_dataset = get_pubs_targets("test")
val_dataset = get_pubs_targets("val")
print(len(train_dataset), "train")
print(len(test_dataset), "test")
print(len(val_dataset), "val")

def write_dataset_to_files(dataset, source_file, target_file):
    source_texts = [item['source'] for item in dataset]
    target_texts = [item['target'] for item in dataset]
    with open(source_file, 'w', encoding='utf-8') as f:
        for text in source_texts:
            f.write(text + '\n')
    with open(target_file, 'w', encoding='utf-8') as f:
        for text in target_texts:
            f.write(text + '\n')

write_dataset_to_files(train_dataset, '/Users/lee/GitHub/cuneiform/data/e_train.tr', '/Users/lee/GitHub/cuneiform/data/e_train.en')


write_dataset_to_files(train_dataset, '/Users/lee/GitHub/cuneiform/data/s_train.tr', '/Users/lee/GitHub/cuneiform/data/s_train.en')
write_dataset_to_files(test_dataset, '/Users/lee/GitHub/cuneiform/data/s_test.tr', '/Users/lee/GitHub/cuneiform/data/s_test.en')
write_dataset_to_files(val_dataset, '/Users/lee/GitHub/cuneiform/data/s_valid.tr', '/Users/lee/GitHub/cuneiform/data/s_valid.en')

###Akkadian
def process_and_split_text_file(input_file_path, train_file_path, test_file_path, val_file_path, test_split=5, val_split=5):
    """
    Processes a text file to remove all characters before and including the colon ':'
    on each line and removes any leading and trailing whitespace. It then splits the data
    into training, testing, and validation sets based on provided percentages.
    Parameters:
    input_file_path (str): Path to the input text file.
    train_file_path (str): Path to the output text file for the training data.
    test_file_path (str): Path to the output text file for the test data.
    val_file_path (str): Path to the output text file for the validation data.
    test_split (int): Percentage of data to allocate to the test set.
    val_split (int): Percentage of data to allocate to the validation set.
    Returns:
    None
    """
    try:
        # Open the input file and read the lines
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        # Process each line
        processed_lines = [line.split(':', 1)[1].strip() if ':' in line else line.strip() for line in lines]
        # Initialize datasets
        train, test, val = [], [], []
        # Split data according to the specified percentages
        for line in processed_lines:
            rnd = random.randint(1, 100)
            if rnd <= test_split:
                test.append(line)
            elif rnd <= test_split + val_split:
                val.append(line)
            else:
                train.append(line)
        # Write each dataset to its respective file
        for dataset, path in zip([train, test, val], [train_file_path, test_file_path, val_file_path]):
            with open(path, 'w', encoding='utf-8') as file:
                for item in dataset:
                    file.write(item + '\n')
        print("Files processed successfully. Outputs saved to:", train_file_path, test_file_path, val_file_path)
    except Exception as e:
        print("An error occurred:", str(e))

# Example usage of the function
input_path = '/Users/lee/GitHub/Akkademia/NMT_input/translation_per_line.txt'  # Path to your input file
train_path = '/Users/lee/GitHub/cuneiform/data/a_train.en'  # Path to your training output file
test_path = '/Users/lee/GitHub/cuneiform/data/a_test.en'    # Path to your test output file
val_path = '/Users/lee/GitHub/cuneiform/data/a_valid.en'      # Path to your validation output file
process_and_split_text_file(input_path, train_path, test_path, val_path, test_split=5, val_split=5)

###Transcription
input_path = '/Users/lee/GitHub/Akkademia/NMT_input/transcriptions_per_line.txt'  # Path to your input file
train_path = '/Users/lee/GitHub/cuneiform/data/a_train.tr'  # Path to your training output file
test_path = '/Users/lee/GitHub/cuneiform/data/a_test.tr'    # Path to your test output file
val_path = '/Users/lee/GitHub/cuneiform/data/a_valid.tr'      # Path to your validation output file
process_and_split_text_file(input_path, train_path, test_path, val_path, test_split=5, val_split=5)


###Cuneiform
input_path = '/Users/lee/GitHub/Akkademia/NMT_input/signs_per_line.txt'  # Path to your input file
train_path = '/Users/lee/GitHub/cuneiform/data/a_train.cu'  # Path to your training output file
test_path = '/Users/lee/GitHub/cuneiform/data/a_test.cu'    # Path to your test output file
val_path = '/Users/lee/GitHub/cuneiform/data/a_valid.cu'      # Path to your validation output file
process_and_split_text_file(input_path, train_path, test_path, val_path, test_split=5, val_split=5)


import random

def process_and_split_text_files(input_paths, output_paths, test_split=5, val_split=5):
    """
    Processes multiple text files to remove all characters before and including the colon ':'
    on each line, removes any leading and trailing whitespace, and then splits the data
    into training, testing, and validation sets based on provided percentages.
    This function ensures that the split is synchronized across all input files.
    Parameters:
    input_paths (dict): Dictionary with keys 'cuneiform', 'transcription', 'english' and their corresponding file paths.
    output_paths (dict): Dictionary with keys 'train', 'test', 'val', each containing a dict with paths for 'cuneiform', 'transcription', 'english'.
    test_split (int): Percentage of data to allocate to the test set.
    val_split (int): Percentage of data to allocate to the validation set.
    Returns:
    None
    """
    try:
        # Read and process lines from all files
        data = {key: [] for key in input_paths}
        for key, path in input_paths.items():
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                data[key] = [line.split(':', 1)[1].strip() if ':' in line else line.strip() for line in lines]
        # Number of lines (assuming all files have the same number of lines)
        num_lines = len(data['cuneiform'])
        # Initialize datasets
        datasets = {key: {'train': [], 'test': [], 'val': []} for key in input_paths}
        # Split data according to the specified percentages
        for i in range(num_lines):
            rnd = random.randint(1, 100)
            if rnd <= test_split:
                split = 'test'
            elif rnd <= test_split + val_split:
                split = 'val'
            else:
                split = 'train'
            for key in input_paths:
                datasets[key][split].append(data[key][i])
        # Write each dataset to its respective file
        for split_key, split_paths in output_paths.items():
            for data_key, file_path in split_paths.items():
                with open(file_path, 'w', encoding='utf-8') as file:
                    for item in datasets[data_key][split_key]:
                        file.write(item + '\n')
        print("Files processed and split successfully. Outputs saved to the provided paths.")
    except Exception as e:
        print("An error occurred:", str(e))

# Example usage
input_paths = {
    'cuneiform': '/Users/lee/GitHub/Akkademia/NMT_input/signs_per_line.txt',
    'transcription': '/Users/lee/GitHub/Akkademia/NMT_input/transcriptions_per_line.txt',
    'english': '/Users/lee/GitHub/Akkademia/NMT_input/translation_per_line.txt'
}
output_paths = {
    'train': {'cuneiform': '/Users/lee/GitHub/cuneiform/data/a_train.cu', 'transcription': '/Users/lee/GitHub/cuneiform/data/a_train.tr', 'english': '/Users/lee/GitHub/cuneiform/data/a_train.en'},
    'test': {'cuneiform': '/Users/lee/GitHub/cuneiform/data/a_test.cu', 'transcription': '/Users/lee/GitHub/cuneiform/data/a_test.tr', 'english': '/Users/lee/GitHub/cuneiform/data/a_test.en'},
    'val': {'cuneiform': '/Users/lee/GitHub/cuneiform/data/a_valid.cu', 'transcription': '/Users/lee/GitHub/cuneiform/data/a_valid.tr', 'english': '/Users/lee/GitHub/cuneiform/data/a_valid.en'}
}

process_and_split_text_files(input_paths, output_paths, test_split=5, val_split=5)

####Unlabeled
def target_not_ok(target_text):
    if len(target_text) == 0:
        return True
    if len(set(target_text.replace(" ", ""))) < 2:
        return True
    return False


def pub_has_no_text(pub):
    for a in pub.text_areas:
        for l in a.lines:
            # Check if line has text, "en" is in languages, and target is not okay
            # Also check if 'en' is not present
            if (len(l.text) > 0 and
                ("en" not in l.languages)):
                return True
    return False

akk_pubs = [x for x in publications if x.language == "akk" and pub_has_no_text(x)]
print(len(akk_pubs), "akk pubs")

sux_pubs = [x for x in publications if x.language == "sux" and pub_has_no_text(x)]
print(len(sux_pubs), "sux pubs")

elx_pubs = [x for x in publications if x.language == "elx" and pub_has_no_text(x)]
print(len(elx_pubs), "elx pubs")

arc_pubs = [x for x in publications if x.language == "arc" and pub_has_no_text(x)]
print(len(arc_pubs), "arc pubs")

uga_pubs = [x for x in publications if x.language == "uga" and pub_has_no_text(x)]
print(len(uga_pubs), "uga pubs")

peo_pubs = [x for x in publications if x.language == "peo" and pub_has_no_text(x)]
print(len(peo_pubs), "peo pubs")

grc_pubs = [x for x in publications if x.language == "grc" and pub_has_no_text(x)]
print(len(grc_pubs), "grc pubs")

print("akk")
akk_u_split = train_test_split_single(akk_pubs)
print("sux")
sux_u_split = train_test_split_single(sux_pubs)
print("elx")
elx_u_split = train_test_split_single(elx_pubs)


output_obj = { "akk": akk_u_split, "sux": sux_u_split, "elx": elx_u_split}
output_obj.keys()

#output_json = json.dumps(output_obj)
#output_json[:100]

#with open(output_path, "wt") as f:
#    f.write(output_json)

dataset_index = output_obj
source_langs = set(["elx"])

def get_single_prefix(src_lang):
    s = languages.all_languages[src_lang]
    return f""


def get_source_pubs(dataset, source_lang='elx'):
    sources = []  # List to store sources
    added_sources = set()  # Set to avoid duplicates
    def add_line_ranges(area, b, e):
        ls = " ".join([x.text for x in area.lines[b:e]])
        ls = " ".join(ls.split())  # Normalize spaces
        prefixed_ls = source_prefix + ls
        if prefixed_ls in added_sources:
            return
        added_sources.add(prefixed_ls)
        sources.append(prefixed_ls)  # Only add the source text
    for s in source_langs:
        pub_index = dataset_index[s][dataset]
        print("Preparing", s)
        source_prefix = get_single_prefix(s)  # Adjusted to not require target language
        for pub in tqdm([p for p in publications if p.language == s and p.id in pub_index]):
            for area in pub.text_areas:
                use_paragraphs = False
                use_lines = True
                if use_paragraphs:
                    paragraphs = area.lines_to_paragraphs(s)
                    line_ranges = []
                    for p in paragraphs:
                        wlines = wrap_paragraph(p, area.lines, s)
                        line_ranges.extend(wlines)
                    for b, e in line_ranges:
                        add_line_ranges(area, b, e)
                if use_lines:
                    for i, _ in enumerate(area.lines):
                        add_line_ranges(area, i, i + 1)
    random.shuffle(sources)
    return Dataset.from_dict({"source": sources})  # Return a dataset with only sources


train_dataset = get_source_pubs("train")
print(len(train_dataset), "train")

def write_source_dataset_to_files(dataset, source_file):
    source_texts = [item['source'] for item in dataset]
    with open(source_file, 'w', encoding='utf-8') as f:
        for text in source_texts:
            f.write(text + '\n')

write_source_dataset_to_files(train_dataset, '/Users/lee/GitHub/cuneiform/data/e_u_train.tr')
