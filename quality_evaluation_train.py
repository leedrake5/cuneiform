prediction_length = 60

import sys, os, datetime, pwd
user_directory = pwd.getpwuid(os.getuid()).pw_dir

save_directory = os.path.join(user_directory, 'GitHub', 'results', 't5-base-p-l-akk-en-20241110-154709')

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel

from datasets import Dataset

import pandas as pd
from glob import glob

evaluation_files = [f for f in os.listdir(os.path.join(save_directory, 'evaluation')) if f.endswith('.csv')]

train_dfs, test_dfs = [], [] # Placeholders for DataFrames

for file in evaluation_files:
    filepath = os.path.join(save_directory, 'evaluation', file)
    df = pd.read_csv(filepath)   # Read the csv file into a pandas dataframe
    if '_train_set' in file:
        train_dfs.append(df)
    elif '_test_set' in file:
        train_dfs.append(df)
    elif "_val_set" in file:
        test_dfs.append(df)

# Merge the DataFrames by column name
train_set = pd.concat(train_dfs, axis=0).reset_index(drop=True).dropna()  # `axis=1` for merging by columns
test_set = pd.concat(test_dfs, axis=0).reset_index(drop=True).dropna()


def remove_prompt(text):
    # Split the text at the first occurrence of ':'
    parts = text.split(':', 1)
    if len(parts) > 1:
        # Return the part after the colon, stripping any leading/trailing whitespace
        return parts[1].strip()
    else:
        # If no colon is found, return the original text
        return text.strip()

train_og_prompts = train_set.iloc[:, 0].tolist()
train_prompts = list(map(remove_prompt, train_og_prompts))
train_predictions = train_set['Adjusted Predicted'].tolist()
train_bleu_scores = train_set['Adjusted BLEU-4'].tolist()

test_og_prompts = test_set.iloc[:, 0].tolist()
test_prompts = list(map(remove_prompt, test_og_prompts))
test_predictions = test_set['Adjusted Predicted'].tolist()
test_bleu_scores = test_set['Adjusted BLEU-4'].tolist()

train_data_dict = {"prompt": [single for single in train_prompts], "prediction": [single for single in train_predictions], "bleu_score": [single for single in train_bleu_scores]}
test_data_dict = {"prompt": [single for single in test_prompts], "prediction": [single for single in test_predictions], "bleu_score": [single for single in test_bleu_scores]}

from datasets import DatasetDict, Dataset
train_dataset = Dataset.from_dict(train_data_dict)
test_dataset = Dataset.from_dict(test_data_dict)
translations = DatasetDict({"train": train_dataset, "test": test_dataset})


def preprocess_function(examples):
    ref_texts = examples['prompt']  # Adjusted to 'prompt' if that's your dataset's reference text column
    pred_texts = examples['prediction']
    bleu_scores = examples['bleu_score']
    # Compute lengths of texts
    ref_lengths = [len(text) for text in ref_texts]
    pred_lengths = [len(text) for text in pred_texts]
    # Tokenize reference texts
    ref_encodings = tokenizer(
        ref_texts,
        padding='max_length',
        truncation=True,
        max_length=128,
    )
    # Tokenize prediction texts
    pred_encodings = tokenizer(
        pred_texts,
        padding='max_length',
        truncation=True,
        max_length=128,
    )
    # Prepare labels
    labels = bleu_scores
    # Combine encodings and include lengths
    encodings = {
        'ref_input_ids': ref_encodings['input_ids'],
        'ref_attention_mask': ref_encodings['attention_mask'],
        'pred_input_ids': pred_encodings['input_ids'],
        'pred_attention_mask': pred_encodings['attention_mask'],
        'labels': labels,
        'ref_length': ref_lengths,
        'pred_length': pred_lengths,
    }
    return encodings

model_name = 'google-bert/bert-base-multilingual-cased'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
base_model = BertModel.from_pretrained(model_name)

###Update Tokenizer
# Extract unique tokens from English dataset
def extract_unique_tokens(dataset, column):
    unique_tokens = set()
    for example in dataset:
        tokens = example[column].split()  # Adjust token splitting if necessary
        unique_tokens.update(tokens)
    return unique_tokens

train_unique_tokens_prompt = extract_unique_tokens(translations['train'], 'prompt')
test_unique_tokens_prompt = extract_unique_tokens(translations['test'], 'prompt')
unique_tokens_prompt = train_unique_tokens_prompt.union(test_unique_tokens_prompt)

train_unique_tokens_prediction = extract_unique_tokens(translations['train'], 'prediction')
test_unique_tokens_prediction = extract_unique_tokens(translations['test'], 'prediction')
unique_tokens_prediction = train_unique_tokens_prediction.union(test_unique_tokens_prediction)

unique_new_tokens = unique_tokens_prompt.symmetric_difference(unique_tokens_prediction)
print(f'Total training data tokens: {len(unique_new_tokens)}')


# Get current tokenizer vocabulary
current_vocab = set(tokenizer.get_vocab().keys())
print(f'Total current tokens: {len(current_vocab)}')

# Find new tokens that are not in the current vocabulary
new_tokens = unique_new_tokens - current_vocab
print(f'Total new tokens: {len(new_tokens)}')

cleaned_new_tokens = []
for token in new_tokens:
    if token.strip() == '':
        continue  # Skip empty tokens
    if ' ' in token:
        continue  # Skip tokens with spaces
    cleaned_new_tokens.append(token)

print(f"Total cleaned new tokens: {len(cleaned_new_tokens)}")

# Add new tokens to the tokenizer
if cleaned_new_tokens:
    num_added_toks = tokenizer.add_tokens(cleaned_new_tokens)
    print(f"Added {num_added_toks} tokens")
else:
    print("No valid new tokens to add")

# Resize model token embeddings to accommodate new tokens
base_model.resize_token_embeddings(len(tokenizer))

###Map tokens to data
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

from transformers import PreTrainedModel

class TranslationQualityEstimator(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        model_name = config.model_name_or_path
        self.encoder = base_model  # Use your pre-loaded base model
        hidden_size = self.encoder.config.hidden_size
        # Define convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1  # 'same' padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        # Define fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2 + 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.loss_fn = nn.MSELoss()
    def forward(
        self,
        ref_input_ids=None,
        ref_attention_mask=None,
        pred_input_ids=None,
        pred_attention_mask=None,
        labels=None,
        ref_length=None,
        pred_length=None,
    ):
        # Encode reference texts
        ref_outputs = self.encoder(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            return_dict=True,
        )
        # Encode prediction texts
        pred_outputs = self.encoder(
            input_ids=pred_input_ids,
            attention_mask=pred_attention_mask,
            return_dict=True,
        )
        # Get last hidden states (batch_size, seq_length, hidden_size)
        ref_embedding = ref_outputs.last_hidden_state
        pred_embedding = pred_outputs.last_hidden_state
        # Transpose to (batch_size, hidden_size, seq_length)
        ref_embedding = ref_embedding.transpose(1, 2)
        pred_embedding = pred_embedding.transpose(1, 2)
        # Apply Conv1D and Dropout to reference embeddings
        ref_conv = self.conv1(ref_embedding)
        ref_conv = self.relu(ref_conv)
        ref_conv = self.dropout(ref_conv)
        ref_conv = self.conv2(ref_conv)
        ref_conv = self.relu(ref_conv)
        ref_conv = self.dropout(ref_conv)
        # Apply Conv1D and Dropout to prediction embeddings
        pred_conv = self.conv1(pred_embedding)
        pred_conv = self.relu(pred_conv)
        pred_conv = self.dropout(pred_conv)
        pred_conv = self.conv2(pred_conv)
        pred_conv = self.relu(pred_conv)
        pred_conv = self.dropout(pred_conv)
        # Global Max Pooling over the sequence length
        ref_pooled = torch.max(ref_conv, dim=2)[0]  # Shape: (batch_size, hidden_size)
        pred_pooled = torch.max(pred_conv, dim=2)[0]  # Shape: (batch_size, hidden_size)
        # Combine embeddings and lengths
        # Convert lengths to tensors and normalize if necessary
        ref_length = ref_length.unsqueeze(1).float()
        pred_length = pred_length.unsqueeze(1).float()
        # Concatenate embeddings and lengths
        combined_embedding = torch.cat((ref_pooled, pred_pooled, ref_length, pred_length), dim=1)
        # Pass through fully connected layers
        x = self.fc1(combined_embedding)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)  # Output shape: (batch_size)
        outputs = {'logits': logits}
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs['loss'] = loss
        return outputs

from transformers import PretrainedConfig

class TranslationQualityConfig(PretrainedConfig):
    model_type = "translation_quality"
    def __init__(self, model_name_or_path='bert-base-multilingual-cased', **kwargs):
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path

config = TranslationQualityConfig(model_name_or_path=model_name)
model = TranslationQualityEstimator(config)

from transformers import default_data_collator

def data_collator(features):
    batch = {
        'ref_input_ids': torch.stack([torch.tensor(f['ref_input_ids'], dtype=torch.long) for f in features]),
        'ref_attention_mask': torch.stack([torch.tensor(f['ref_attention_mask'], dtype=torch.long) for f in features]),
        'pred_input_ids': torch.stack([torch.tensor(f['pred_input_ids'], dtype=torch.long) for f in features]),
        'pred_attention_mask': torch.stack([torch.tensor(f['pred_attention_mask'], dtype=torch.long) for f in features]),
        'labels': torch.tensor([f['labels'] for f in features], dtype=torch.float),
        'ref_length': torch.tensor([f['ref_length'] for f in features], dtype=torch.float),
        'pred_length': torch.tensor([f['pred_length'] for f in features], dtype=torch.float),
    }
    return batch

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=os.path.join(save_directory, 'qe_model'),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-4,
    weight_decay=0.01,
    logging_steps=10,
)

from transformers import Trainer
import numpy as np
from scipy.stats import pearsonr

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.flatten()
    labels = labels.flatten()
    mse = ((predictions - labels) ** 2).mean().item()
    pearson_corr, _ = pearsonr(predictions, labels)
    return {'mse': mse, 'pearson': pearson_corr}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()
####Push it
trainer.push_to_hub()



