### Cuneiform

This repository includes code for training new models using huggingface's transformers package. An example of this using [google's t5-small](https://huggingface.co/google-t5/t5-small) as a baseline is available [here](https://huggingface.co/Thalesian/akk-111m). These models are designed to handle a variety of translation and transcription tasks. 

- Akkadian: 𒄿 𒈾 𒌗 𒃶 𒌓 𒐉 𒆚 𒀀 𒈾 𒆳 𒆸 𒄭 𒇻 𒁺 𒅅 𒆳 𒁀 𒀀 𒍝 𒆳 𒊓 𒅈 𒁀 𒇷 𒀀 𒆳 𒁲 𒁺 𒀀 𒆷 𒀀 𒁲 𒌷 𒈨 𒌍 𒉌 𒃻 𒅆 𒁲 𒀀 𒇉 𒊒 𒌑 𒊒 𒊭 𒆳 𒈨 𒄴 𒊑 𒀝 𒋤 𒊩 𒆷 𒋢 𒉡 𒃻 𒋗 𒈨 𒌍 𒋗 𒉡 𒌑 𒊺 𒍝 𒀀 𒀀 𒈾 𒌷 𒅀 𒀸 𒋩 𒌒 𒆷' 
- English: 'in the month kislimu the fourth day i marched to the land habhu i conquered the lands bazu sarbaliu and didualu together with the cities on the banks of the river ruru of the land mehru i brought forth their booty and possessions and brought them to my city assur' 
- Prediction: 'in the mo nth kislev ix i marched to the land habhu the lands habhu and habhu together with cities in the environs of the land hatti i brought their booty possessions and possessions to my city assur'

# Models

## [akk-111m](https://huggingface.co/Thalesian/akk-111m)

This model was trained from scratch on the [Akkademia dataset](https://github.com/gaigutherz/Akkademia).
It achieves the following categorical cross-entropy results on the evaluation set (512 tokens):
- Loss: 0.0753

Cuneiform -> English Bleu score
- 500 tokens: 38.91
- 100 tokens: 43.13

Transliterated -> English Bleu score
- 500 tokens: 37.02
- 100 tokens: 41.67

Cuneiform -> Transliteration Bleu score
- 500 tokens: 94.31
- 100 tokens: 94.36

Cuneiform -> Transliteration Accuracy
- 100 tokens: 50% (note a single missed character significantly decreases accuracy in seq2seq models, see Bleu score for positional flexibility)

### Model description

This is an instruct model, meaning it is capable of multiple tasks. It is intended for primarily translation + transliteration, but it can also be used for reverse translation as well. 

#### Translation Instructions: 
- "Translate Akkadian cuneiform to English" + cuneiform signs -> English
- "Translate Akkadian simple transliteration to English" + simple transliteration -> English
- "Translate Akkadian grouped transliteration to English" + transliteration with spacial symbols -> English
- "Translate English to Akkadian cuneiform" + English -> Akkadian cuneiform signs
- "Translate English to simple Akkadian transliteration" + English -> Akkadian simple transliteration with no special symbols
- "Translate English to grouped Akkadian transliteration" + English -> Akkadian transliteration grouped into words with special symbols

#### Transliteration Instructions:
- "Transliterate Akkadian cuneiform to simple Latin Characters" + cuneiform signs -> transliteration with no special symbols
- "Transliterate Akkadian cuneiform to grouped Latin characters" + cuneiform signs -> transliteration with special symbols/subscripts
- "Group Akkadian transliteration into likely words" + simple transliteration -> transliteration with special symbols/subscripts


### Intended uses & limitations

This model is designed to facilitate the translation/transliteration of Akkadian cuneiform. It may have limited facility in the reverse (e.g. translate English to Akkadian cuneiform) but these use cases are untested. 

### Training and evaluation data

Data was used from the [Akkademia project](https://github.com/gaigutherz/Akkademia), previously published in [PNAS Nexus](https://academic.oup.com/pnasnexus/article/2/5/pgad096/7147349). More information on the training data, as well as the test and validation splits, can be found on both the GitHub and published methodology. 

### Training procedure

Because of the unequal distribution of data (many short sequences + long sequences) data was trained with different padded lengths: 
An initial few epochs with a max length of 56 tokens
A follow-up series of epochs at 128 tokens
The same for 256 tokens
A final 5 epochs for 512 tokens

The origional t5-small model had its tokens and embedding layers expanded by the additional linguistic data. Cuneiform symbols were split by spaces to be fed directly into the model, following the instructions detailed above. 

#### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4e-05
- train_batch_size: 9
- eval_batch_size: 9
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5
- mixed_precision_training: Native AMP

#### Training results 512 tokens

| Training Loss | Epoch  | Step   | Validation Loss |
|:-------------:|:------:|:------:|:---------------:|
| 0.0721        | 0.5524 | 25000  | 0.0781          |
| 0.0666        | 1.1049 | 50000  | 0.0772          |
| 0.0642        | 1.6573 | 75000  | 0.0764          |
| 0.0645        | 2.2097 | 100000 | 0.0759          |
| 0.0634        | 2.7621 | 125000 | 0.0755          |
| 0.0576        | 3.3146 | 150000 | 0.0756          |
| 0.0634        | 3.8670 | 175000 | 0.0755          |
| 0.0604        | 4.4194 | 200000 | 0.0754          |
| 0.0627        | 4.9718 | 225000 | 0.0753          |


#### Framework versions

- Transformers 4.41.2
- Pytorch 2.3.1+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
