### Multilingual_CLIP:
Code from https://github.com/FreddeFrallan/Multilingual-CLIP. Modified the embedding dimension to
match the dimension for mBERT for comparison.


### Multilingual tasks to be tested on:
* Enlgish-Chines/Chinese-English
* German-English/English-German
* Russian-English/English-Russian

### Dataset used:
WMT 19 dataset from huggingface

### Notes:
* Very initial experiments:
    * The pre-training for mBERT and mCLIP is not using the same dataset
    * cannot really do the vector multiplication because the vector length is not aligned