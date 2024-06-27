# ______________________________________________________
### [ ex5 ] Sequence of words: Predicting a "masked" word
# ______________________________________________________
###--- --- --- By default Transformers pipeline
### pipeline(task='fill-mask', model=model_path)

# The LLM is given a sequence of words. The goal is to predict a “masked” word
# The LLM is bert-base-uncased available at 
# https://huggingface.co/bert-base-uncased

# BERT was trained with the masked language modeling and next sentence prediction objectives.
# Example: "Rome is the [MASK] of Italy."
# ___________________________
#--- --- --- install and load libraries
!pip install transformers
from transformers import pipeline

#--- --- --- Transformers pipeline: 3 lines of code
pipe = pipeline(task='fill-mask', model='bert-base-uncased')
text_data = "Rome is the [MASK] of Italy."
print(pipe(text_data))

# ___________________________ output
### [{'score': 0.9928465485572815, 'token': 3007, 'token_str': 'capital', 'sequence': 'rome is the capital of italy.'}, {'score': 0.0012917392887175083, 'token': 2415, 'token_str': 'center', 'sequence': 'rome is the center of italy.'}, {'score': 0.0009973271517083049, 'token': 14508, 'token_str': 'birthplace', 'sequence': 'rome is the birthplace of italy.'}, {'score': 0.0008744205115363002, 'token': 2540, 'token_str': 'heart', 'sequence': 'rome is the heart of italy.'}, {'score': 0.0006965713109821081, 'token': 2803, 'token_str': 'centre', 'sequence': 'rome is the centre of italy.'}]
