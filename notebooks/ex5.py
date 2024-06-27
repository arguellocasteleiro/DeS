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
