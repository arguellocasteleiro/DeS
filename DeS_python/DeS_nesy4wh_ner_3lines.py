# -*- coding: utf-8 -*-
"""NeSy4WH_NER_3lines.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1S...
"""

#--- --- --- install and load libraries
!pip install transformers
from transformers import pipeline

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')

#2 input data
text_data = "mood changes (common) Irritability and mood swings, which range from sadness and crying for no reason"

#3 show pipeline output
print(pipe(text_data))

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="token-classification", model="d4data/biomedical-ner-all", aggregation_strategy='simple')

#2 input data
text_data = "mood changes (common) Irritability and mood swings, which range from sadness and crying for no reason"

#3 show pipeline output
print(pipe(text_data))