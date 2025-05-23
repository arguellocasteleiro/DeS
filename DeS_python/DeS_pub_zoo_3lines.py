# -*- coding: utf-8 -*-
"""pub-zoo_3lines.ipynb

Automatically generated by Colab.
"""
# ___________________________ token classification

#--- --- --- install and load libraries
!pip install transformers accelerate

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

#0 preliminaries: input data excerptArr[5]
text_paragraph = "PCR is used to confirm the presence of Coxiella burnetii, typically following the identification of suspicious acid-fast bodies in Modified Ziehl-Neelsen (MZN)-stained smears of placentae (or foetal samples). Confirmation of Q fever as a cause of fetopathy requires histopathology and immunohistochemistry of placental tissue, in addition to a positive PCR result. In each case when C. burnetii is detected by PCR, public health colleagues are informed of the incident and the zoonotic potential of this organism is highlighted to the farmer and private veterinary surgeon, with the provision of an advisory sheet about Q fever."

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="token-classification", model="Clinical-AI-Apollo/Medical-NER", aggregation_strategy='simple')

#2 input data
text_data = text_paragraph

#3 show pipeline output
print(pipe(text_data))

# ___________________________ token classification with zero-shot

#--- --- --- install and load libraries
!pip install transformers accelerate bitsandbytes

import transformers
import torch

#0 preliminaries: prompt for input data excerptArr[5]
text_paragraph = "PCR is used to confirm the presence of Coxiella burnetii, typically following the identification of suspicious acid-fast bodies in Modified Ziehl-Neelsen (MZN)-stained smears of placentae (or foetal samples). Confirmation of Q fever as a cause of fetopathy requires histopathology and immunohistochemistry of placental tissue, in addition to a positive PCR result. In each case when C. burnetii is detected by PCR, public health colleagues are informed of the incident and the zoonotic potential of this organism is highlighted to the farmer and private veterinary surgeon, with the provision of an advisory sheet about Q fever."

text_prompt = "Identify and classify named entities related to medical information from the text below." + "\n" + "Text: " +  text_paragraph + "\n"

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipeline = transformers.pipeline(task="text-generation", model="HPAI-BSC/Llama3.1-Aloe-Beta-8B", model_kwargs={"torch_dtype": torch.bfloat16},device_map="auto",)

#2 input data
text_data = [{"role": "system", "content": "You are an expert medical assistant."},{"role": "user", "content": text_prompt },]

#3 show pipeline output
print(pipeline(pipeline.tokenizer.apply_chat_template(text_data, tokenize=False, add_generation_prompt=True),max_new_tokens=256, eos_token_id=[pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")],do_sample=True,temperature=0.6,top_p=0.9,)[0]["generated_text"][410:])

# ___________________________ 