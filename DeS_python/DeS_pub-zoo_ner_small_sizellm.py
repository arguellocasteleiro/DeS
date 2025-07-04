# -*- coding: utf-8 -*-
"""pub-zoo_NER_small-sizeLLM.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1yH...
"""

#0 install libraries & load libraries
!pip install transformers accelerate
from transformers import pipeline

#0 preliminaries: input data excerptArr[5]
text_paragraph = """
PCR is used to confirm the presence of Coxiella burnetii, typically following the identification of
suspicious acid-fast bodies in Modified Ziehl-Neelsen (MZN)-stained smears of placentae (or foetal samples).
Confirmation of Q fever as a cause of fetopathy requires histopathology and
immunohistochemistry of placental tissue, in addition to a positive PCR result.
In each case when C. burnetii is detected by PCR, public health colleagues are
informed of the incident and the zoonotic potential of this organism is highlighted to the farmer and
private veterinary surgeon, with the provision of an advisory sheet about Q fever.
"""

#1 instantiate pipeline
pipe = pipeline(task="token-classification", model="Clinical-AI-Apollo/Medical-NER",aggregation_strategy="simple")

#2 input data
text_data = text_paragraph

#3 show pipeline output
print(pipe(text_data))