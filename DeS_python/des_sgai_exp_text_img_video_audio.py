# -*- coding: utf-8 -*-
"""DeS_SGAI-exp_text-img-video-audio.ipynb

Automatically generated by Colab.

"""

# ___________________________ Requirements
### install libraries & load libraries
!pip install transformers accelerate diffusers
!pip install pillow
!pip install datasets

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import requests
from PIL import Image
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from datasets import load_dataset, Audio

# ______________________________________________________ Text
# ___________________________
###  Text Classification: Sentiment Analysis (SA)

###--- --- --- Requirements
### install libraries & load libraries
# !pip install transformers
# from transformers import pipeline

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="sentiment-analysis", model='siebert/sentiment-roberta-large-english')

#2 input data
text_data = ["I love you", "I hate you"] ### a vector (one-dimensional array) with 2 string values

#3 show pipeline output
print(pipe(text_data))

# ___________________________
### Token Classification: Named-entity recognition (NER)

###--- --- --- Requirements
### install libraries & load libraries
# !pip install transformers
# from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

#--- --- --- preliminaries
### select model
model_path = "d4data/biomedical-ner-all"

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="ner", model= AutoModelForTokenClassification.from_pretrained(model_path), tokenizer= AutoTokenizer.from_pretrained(model_path))

#2 input data
text_data = "I have a mild headache"

#3 show pipeline output
print(pipe(text_data))

# ______________________________________________________ Image / Video
# ___________________________
### zero-shot-image-classification

###--- --- --- Requirements
### install libraries & load libraries
# !pip install transformers
# !pip install pillow

# import requests
# from transformers import pipeline
# from PIL import Image

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="zero-shot-image-classification", model="openai/clip-vit-large-patch14")

#2 input data
img_data = Image.open(requests.get("https://artistproject.metmuseum.org/-/media/images/homepage/locations/fifth-avenue/fifth_ave_mobile_darkened2_final.jpg", stream=True).raw)

#3 show pipeline output
print(pipe(img_data, candidate_labels=["building", "column", "car", "animal", "cat", "flowers"]))

# ___________________________
### text to video

###--- --- --- Requirements
### install libraries & load libraries
# !pip install transformers accelerate diffusers

# import torch
# from diffusers import DiffusionPipeline
# from diffusers.utils import export_to_video

#--- --- --- preliminaries
###--- access to Google drive
from google.colab import drive
drive.mount('/content/drive')
###--- path to save the video generated
v_path = "/content/drive/MyDrive/newTut101_files/DarthVader_ex.mp4"

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16").to("cuda")

#2 input data
text_prompt = "Darth Vader surfing a wave"

#3 pipeline output
video_frames = pipe(text_prompt, num_frames=32).frames[0]

###3 extra line: save the video
video_path = export_to_video(video_frames, fps=10, output_video_path=v_path)

# ______________________________________________________ Audio
# ___________________________
### Using existing datasets:
### loading and playing audio files

###--- --- --- Requirements
### install libraries & load libraries
# !pip install datasets
# from datasets import load_dataset, Audio

#--- --- --- preliminaries
### load audio file
### first file from train dataset of Minds-14: e-banking speech dataset Intent Classification
minds14 = load_dataset("PolyAI/minds14", name="en-AU", split="train[:1]")
audio_files = minds14.cast_column("audio", Audio(sampling_rate=16_000))
audio_sample = audio_files[0]

### play audio file
from IPython.display import Audio
Audio(audio_sample["audio"]["array"], rate=audio_sample["audio"]["sampling_rate"])

# ___________________________
###  Audio Classification: Keyword Spotting (KWS)

###--- --- --- Requirements
### install libraries & load libraries
# !pip install transformers datasets
# from transformers import pipeline
# from datasets import load_dataset, Audio

#--- --- --- preliminaries
### load audio file [ use the same file for earlier ]
### first file from train dataset of Minds-14: e-banking speech dataset Intent Classification


#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14")

#2 input data
audio_data = audio_sample["audio"]["array"]

#3 show pipeline output
print(pipe(audio_data))

# ___________________________
###  Audio Classification: Language Identification (LID)

###--- --- --- Requirements
### install libraries & load libraries
# !pip install transformers datasets
# from transformers import pipeline
# from datasets import load_dataset, Audio

#--- --- --- preliminaries
### load audio file [ use the same file for earlier ]
### first file from train dataset of Minds-14: e-banking speech dataset Intent Classification

#--- --- --- Transformers pipeline: 3 lines of code
#1 instantiate pipeline
pipe = pipeline(task="audio-classification", model="facebook/mms-lid-4017")

#2 input data
audio_data = audio_sample["audio"]["array"]

#3 show pipeline output
print(pipe(audio_data))