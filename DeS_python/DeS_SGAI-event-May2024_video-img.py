# -*- coding: utf-8 -*-
"""DeS_SGAI-event-May2024_video-img.ipynb

Automatically generated by Colab.

# ___________________________
### zero-shot-image-classification
# ___________________________

#--- --- --- install and load libraries
!pip install transformers
!pip install pillow

import requests
from transformers import pipeline
from PIL import Image

#--- --- --- Transformers pipeline: 3 lines of code

pipe = pipeline(task="zero-shot-image-classification", model="openai/clip-vit-large-patch14")

img_data = Image.open(requests.get("https://www.metmuseum.org/-/media/images/visit/met-fifth-avenue/fifthave_teaser.jpg", stream=True).raw)

print(pipe(img_data, candidate_labels=["building", "column", "car", "animal", "cat", "flowers"]))

# ___________________________

# ___________________________
### text to video
# ___________________________
#--- --- --- install and load libraries
!pip install transformers accelerate diffusers

import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

#--- --- --- preliminaries
###--- access to Google drive
from google.colab import drive
drive.mount('/content/drive')
###--- path to save the video generated
v_path = "/content/drive/MyDrive/newTut101_files/DarthVader_ex.mp4"


#--- --- --- Transformers pipeline: 3 lines of code

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16").to("cuda")

text_prompt = "Darth Vader surfing a wave"

video_frames = pipe(text_prompt, num_frames=32).frames[0]

###--- extra line: save the video
video_path = export_to_video(video_frames, fps=10, output_video_path=v_path)

# ___________________________