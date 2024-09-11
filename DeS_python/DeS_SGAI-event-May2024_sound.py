# -*- coding: utf-8 -*-
"""DeS_SGAI-event-May2024_sound.ipynb

Automatically generated by Colab.

# ___________________________
### Using existing datasets:
### loading and playing audio files
# ___________________________

#--- --- --- install and load libraries
!pip install datasets
from datasets import load_dataset, Audio

#--- --- --- load audio file
### first file from train dataset of Minds-14: e-banking speech dataset Intent Classification
minds14 = load_dataset("PolyAI/minds14", name="en-AU", split="train[:1]")
audio_files = minds14.cast_column("audio", Audio(sampling_rate=16_000))
audio_sample = audio_files[0]

#--- --- --- play audio file
from IPython.display import Audio
Audio(audio_sample["audio"]["array"], rate=audio_sample["audio"]["sampling_rate"])

# ___________________________

# ___________________________
###  Audio Classification: Keyword Spotting (KWS)
# ___________________________

#--- --- --- install and load libraries
!pip install transformers datasets
from transformers import pipeline
from datasets import load_dataset, Audio

#--- --- --- preliminaries
### first file from train dataset of Minds-14: e-banking speech dataset Intent Classification
minds14 = load_dataset("PolyAI/minds14", name="en-AU", split="train[:1]")
audio_files = minds14.cast_column("audio", Audio(sampling_rate=16_000))
audio_sample = audio_files[0]

#--- --- --- Transformers pipeline: 3 lines of code

pipe = pipeline(task="audio-classification", model="anton-l/xtreme_s_xlsr_300m_minds14")

audio_data = audio_sample["audio"]["array"]

print(pipe(audio_data))

# ___________________________

# ___________________________
###  Audio Classification: Language Identification (LID)
# ___________________________

#--- --- --- install and load libraries
!pip install transformers datasets
from transformers import pipeline
from datasets import load_dataset, Audio

#--- --- --- preliminaries
### first file from train dataset of Minds-14: e-banking speech dataset Intent Classification
minds14 = load_dataset("PolyAI/minds14", name="en-AU", split="train[:1]")
audio_files = minds14.cast_column("audio", Audio(sampling_rate=16_000))
audio_sample = audio_files[0]

#--- --- --- Transformers pipeline: 3 lines of code

pipe = pipeline(task="audio-classification", model="facebook/mms-lid-4017")

audio_data = audio_sample["audio"]["array"]

print(pipe(audio_data))
# ___________________________