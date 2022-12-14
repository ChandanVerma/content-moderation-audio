# Audio Moderation

## Requirements
- python 3.8
- Install spleeter from https://github.com/deezer/spleeter
- Needs 1 GPU
- `pip install -r requirements.txt`

## To replicate the trained models
Prepare the data:
```
cd training
python 00_download_audio_data.py
python 01_use_spleeter.py
```
Train singing detection model. The objective is to classify if an audio is someone singing or talking.
```
cd singing_detection_model
python 03_create_dataset.py
python 04_train_model.py
```
Train speech quality model. The objective is to classiy if a speech quality is good or not. We assume if the speech is too soft/ intelligible, the quality is poor.
```
cd training
python 05_create_dataset.py
python 06_train_model.py
```
## To perform inference on 2 audio examples
```
cd src
python moderate_audio.py
```
