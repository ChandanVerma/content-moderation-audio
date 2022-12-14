import os
import torchaudio
import numpy as np
import pandas as pd
import torchaudio.transforms as T
from pyannote.audio import Pipeline
from sklearn.model_selection import train_test_split
from tqdm import trange

# voice detection model
voice_detect_pipeline = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection"
)

# Read data
df = pd.read_csv("../music_data.csv")
df = df[df["COUNTRY"].isin(["US", "BR"])]
df = df.reset_index(drop=True)

# Prepare pseudo labels
print("Preparing pseudo labels...")
labels = []
wavfiles = []
for j in trange(len(df)):
    audio_url = df.loc[j, "SONG_URL"]
    spleeter_filename = os.path.join(
        "../../fixtures/output",
        os.path.basename(audio_url).split(".")[0],
        "accompaniment.wav",
    )
    wav_filename = os.path.join(
        "../../fixtures/downloaded_audio",
        "{}.wav".format(os.path.basename(audio_url).split(".")[0]),
    )
    y, sample_rate = torchaudio.load(spleeter_filename)
    resampler = T.Resample(sample_rate, 16000, dtype=y.dtype)
    resampled_waveform = resampler(y)

    voice_detect_output = voice_detect_pipeline(
        {"waveform": resampled_waveform, "sample_rate": 16000}
    )
    if len(voice_detect_output) > 0:
        wavfiles.append(wav_filename)
        y = resampled_waveform.numpy()

        if np.mean(abs(y)) <= 0.02:
            labels.append(1)  # speech
        else:
            labels.append(0)  # music

df["label"] = labels
df["wavfile"] = wavfiles

# Train test split
X_train, X_test, _, _ = train_test_split(
    np.arange(len(df)),
    np.arange(len(df)),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].values,
)

df_train = df.iloc[X_train].reset_index(drop=True)
df_test = df.iloc[X_test].reset_index(drop=True)
df_train.to_csv("../singing_detection_model/train.csv", index=False)
print("Saved: ", "../singing_detection_model/train.csv")
df_test.to_csv("../singing_detection_model/valid.csv", index=False)
print("Saved: ", "../singing_detection_model/valid.csv")
