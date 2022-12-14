"""
Downloads OG sounds from US and BR only.
And convert .mp3 files to .wav files.
"""
import os
import requests
import subprocess
import pandas as pd


def download_audio(save_dir, audio_url):
    """_summary_

    Args:
        save_dir (str): directory where audio will be temporarily saved to.
        audio_url (str): URL of original sound

    Returns:
        str: filepath to the downloaded audio
    """
    filename = os.path.join(save_dir, os.path.basename(audio_url))
    if not os.path.exists(filename):
        response = requests.get(audio_url)
        with open(filename, "wb") as f:
            f.write(response.content)
    print("Downloaded: ", filename)
    return filename


df = pd.read_csv("music_data.csv")
df = df[df["COUNTRY"].isin(["US", "BR"])]
df = df.reset_index(drop=True)

save_dir = "../fixtures/downloaded_audio"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for j in range(len(df)):
    audio_url = df.loc[j, "SONG_URL"]
    filename = download_audio(save_dir, audio_url)
    wav_filename = os.path.join(
        save_dir, "{}.wav".format(os.path.basename(audio_url).split(".")[0])
    )
    subprocess.call(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            filename,
            wav_filename,
            "-y",
        ]
    )

    # remove mp3 files as we do not need them anymore
    os.remove(filename)
