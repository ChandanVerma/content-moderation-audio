"""
Use spleeter-gpu (https://github.com/deezer/spleeter)
to split each audio track into speech and accompaniement audio signals.

This is used to create pseudo, noisy labels for training a singing detection
and speech quality CNN model later.
"""
import os
import pandas as pd
import subprocess

save_dir = "../fixtures/downloaded_audio"
files = os.listdir(save_dir)

for f in files:
    audio_file = os.path.join(save_dir, f)

    subprocess.call(
        ["spleeter", "separate", "-o", "../fixtures/output", "-i", audio_file]
    )
