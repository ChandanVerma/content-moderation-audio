import os
import sys
import numpy as np
import subprocess
import librosa
import torch
import torchaudio
import regex as re
import requests
import torchaudio.transforms as T

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

from quality_detect_arch import QualityDetectNet
from singing_detect_arch import SingingDetectNet
from detoxify import Detoxify
from better_profanity import profanity
from pyannote.audio import Pipeline
from transformers import (
    Wav2Vec2Processor,
    HubertForCTC,
    Wav2Vec2ForCTC,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from utils import (
    generate_metadata,
    get_melspec,
    transcribe_audio,
    translate_to_en,
    map_pred_to_result,
)


class ModerateAudio:
    def __init__(
        self,
    ):
        """
        Load all pretrained models.
        """
        print("Loading models...")
        self.device = "cuda:0"
        self.speech_classification = SingingDetectNet().to(self.device)
        self.speech_classification.load_state_dict(
            torch.load("../model_checkpoints/singing_detect_ckpt.pth")
        )
        self.speech_classification.eval()

        self.poor_quality_classification = QualityDetectNet().to(self.device)
        self.poor_quality_classification.load_state_dict(
            torch.load("../model_checkpoints/quality_detect_ckpt.pth")
        )
        self.poor_quality_classification.eval()

        custom_badwords = ["whatsapp", "text"]
        profanity.add_censor_words(custom_badwords)
        profanity.load_censor_words(
            whitelist_words=[
                "god",
                "kkk",
                "omg",
                "lmao",
                "lmfao",
                "damn",
                "plss",
                "gai",
                "wtf",
            ]
        )
        self.en_m = Detoxify("original", device=self.device)
        self.pt_m = Detoxify("multilingual", device=self.device)

        self.voice_detect_pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection"
        )
        self.speaker_segmentation_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-segmentation"
        )

        self.en_transcribe_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        )
        self.en_transcribe_model = HubertForCTC.from_pretrained(
            "facebook/hubert-large-ls960-ft"
        ).to(self.device)

        self.pt_transcribe_processor = Wav2Vec2Processor.from_pretrained(
            "Edresson/wav2vec2-large-xlsr-coraa-portuguese"
        )
        self.pt_transcribe_model = Wav2Vec2ForCTC.from_pretrained(
            "Edresson/wav2vec2-large-xlsr-coraa-portuguese"
        ).to(self.device)

        self.translate_tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-ROMANCE-en", cache_dir="../model_checkpoints"
        )
        self.translate_model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-ROMANCE-en", cache_dir="../model_checkpoints"
        ).to(self.device)

        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            n_mels=n_mels,
            mel_scale="htk",
        )
        print("Models loaded.")

    def get_result(self, audio_url, country, save_dir="../fixtures/downloaded_audio"):
        """Get moderation result of audio track.

        Args:
            audio_url (str): URL of original sound
            country (str): country of creation e.g. "US", "BR" etc.
            save_dir (str, optional): directory where audio will be temporarily saved to. \
            Defaults to "../fixtures/downloaded_audio".

        Returns:
            dict: dictionary containing all outputs
        """
        metadata = generate_metadata()

        # Download audio file
        filename = os.path.join(save_dir, os.path.basename(audio_url))
        response = requests.get(audio_url)
        with open(filename, "wb") as f:
            f.write(response.content)
        print("Downloaded: ", filename)

        # Convert to .wav files
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

        # Remove mp3 files as we do not need them anymore
        os.remove(filename)

        # Read audio signal
        audio_tensor, sample_rate = torchaudio.load(wav_filename)
        audio_tensor = torch.mean(audio_tensor, axis=0)[None, :]

        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000, dtype=audio_tensor.dtype)
            audio_tensor = resampler(audio_tensor)
        audio_arr = audio_tensor.numpy()

        if np.sum(abs(audio_arr)) == 0:
            metadata["muted"] = True
        else:
            metadata["muted"] = False

        if not metadata["muted"]:
            voice_detect_output = self.voice_detect_pipeline(
                {"waveform": audio_tensor, "sample_rate": 16000}
            )

            if len(voice_detect_output) > 0:
                metadata["voice-detected"] = True
            else:
                metadata["voice-detected"] = False

            if metadata["voice-detected"]:
                with torch.no_grad():
                    cnn_input = get_melspec(audio_tensor, self.mel_spectrogram).to(
                        self.device
                    )
                    poor_quality_output = torch.argmax(
                        self.poor_quality_classification(cnn_input), axis=-1
                    )

                if poor_quality_output == 1:
                    metadata["poor-quality"] = True
                else:
                    metadata["poor-quality"] = False

                if not metadata["poor-quality"]:

                    if country == "US":
                        lang = "English"
                    if country == "BR":
                        lang = "Portuguese"

                    metadata["predicted_language"] = lang

                    if lang in ["English", "Portuguese"]:
                        if lang == "English":
                            lang_model = self.en_m
                            transcribe_processor = self.en_transcribe_processor
                            transcribe_model = self.en_transcribe_model
                        else:
                            lang_model = self.pt_m
                            transcribe_processor = self.pt_transcribe_processor
                            transcribe_model = self.pt_transcribe_model

                        with torch.no_grad():
                            pred = torch.argmax(
                                self.speech_classification(cnn_input), axis=-1
                            )

                        if pred == 1:
                            metadata["speech"] = True
                            metadata["singing"] = False
                        else:
                            metadata["speech"] = False
                            metadata["singing"] = True

                        segments_output = self.speaker_segmentation_pipeline(
                            {"waveform": audio_tensor, "sample_rate": 16000}
                        )
                        sentences = transcribe_audio(
                            audio_arr=audio_arr,
                            segments_output=segments_output,
                            transcribe_processor=transcribe_processor,
                            transcribe_model=transcribe_model,
                            device=self.device,
                        )
                        metadata["transcription"] = sentences

                        if len(sentences) > 0:
                            if lang == "Portuguese":
                                translated_sent = translate_to_en(
                                    sentences,
                                    self.translate_tokenizer,
                                    self.translate_model,
                                    self.device,
                                )
                                metadata["transcription_translated"] = translated_sent

                            if metadata["speech"]:
                                pred_en, prob_en = map_pred_to_result(
                                    lang_model.predict([sentences])
                                )

                                if pred_en:
                                    metadata["to-be-moderated"] = True
                                else:
                                    metadata["to-be-moderated"] = False

                            elif metadata["singing"]:
                                if lang == "Portuguese":
                                    sentences = translated_sent
                                num_profane_words = profanity.censor(sentences).count(
                                    "****"
                                ) - sentences.count("****")
                                num_words = len(re.findall(r"\w+", sentences))

                                if num_profane_words / (num_words + 1) >= 0.5:
                                    metadata["to-be-moderated"] = True
                                else:
                                    metadata["to-be-moderated"] = False
                        else:
                            # if transcription is nothing, default True
                            metadata["to-be-moderated"] = True
        return metadata


if __name__ == "__main__":
    engine = ModerateAudio()
    # test US sample
    audio_url = "https://lomotif-prod.s3.amazonaws.com/music/original-music/2022/6/28/b6b940e9956fc9b5d82385c2521af284b9525cc9a4bb7afbdb43afc9ccc7859f.mp3"
    country = "US"
    output = engine.get_result(audio_url=audio_url, country=country)
    print("US sample output: \n", output)

    # test BR sample
    audio_url = "https://lomotif-prod.s3.amazonaws.com/music/original-music/2022/6/23/de8ea1e4214e3e364ea09394e74f06fd64f0f9d65d8874c87ac22c8b98f69a30.mp3"
    country = "BR"
    output = engine.get_result(audio_url=audio_url, country=country)
    print("BR sample output: \n", output)
