import os
import torch
import torchvision
import numpy as np
import torch.nn.functional as F


def fill_gaps(output, max_gap=3, time_thresh=0.5):
    """Join up consecutive human speech segments if they are <= 3 seconds \
    apart. And remove speech segments if they are < 0.5 seconds long.

    Args:
        output (pyannote speaker-segmentation output): output from speech detection
        max_gap (float, optional): Threshold to join up speech consecutive segments.\
        Defaults to 3.
        time_thresh (float, optional): Threshold to remove speech segments that are too short. \
        Defaults to 0.5.

    Returns:
        list: final speech segments
    """
    prev_end, prev_start = -100, -100

    segm_new = []
    for segm in output.get_timeline().support():
        start = segm.start
        end = segm.end

        if start - prev_end > max_gap:
            segm_new.append([start, end])
        else:
            segm_new[-1][1] = end

        prev_end = end
        prev_start = start

    return [x for x in segm_new if x[1] - x[0] >= time_thresh]


def remove_utterance(output, time_thresh=0.5):
    """Remove speech segments if they are < 0.5 seconds long.

    Args:
        output (speechbrain speech detect object): output from speech detection
        time_thresh (float, optional): Threshold to remove speech segments that are too short. \
        Defaults to 0.5.

    Returns:
        list: final speech segments
    """
    segm_new = [
        [x.start, x.end]
        for x in output.get_timeline().support()
        if x.end - x.start >= time_thresh
    ]

    return segm_new


def get_melspec(
    audio_tensor,
    mel_spectrogram,
    target_size=(64, 256),
    max_num_frames=480000,
):
    """Get mel-spectrogram of audio.

    Args:
        audio_tensor (torch.tensor): torch tensor of audio signal
        mel_spectrogram (torchaudio.transforms.MelSpectrogram): Mel-spectrogram function
        target_size (tuple, optional): Mel-spectrogram final image size after resizing.\
        Defaults to (64, 256).
        max_num_frames (int, optional): Max length of audio signal.\
        Defaults to 480000 as we assume it is maximum 30 seconds audio and 16000Hz \
        30*16000=480000

    Returns:
        torch.tensor: resized mel-spectrogram of audio tensor
    """

    # padding
    if audio_tensor.shape[1] > max_num_frames:
        padded = audio_tensor[:, 0:max_num_frames]
    elif audio_tensor.shape[1] < max_num_frames:
        num_pad = max_num_frames - audio_tensor.shape[1]
        padded = F.pad(audio_tensor, (0, num_pad), mode="constant")

    melspec = mel_spectrogram(padded)
    resized_melspec = torchvision.transforms.Resize(size=target_size)(melspec)[
        None, :, :, :
    ]
    return resized_melspec


def map_pred_to_result(pred):
    """Map outputs of Detoxify model to final predictions.

    Args:
        pred (dict): Detoxify prediction outputs

    Returns:
        tuple: True if NSFW content is detected otherwise False (bool), \
        maximum probability of all predicted class labels (float)
    """
    labels = list(pred.keys())
    probs = np.array(list(pred.values())).squeeze()
    max_prob = probs[np.argmax(probs)]
    if max_prob > 0.5:
        return True, max_prob
    else:
        return False, max_prob


def transcribe_audio(
    audio_arr,
    segments_output,
    transcribe_processor,
    transcribe_model,
    device,
    rate=16000,
):
    """Transcribe audio file

    Args:
        audio_arr (np.array): numpy array of audio signal
        segments_output (pyannote speaker-segmentation outputs): speech detection outputs
        transcribe_processor (Wav2Vec2Processor): processor of transcription model
        transcribe_model (Wav2Vec2ForCTC): transcription model
        device (str): indicates cuda device e.g. "cuda:0"
        rate (int, optional): Sample rate of audio signal. Defaults to 16000.

    Returns:
        str: transcription of audio
    """
    if len(segments_output.labels()) == 1:
        segments = fill_gaps(segments_output)
    else:
        segments = remove_utterance(segments_output)

    sentences = []
    for segm in segments:
        y = audio_arr[0, int(rate * segm[0]) : int(rate * segm[1])]

        torch.cuda.empty_cache()
        with torch.no_grad():
            input_values = transcribe_processor(
                y, return_tensors="pt", sampling_rate=rate
            ).input_values
            logits = transcribe_model(input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = transcribe_processor.decode(predicted_ids[0])
        sentences.append(transcription.lower())
    sentences = ". ".join(sentences)
    return sentences


def translate_to_en(text, translate_tokenizer, translate_model, device):
    """Translate text to english.

    Args:
        text (str): text in source language
        translate_tokenizer (AutoTokenizer): tokenizer of translation model
        translate_model (AutoModelForSeq2SeqLM): translation model
        device (str): indicates cuda device e.g. "cuda:0"

    Returns:
        str: source text in english
    """
    torch.cuda.empty_cache()
    with torch.no_grad():
        batch = translate_tokenizer([text], return_tensors="pt").to(device)
        generated_ids = translate_model.generate(**batch)
        trans = translate_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
    trans = trans.lower()
    return trans


def generate_metadata():
    """Template to store outputs.

    Returns:
        dict: template to store outputs
    """
    return {
        "muted": None,
        "poor-quality": None,
        "voice-detected": None,
        "predicted_language": None,
        "singing": None,
        "speech": None,
        "to-be-moderated": None,
        "transcription": None,
        "transcription_translated": None,
    }
