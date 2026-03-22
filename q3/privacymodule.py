import librosa
import numpy as np


def get_pitch_shift(gender, age):
    pitch_shift = 0

    if gender == "male":
        pitch_shift += 4
    else:
        pitch_shift -= 2

    if age == "old":
        pitch_shift += 2
    else:
        pitch_shift -= 1

    return pitch_shift


def apply_privacy_transform(audio, sr, gender, age):
    shift = get_pitch_shift(gender, age)

    transformed_audio = librosa.effects.pitch_shift(
        audio.astype(np.float32),
        sr=sr,
        n_steps=shift
    )

    return transformed_audio