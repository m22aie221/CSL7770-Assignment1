import librosa
import numpy as np

def pitch_shift(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def speed_change(audio, speed=1.1):
    return librosa.effects.time_stretch(audio, speed)

def privacy_transform(audio, sr):
    audio = pitch_shift(audio, sr)
    audio = add_noise(audio)
    audio = speed_change(audio)
    return audio