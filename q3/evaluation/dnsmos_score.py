import numpy as np
import librosa
import onnxruntime as ort


def load_dnsmos_model(model_path="dnsmos.onnx"):
    return ort.InferenceSession(model_path)


def preprocess_audio(audio, sr):
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    target_len = 144160

    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    return audio.astype(np.float32)


def compute_dnsmos(audio, sr, session):
    audio = preprocess_audio(audio, sr)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: audio.reshape(1, -1)})

    sig, bak, ovr = output[0][0]

    return {
        "SIG": float(sig),
        "BAK": float(bak),
        "OVR": float(ovr)
    }