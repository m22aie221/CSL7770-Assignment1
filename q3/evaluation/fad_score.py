import numpy as np
import librosa
from scipy.linalg import sqrtm


def extract_mfcc_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return np.mean(mfcc, axis=1)


def compute_fad_score(real_audio, fake_audio, sr):
    real_feat = extract_mfcc_features(real_audio, sr)
    fake_feat = extract_mfcc_features(fake_audio, sr)

    real_feat = np.expand_dims(real_feat, axis=0)
    fake_feat = np.expand_dims(fake_feat, axis=0)

    mu1 = np.mean(real_feat, axis=0)
    mu2 = np.mean(fake_feat, axis=0)

    sigma1 = np.cov(real_feat, rowvar=False) + np.eye(real_feat.shape[1]) * 1e-6
    sigma2 = np.cov(fake_feat, rowvar=False) + np.eye(fake_feat.shape[1]) * 1e-6

    covmean = sqrtm(sigma1 @ sigma2)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = np.linalg.norm(mu1 - mu2) + np.trace(sigma1 + sigma2 - 2 * covmean)

    return float(fad)