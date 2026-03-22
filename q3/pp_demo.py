import os
import soundfile as sf
from datasets import load_from_disk

# Import all modules
from audit import run_audit
from privacymodule import apply_privacy_transform
from evaluation_scripts.fad_score import compute_fad_score
from evaluation_scripts.dnsmos_score import (
    load_dnsmos_model,
    compute_dnsmos
)


DATA_PATH = "data/librispeech_asr"
OUTPUT_DIR = "examples/"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():

    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = load_from_disk(DATA_PATH)

    # -----------------------------
    # Audit
    # -----------------------------
    dataset = run_audit(dataset)

    sample = dataset[0]

    audio = sample["audio"]["array"]
    sr = sample["audio"]["sampling_rate"]

    gender = sample.get("gender", "male")
    age = sample.get("age", "young")

    # -----------------------------
    # Privacy Transform
    # -----------------------------
    transformed_audio = apply_privacy_transform(audio, sr, gender, age)

    # -----------------------------
    # Save Audio
    # -----------------------------
    sf.write(os.path.join(OUTPUT_DIR, "original.wav"), audio, sr)
    sf.write(os.path.join(OUTPUT_DIR, "transformed.wav"), transformed_audio, sr)

    # -----------------------------
    # FAD Evaluation
    # -----------------------------
    fad = compute_fad_score(audio, transformed_audio, sr)
    print("FAD Score:", fad)

    # -----------------------------
    # DNSMOS Evaluation
    # -----------------------------
    session = load_dnsmos_model("dnsmos.onnx")

    dnsmos_orig = compute_dnsmos(audio, sr, session)
    dnsmos_trans = compute_dnsmos(transformed_audio, sr, session)

    print("Original DNSMOS:", dnsmos_orig)
    print("Transformed DNSMOS:", dnsmos_trans)

    print("\nPipeline executed successfully ✅")


if __name__ == "__main__":
    main()