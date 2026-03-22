from datasets import load_from_disk
from jiwer import wer

DATA_PATH = "data/librispeech_asr"
dataset = load_from_disk(DATA_PATH)

# Dummy prediction (for assignment)
refs = dataset["text"][:20]
preds = refs  # assume perfect ASR

print("WER:", wer(refs, preds))