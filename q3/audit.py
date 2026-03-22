import os
from datasets import load_from_disk
import matplotlib.pyplot as plt
from collections import Counter

DATA_PATH = "data/librispeech_asr"

dataset = load_from_disk(DATA_PATH)

# ---- Documentation Debt ----
required_fields = ["gender", "age", "accent"]
missing = [f for f in required_fields if f not in dataset.features]

print("Missing fields:", missing)
print("Documentation Debt Score:", len(missing)/len(required_fields))

# ---- Synthetic Demographics ----
import random

def assign_demo(example):
    example["gender"] = random.choice(["male", "female"])
    example["age"] = random.choice(["young", "old"])
    return example

dataset = dataset.map(assign_demo)

# ---- Distribution ----
gender_counts = Counter(dataset["gender"])
age_counts = Counter(dataset["age"])

print("Gender:", gender_counts)
print("Age:", age_counts)

# ---- Plot ----
plt.figure()
plt.bar(gender_counts.keys(), gender_counts.values())
plt.title("Gender Distribution")
plt.savefig("audit_gender.png")

plt.figure()
plt.bar(age_counts.keys(), age_counts.values())
plt.title("Age Distribution")
plt.savefig("audit_age.png")