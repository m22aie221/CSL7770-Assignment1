import random


def run_audit(dataset):

    required_fields = ["gender", "age", "accent"]
    missing = [f for f in required_fields if f not in dataset.features]

    print("Missing fields:", missing)
    print("Documentation Debt:", len(missing) / len(required_fields))

    random.seed(42)

    def assign(example):
        example["gender"] = random.choice(["male", "female"])
        example["age"] = random.choice(["young", "old"])
        return example

    dataset = dataset.map(assign)

    return dataset