from pathlib import Path
import pandas as pd
import random

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "expanded_design_labels.csv"
OUTPUT = ROOT / "data" / "processed"

SEED = 42
random.seed(SEED)

df = pd.read_csv(INPUT)

designs = sorted(df["design_id"].unique())

random.shuffle(designs)

n = len(designs)

n_train = int(n * 0.70)
n_val = int(n * 0.15)

train_designs = designs[:n_train]
val_designs = designs[n_train:n_train + n_val]
test_designs = designs[n_train + n_val:]

df["split"] = df["design_id"].apply(
    lambda x:
        "train" if x in train_designs
        else "val" if x in val_designs
        else "test"
)
output_file = OUTPUT / "manifest.csv"

df.to_csv(output_file, index=False)

print("================================")
print("DATASET SPLIT CREATED")
print("================================")

print("Total images:", len(df))
print("Total designs:", len(designs))

print()
print("Train designs:", len(train_designs))
print("Validation designs:", len(val_designs))
print("Test designs:", len(test_designs))

print()
print("Images per split:")
print(df["split"].value_counts())

print()
print("Saved to:")
print(output_file)