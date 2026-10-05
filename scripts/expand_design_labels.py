from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

pairs_file = ROOT / "outputs" / "candidate_pairs.csv"
old_labels = ROOT / "data" / "processed" / "initial_design_labels.csv"
new_labels = ROOT / "data" / "processed" / "expanded_design_labels.csv"

pairs = pd.read_csv(pairs_file)

# Use the 60 strongest candidate pairs
pairs = pairs[pairs["similarity"] >= 0.93].head(60)

# Start with our existing verified labels
labels = pd.read_csv(old_labels)

# Existing image -> design mapping
mapping = dict(zip(labels["image_name"], labels["design_id"]))

next_id = len(set(mapping.values())) + 1

def get_design(image):
    return mapping.get(image)
for _, row in pairs.iterrows():

    a = row["image_1"]
    b = row["image_2"]

    da = get_design(a)
    db = get_design(b)

    if da and db:
        continue

    if da:
        mapping[b] = da

    elif db:
        mapping[a] = db

    else:
        design = f"design_{next_id:03d}"
        next_id += 1

        mapping[a] = design
        mapping[b] = design

# Save
result = pd.DataFrame(
    [
        {"image_name": image, "design_id": design}
        for image, design in sorted(mapping.items())
    ]
)

result.to_csv(new_labels, index=False)

print("================================")
print("EXPANDED LABELS CREATED")
print("================================")
print("Images labelled:", len(result))
print("Design groups:", result["design_id"].nunique())
print("Saved to:", new_labels)