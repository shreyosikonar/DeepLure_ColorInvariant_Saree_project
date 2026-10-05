from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "candidate_pairs.csv"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUT_DIR / "initial_design_labels.csv"

with open(INPUT, "r", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))[:30]

parent = {}

def find(x):
    if x not in parent:
        parent[x] = x
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra

for p in pairs:
    union(p["image_1"], p["image_2"])

groups = {}
for image in parent:
    root = find(image)
    groups.setdefault(root, []).append(image)

rows = []
for n, group in enumerate(groups.values(), 1):
    design_id = f"design_{n:03d}"
    for image in sorted(group):
        rows.append({"image_name": image, "design_id": design_id})

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_name", "design_id"])
    writer.writeheader()
    writer.writerows(rows)

print("Initial labels created successfully")
print("Verified pairs:", len(pairs))
print("Design groups:", len(groups))
print("Images labelled:", len(rows))
print("Saved to:", OUTPUT)
