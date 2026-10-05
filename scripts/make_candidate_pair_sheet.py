from pathlib import Path
from PIL import Image, ImageDraw
import csv
import math


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HANDLOOM_DIR = (
    PROJECT_ROOT
    / "data"
    / "RAW"
    / "handloom_sarees-20261005T114814Z-1-001"
)

INPUT_FILE = PROJECT_ROOT / "outputs" / "candidate_pairs.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "candidate_pair_sheet.jpg"


# ============================================================
# SETTINGS
# ============================================================

TOP_PAIRS = 30

IMAGE_W = 300
IMAGE_H = 250

LABEL_H = 45

PAIR_W = IMAGE_W * 2
PAIR_H = IMAGE_H + LABEL_H

COLS = 1
ROWS = TOP_PAIRS


# ============================================================
# READ CANDIDATE PAIRS
# ============================================================

pairs = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    for row in reader:

        pairs.append(row)

pairs = pairs[:TOP_PAIRS]

print("Pairs loaded:", len(pairs))
# ============================================================
# CREATE SHEET
# ============================================================

sheet = Image.new(
    "RGB",
    (PAIR_W, ROWS * PAIR_H),
    "white"
)

draw = ImageDraw.Draw(sheet)


# ============================================================
# CREATE PAIR IMAGES
# ============================================================

for i, pair in enumerate(pairs):
    image1_name = pair["image_1"]
    image2_name = pair["image_2"]

    similarity = float(pair["similarity"])

    image1_paths = list(HANDLOOM_DIR.rglob(image1_name))
    image2_paths = list(HANDLOOM_DIR.rglob(image2_name))

    if not image1_paths or not image2_paths:
        print("Could not find:", image1_name, image2_name)
        continue




    path1 = image1_paths[0]
    path2 = image2_paths[0]

    try:

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        img1.thumbnail((IMAGE_W - 10, IMAGE_H - 10))
        img2.thumbnail((IMAGE_W - 10, IMAGE_H - 10))

        y = i * PAIR_H

        # White background for each pair
        draw.rectangle(
            [0, y, PAIR_W, y + PAIR_H],
            fill="white"
        )

        # Center first image
        x1 = (IMAGE_W - img1.width) // 2

        sheet.paste(
            img1,
            (x1, y + 5)
        )

        # Center second image
        x2 = IMAGE_W + (IMAGE_W - img2.width) // 2

        sheet.paste(
            img2,
            (x2, y + 5)
        )

        # Labels
        draw.text(
            (10, y + IMAGE_H + 5),
            f"{i + 1}. {image1_name}",
            fill="black"
        )

        draw.text(
            (IMAGE_W + 10, y + IMAGE_H + 5),
            f"{image2_name}   similarity={similarity:.4f}",
            fill="black"
        )

    except Exception as e:

        print("Error processing pair:")
        print(image1_name, image2_name)
        print(e)


# ============================================================
# SAVE
# ============================================================

sheet.save(
    OUTPUT_FILE,
    quality=90
)

print()
print("Candidate pair sheet created:")
print(OUTPUT_FILE)