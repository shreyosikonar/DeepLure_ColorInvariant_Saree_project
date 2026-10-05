from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Handloom dataset
HANDLOOM_DIR = (
    PROJECT_ROOT
    / "data"
    / "RAW"
    / "handloom_sarees-20261005T114814Z-1-001"
)

# Find all JPG images recursively
images = sorted(HANDLOOM_DIR.rglob("*.jpg"))

print("Images found:", len(images))

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Contact sheet settings
THUMB_W = 180
THUMB_H = 180
LABEL_H = 35
COLS = 5

ROWS = math.ceil(len(images) / COLS)

sheet = Image.new(
    "RGB",
    (COLS * THUMB_W, ROWS * (THUMB_H + LABEL_H)),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, image_path in enumerate(images):

    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((THUMB_W - 10, THUMB_H - 10))

        x = (i % COLS) * THUMB_W
        y = (i // COLS) * (THUMB_H + LABEL_H)

        # Center image
        paste_x = x + (THUMB_W - img.width) // 2
        paste_y = y + (THUMB_H - img.height) // 2

        sheet.paste(img, (paste_x, paste_y))

        # Filename
        filename = image_path.name

        # Remove extension for shorter label
        filename = Path(filename).stem

        draw.text(
            (x + 5, y + THUMB_H + 5),
            filename,
            fill="black"
        )

    except Exception as e:
        print("Could not process:", image_path)
        print("Error:", e)

output_file = OUTPUT_DIR / "handloom_contact_sheet.jpg"

sheet.save(
    output_file,
    quality=90
)

print()
print("Contact sheet created:")

print(output_file)