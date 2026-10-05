from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "RAW" / "handloom_sarees-20261005T114814Z-1-001"
INPUT = ROOT / "outputs" / "candidate_pairs.csv"
OUT = ROOT / "outputs"

df = pd.read_csv(INPUT)
df = df[df["similarity"] >= 0.93].head(60)

W = 700
H = 300

for batch in range(3):
    part = df.iloc[batch * 20:(batch + 1) * 20]

    sheet = Image.new("RGB", (W, H * len(part)), "white")
    draw = ImageDraw.Draw(sheet)

    for n, (_, row) in enumerate(part.iterrows()):
        p1 = list(DATA.rglob(row["image_1"]))[0]
        p2 = list(DATA.rglob(row["image_2"]))[0]

        im1 = Image.open(p1).convert("RGB")
        im2 = Image.open(p2).convert("RGB")

        im1.thumbnail((330, 250))
        im2.thumbnail((330, 250))

        y = n * H

        sheet.paste(im1, (0, y))
        sheet.paste(im2, (350, y))

        draw.text(
            (10, y + 255),
            f'{row["image_1"]}  <->  {row["image_2"]}  similarity={row["similarity"]:.4f}',
            fill="black"
        )

    output = OUT / f"candidates_{batch + 1}.jpg"
    sheet.save(output, quality=90)

    print("Created:", output)

print("DONE")
