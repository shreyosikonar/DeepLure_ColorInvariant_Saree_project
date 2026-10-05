from pathlib import Path
import csv
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HANDLOOM_DIR = (
    PROJECT_ROOT
    / "data"
    / "RAW"
    / "handloom_sarees-20261005T114814Z-1-001"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "candidate_pairs.csv"


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 3. LOAD PRETRAINED RESNET18
# ============================================================

print("Loading ResNet18...")

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Remove final classification layer
model.fc = nn.Identity()

model = model.to(device)
model.eval()


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Important:
    # Convert to grayscale so that color has less influence.
    transforms.Grayscale(num_output_channels=3),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. FIND IMAGES
# ============================================================

image_paths = sorted(HANDLOOM_DIR.rglob("*.jpg"))

print("Images found:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No JPG images found.")



# ============================================================
# 6. EXTRACT EMBEDDINGS
# ============================================================

embeddings = []

print("Extracting visual embeddings...")

with torch.no_grad():

    for i, image_path in enumerate(image_paths):

        try:
            image = Image.open(image_path).convert("RGB")

            image = transform(image)
            image = image.unsqueeze(0).to(device)

            embedding = model(image)

            # L2 normalize
            embedding = torch.nn.functional.normalize(
                embedding,
                p=2,
                dim=1
            )

            embeddings.append(
                embedding.cpu().numpy()[0]
            )

        except Exception as e:

            print("Error:", image_path.name)
            print(e)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(image_paths)}")


embeddings = np.array(embeddings)

print("Embedding shape:", embeddings.shape)


# ============================================================
# 7. COSINE SIMILARITY
# ============================================================

similarity_matrix = embeddings @ embeddings.T


# ============================================================
# 8. FIND TOP SIMILAR IMAGES
# ============================================================

TOP_K = 5

pairs = []

for i in range(len(image_paths)):

    similarities = similarity_matrix[i].copy()

    # Ignore the image itself
    similarities[i] = -1

    # Get top K similar images
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    for j in top_indices:

        # Save each pair only once
        if i < j:

            pairs.append({
                "image_1": image_paths[i].name,
                "image_2": image_paths[j].name,
                "similarity": float(similarities[j])
            })


# ============================================================
# 9. SORT BY SIMILARITY
# ============================================================

pairs.sort(
    key=lambda x: x["similarity"],
    reverse=True
)

# ============================================================
# 10. SAVE RESULTS
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_1",
            "image_2",
            "similarity"
        ]
    )

    writer.writeheader()

    writer.writerows(pairs)


print()
print("========================================")
print("Candidate pair analysis complete")
print("========================================")
print()
print("Results saved to:")
print(OUTPUT_FILE)
print()
print("Top 30 candidate pairs:")
print()

for pair in pairs[:30]:

    print(
        f"{pair['image_1']:20s} "
        f"<-> "
        f"{pair['image_2']:20s} "
        f"similarity = "
        f"{pair['similarity']:.4f}"
    )


    