from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
HANDLOOM = ROOT / "data" / "RAW" / "handloom_sarees-20261005T114814Z-1-001"
CKPT = ROOT / "outputs" / "color_invariant_resnet18_best.pt"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256

df = pd.read_csv(MANIFEST)
test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

def locate(name):
    hits = list(HANDLOOM.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(512, EMBED_DIM)
)

checkpoint = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

embeddings = []

with torch.no_grad():
    for _, row in test_df.iterrows():
        image = Image.open(locate(row["image_name"])).convert("RGB")
        x = tf(image).unsqueeze(0).to(DEVICE)
        z = F.normalize(model(x), dim=1)
        embeddings.append(z.cpu().numpy()[0])

embeddings = np.asarray(embeddings)
similarity = embeddings @ embeddings.T
n = len(test_df)

# -----------------------------
# Identification: leave-one-out
# -----------------------------
eligible = [
    i for i in range(n)
    if (test_df.iloc[i]["design_id"] == test_df["design_id"]).sum() > 1
]

recall = {}
for k in [1, 5, 10]:
    hits = 0
    total = 0

    for i in eligible:
        scores = similarity[i].copy()
        scores[i] = -np.inf
        order = np.argsort(scores)[::-1][:k]
        true_id = test_df.iloc[i]["design_id"]

        if any(test_df.iloc[j]["design_id"] == true_id for j in order):
            hits += 1
        total += 1

    recall[k] = hits / total if total else float("nan")

# -----------------------------
# Verification
# -----------------------------
y_true = []
y_score = []

for i in range(n):
    for j in range(i + 1, n):
        y_true.append(
            int(test_df.iloc[i]["design_id"] == test_df.iloc[j]["design_id"])
        )
        y_score.append(float(similarity[i, j]))

y_true = np.asarray(y_true)
y_score = np.asarray(y_score)

if len(np.unique(y_true)) == 2:
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    def tar_at_far(target_far):
        valid = tpr[fpr <= target_far]
        return float(valid.max()) if len(valid) else 0.0

    tar_1 = tar_at_far(0.01)
    tar_01 = tar_at_far(0.001)
else:
    auc = float("nan")
    eer = float("nan")
    tar_1 = float("nan")
    tar_01 = float("nan")

# -----------------------------
# Save results
# -----------------------------
results = pd.DataFrame([{
    "test_images": n,
    "test_designs": test_df["design_id"].nunique(),
    "identification_queries": len(eligible),
    "Recall@1": recall[1],
    "Recall@5": recall[5],
    "Recall@10": recall[10],
    "verification_pairs": len(y_true),
    "positive_pairs": int(y_true.sum()),
    "negative_pairs": int((1-y_true).sum()),
    "ROC_AUC": auc,
    "EER": eer,
    "TAR@FAR1%": tar_1,
    "TAR@FAR0.1%": tar_01
}])

results.to_csv(OUT / "evaluation_results.csv", index=False)

print("========================================")
print("COLOR-INVARIANT MODEL EVALUATION")
print("========================================")
print("Device:", DEVICE)
print("Test images:", n)
print("Test designs:", test_df["design_id"].nunique())
print("Identification queries:", len(eligible))
print()
print(f"Recall@1  : {recall[1]:.4f}")
print(f"Recall@5  : {recall[5]:.4f}")
print(f"Recall@10 : {recall[10]:.4f}")
print()
print("Verification pairs:", len(y_true))
print("Positive pairs:", int(y_true.sum()))
print("Negative pairs:", int((1-y_true).sum()))
print(f"ROC-AUC   : {auc:.4f}")
print(f"EER       : {eer:.4f}")
print(f"TAR@FAR1% : {tar_1:.4f}")
print(f"TAR@FAR0.1%: {tar_01:.4f}")
print()
print("Saved:", OUT / "evaluation_results.csv")
