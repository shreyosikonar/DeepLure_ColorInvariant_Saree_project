import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "manifest.csv"
HANDLOOM = ROOT / "data" / "RAW" / "handloom_sarees-20261005T114814Z-1-001"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
P, K = 4, 4
EPOCHS = 20
LR = 1e-4
EMBED_DIM = 256
TEMP = 0.07

df = pd.read_csv(MANIFEST)

def locate(name):
    hits = list(HANDLOOM.rglob(name))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

class SareeDataset(Dataset):
    def __init__(self, frame, train):
        self.frame = frame.reset_index(drop=True)
        self.train = train
        self.train_tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.35, 0.35, 0.65, 0.15),
            transforms.RandomGrayscale(p=0.35),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.eval_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        img = Image.open(locate(row.image_name)).convert("RGB")
        if self.train:
            return torch.stack([self.train_tf(img), self.train_tf(img)]), int(row.design_id_num)
        return self.eval_tf(img), int(row.design_id_num)

class PKSampler(Sampler):
    def __init__(self, frame, p=4, k=4):
        self.p, self.k = p, k
        self.groups = {}
        for i, d in enumerate(frame.design_id_num):
            self.groups.setdefault(int(d), []).append(i)
        self.designs = list(self.groups)

    def __iter__(self):
        ds = self.designs.copy()
        random.shuffle(ds)
        batches = []
        for s in range(0, len(ds), self.p):
            chosen = ds[s:s+P]
            if len(chosen) < 2:
                continue
            batch = []
            for d in chosen:
                batch.extend(random.choices(self.groups[d], k=self.k))
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return max(1, len(self.designs)//self.p)

def supcon_loss(features, labels, temperature=0.07):
    b, v, d = features.shape
    z = F.normalize(features, dim=-1).reshape(b*v, d)
    labs = labels.repeat_interleave(v)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(b*v, device=z.device)
    exp_logits = torch.exp(logits) * (1-eye)
    mask = (labs[:,None] == labs[None,:]).float() * (1-eye)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    positives = mask.sum(1)
    valid = positives > 0
    if not valid.any():
        return logits.sum() * 0
    return -((mask * log_prob).sum(1)[valid] / positives[valid]).mean()

design_map = {d:i for i,d in enumerate(sorted(df.design_id.unique()))}
df["design_id_num"] = df.design_id.map(design_map)

train_df = df[df.split == "train"].copy()
val_df = df[df.split == "val"].copy()

train_loader = DataLoader(
    SareeDataset(train_df, True),
    batch_sampler=PKSampler(train_df, P, K),
    num_workers=0
)
val_loader = DataLoader(
    SareeDataset(val_df, False),
    batch_size=16,
    shuffle=False,
    num_workers=0
)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(512, EMBED_DIM)
)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

print("Device:", DEVICE)
print("Train images:", len(train_df))
print("Validation images:", len(val_df))
print("Train designs:", train_df.design_id.nunique())
print("Validation designs:", val_df.design_id.nunique())

best = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    losses = []

    for views, labels in train_loader:
        b, v, c, h, w = views.shape
        views = views.to(DEVICE)
        labels = labels.to(DEVICE)

        emb = model(views.reshape(b*v, c, h, w)).reshape(b, v, EMBED_DIM)
        loss = supcon_loss(emb, labels, TEMP)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    train_loss = float(np.mean(losses)) if losses else float("nan")
    print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.4f}")

    if train_loss < best:
        best = train_loss
        torch.save(
            {
                "model_state": model.state_dict(),
                "embedding_dim": EMBED_DIM,
                "design_map": design_map,
                "epoch": epoch,
            },
            OUT / "color_invariant_resnet18_best.pt",
        )

print("Training complete.")
print("Checkpoint:", OUT / "color_invariant_resnet18_best.pt")
