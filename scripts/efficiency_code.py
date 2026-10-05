       
from pathlib import Path
import time
import torch
import torch.nn as nn
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "outputs" / "color_invariant_resnet18_best.pt"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(512, 256)
)

checkpoint = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model = model.to(DEVICE)
model.eval()

# Parameter count
params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# FLOPs using torch profiler
x = torch.randn(1, 3, 224, 224, device=DEVICE)

flops = None
try:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True
    ) as prof:
        with torch.no_grad():
            model(x)

    flops = sum(
        event.flops
        for event in prof.key_averages()
        if event.flops is not None
    )
except Exception as e:
    print("FLOPs calculation warning:", e)

# Warm-up
with torch.no_grad():
    for _ in range(5):
        model(x)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

# Latency
runs = 30
start = time.perf_counter()

with torch.no_grad():
    for _ in range(runs):
        model(x)

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
latency_ms = (elapsed / runs) * 1000

# Embedding size
embedding_dim = 256
embedding_bytes = embedding_dim * 4
embedding_kb = embedding_bytes / 1024

print("========================================")
print("MODEL EFFICIENCY")
print("========================================")
print("Device:", DEVICE)
print("Parameters:", f"{params:,}")
print("Trainable parameters:", f"{trainable:,}")

if flops is not None:
    print("FLOPs / image:", f"{flops:,.0f}")
    print("GFLOPs / image:", f"{flops / 1e9:.3f}")
else:
    print("FLOPs / image: unavailable")

print("Embedding dimension:", embedding_dim)
print("Embedding size:", f"{embedding_kb:.2f} KB")
print("Average latency:", f"{latency_ms:.2f} ms/image")

results = {
    "device": str(DEVICE),
    "parameters": params,
    "trainable_parameters": trainable,
    "flops_per_image": flops,
    "gflops_per_image": None if flops is None else flops / 1e9,
    "embedding_dimension": embedding_dim,
    "embedding_size_kb": embedding_kb,
    "latency_ms_per_image": latency_ms
}

import json
with open(OUT / "efficiency_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print()
print("Saved:", OUT / "efficiency_results.json")
