import random
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

class SareeDataset(Dataset):
    def __init__(self, csv_path, train=True, image_size=224):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.train = train
        self.image_size = image_size
        if train:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.8, 1.25)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.5,0.5,0.5,0.15)], p=0.8),
                transforms.RandomGrayscale(p=0.45),
                transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.15),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.15, scale=(0.02,0.12)),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((image_size, image_size)), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r.path).convert('RGB')
        return self.tf(img), int(r.label), r.path

class PKBatchSampler(Sampler):
    def __init__(self, labels, p=8, k=4, seed=42):
        self.labels = list(labels); self.p=p; self.k=k; self.seed=seed
        self.groups={}
        for i,y in enumerate(self.labels): self.groups.setdefault(y,[]).append(i)
    def __iter__(self):
        rng=random.Random(self.seed)
        groups={y:idxs[:] for y,idxs in self.groups.items()}
        for v in groups.values(): rng.shuffle(v)
        ys=list(groups); rng.shuffle(ys)
        for s in range(0, len(ys), self.p):
            chosen=ys[s:s+self.p]
            if len(chosen)<2: continue
            batch=[]
            for y in chosen:
                idxs=groups[y]
                batch += [rng.choice(idxs) for _ in range(self.k)]
            yield batch
    def __len__(self): return max(0, len(self.groups)//self.p)
