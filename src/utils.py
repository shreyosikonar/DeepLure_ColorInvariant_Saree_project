import os, random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(path, model, optimizer=None, epoch=None, metrics=None, config=None):
    obj = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "config": config}
    if optimizer is not None: obj["optimizer"] = optimizer.state_dict()
    torch.save(obj, path)
