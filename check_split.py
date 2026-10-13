# check_split.py
import numpy as np
from PIL import Image
import imagehash
from torchvision import datasets
from sklearn.model_selection import StratifiedGroupKFold

from train import DATA_DIR, VAL_SPLIT, SEED, hash_groups

ds = datasets.ImageFolder(DATA_DIR)
paths = [p for p, _ in ds.samples]
targets = np.array(ds.targets)
groups = hash_groups(paths)            # uses the cache, so it's fast

sgkf = StratifiedGroupKFold(n_splits=round(1 / VAL_SPLIT), shuffle=True, random_state=SEED)
train_idx, val_idx = next(sgkf.split(np.arange(len(paths)), targets, groups))

bits = np.array([imagehash.phash(Image.open(p)).hash.flatten() for p in paths])
train_bits = bits[train_idx]
min_d = np.array([(train_bits != bits[i]).sum(axis=1).min() for i in val_idx])

for t in [6, 8, 10, 12, 16]:
    print(f"nearest train image within distance {t:2d}: {(min_d <= t).mean():.1%} of val")



train_targets = targets[train_idx]
same_d, other_d = [], []
for i in val_idx:
    d = (train_bits != bits[i]).sum(axis=1)
    same = train_targets == targets[i]
    same_d.append(d[same].min())
    other_d.append(d[~same].min())
same_d, other_d = np.array(same_d), np.array(other_d)

print(f"\nmedian nearest SAME-chord train image:      {np.median(same_d):.0f}")
print(f"median nearest DIFFERENT-chord train image: {np.median(other_d):.0f}")
for t in [10, 12, 16]:
    print(f"within {t}: same-chord {(same_d <= t).mean():.1%} | "
          f"different-chord {(other_d <= t).mean():.1%}")


import torch
from torch.utils.data import DataLoader, Subset
from train import val_transforms, build_resnet, device

val_ds = Subset(datasets.ImageFolder(DATA_DIR, transform=val_transforms), val_idx)
loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)  # order matches val_idx

ckpt = torch.load("outputs/chord_resnet18.pt", map_location=device)
model = build_resnet(len(ckpt["class_names"]), freeze_backbone=False).to(device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

preds = []
with torch.no_grad():
    for x, _ in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
correct = torch.cat(preds).numpy() == targets[val_idx]

near = same_d <= 10
print(f"\nhas close same-chord twin in train: n={near.sum():3d}  acc {correct[near].mean():.1%}")
print(f"no close twin:                      n={(~near).sum():3d}  acc {correct[~near].mean():.1%}")