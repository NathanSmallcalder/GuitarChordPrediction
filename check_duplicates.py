
import numpy as np
from PIL import Image
import imagehash
from torchvision import datasets
from sklearn.model_selection import train_test_split

from train import DATA_DIR, VAL_SPLIT, SEED   # safe: train.py only runs main() when executed directly

ds = datasets.ImageFolder(DATA_DIR)
paths = [p for p, _ in ds.samples]
train_idx, val_idx = train_test_split(
    np.arange(len(paths)), test_size=VAL_SPLIT,
    stratify=ds.targets, random_state=SEED,
)

print("Hashing images...")
bits = np.array([imagehash.phash(Image.open(p)).hash.flatten() for p in paths])
train_bits = bits[train_idx]

THRESH = 6   # Hamming distance out of 64 bits; <=6 means visually near-identical
dupes = []
for i in val_idx:
    dists = (train_bits != bits[i]).sum(axis=1)
    j = dists.argmin()
    if dists[j] <= THRESH:
        dupes.append((paths[i], paths[train_idx[j]], int(dists[j])))

print(f"{len(dupes)}/{len(val_idx)} val images ({len(dupes)/len(val_idx):.1%}) "
      f"have a near-duplicate in train")
for v, t, d in dupes[:15]:
    print(f"  dist {d}: {v}  <->  {t}")