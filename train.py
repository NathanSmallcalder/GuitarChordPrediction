"""
Guitar chord classifier: train a baseline CNN or a fine-tuned ResNet18.

Usage:
    python train.py --model cnn
    python train.py --model resnet18
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchvision import datasets, transforms, models

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

import cv2
import mediapipe as mp
import imagehash
from PIL import Image


# =====================================================================
# Settings
# =====================================================================
RAW_DIR = Path("data_raw")   # originals, mixed cropped/uncropped
DATA_DIR = Path("data")      # cleaned crops go here (training reads from this)
OUT_DIR = Path("outputs")    # weights, plots, reports
PADDING = 0.35               # extra margin around the hand, as a fraction
HASH_CACHE = OUT_DIR / "hashes.npy"     # delete outputs/*.npy if the dataset changes
GROUPS_CACHE = OUT_DIR / "groups.npy"
HASH_THRESH = 6              # Hamming distance (out of 64) counted as near-duplicate
PURGE_THRESH = 10            # val images this close to any train image get moved to train

IMG_SIZE = 224
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_WORKERS = 2
SEED = 42

# Baseline CNN
CNN_EPOCHS = 20
CNN_LR = 1e-3

# ResNet: phase 1 (head only), phase 2 (full fine-tune)
HEAD_EPOCHS = 5
HEAD_LR = 1e-3
FT_EPOCHS = 8               # val peaked around epoch 7 last run; more just overfits
FT_LR = 1e-4

# Regularisation (fights overfitting once train acc hits ~100%)
WEIGHT_DECAY = 1e-2          # AdamW: penalises large weights
LABEL_SMOOTHING = 0.1        # soft targets: model can't be 100% sure on train images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mp_hands = mp.solutions.hands

# ImageNet statistics: the pretrained ResNet expects inputs normalised this way
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =====================================================================
# Transforms
# =====================================================================
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# =====================================================================
# Preprocessing (MediaPipe hand cropping)
# =====================================================================
def crop_hand(image_bgr, hands, padding=PADDING):
    """Return a square crop around the detected hand, or None if no hand found."""
    h, w = image_bgr.shape[:2]

    # MediaPipe expects RGB; OpenCV loads BGR
    results = hands.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    if not results.multi_hand_landmarks:
        return None

    # 21 landmarks, coordinates normalised to 0-1 -> convert to pixels
    landmarks = results.multi_hand_landmarks[0].landmark
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    # Square box centred on the hand, enlarged by the padding
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + padding)

    # Clamp to image bounds
    x1, y1 = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    x2, y2 = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))

    return image_bgr[y1:y2, x1:x2]


def preprocess_dataset(raw_dir=RAW_DIR, out_dir=DATA_DIR, size=IMG_SIZE):
    no_hand = []
    exts = {".jpg", ".jpeg", ".png"}
    with mp_hands.Hands(static_image_mode=True,
                        min_detection_confidence=0.5) as hands:
        for img_path in raw_dir.rglob("*"):
            if img_path.suffix.lower() not in exts:
                continue
            image = cv2.imread(str(img_path))
            if image is None:          # corrupt/unreadable file
                continue

            crop = crop_hand(image, hands)
            if crop is None or crop.size == 0:
                crop = image           # fall back to the original
                no_hand.append(img_path)

            crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

            # Mirror the folder structure: data_raw/G/x.jpg -> data/G/x.jpg
            out_path = out_dir / img_path.relative_to(raw_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop)

    print(f"Done. {len(no_hand)} images had no hand detected (kept uncropped).")
    return no_hand


# =====================================================================
# Leakage-safe split: group near-duplicate images together
# =====================================================================
def hash_bits(paths):
    """64-bit perceptual hash per image, as a boolean array [n_images, 64]. Cached."""
    if HASH_CACHE.exists():
        bits = np.load(HASH_CACHE)
        if len(bits) == len(paths):
            return bits
        print("Hash cache doesn't match dataset size, recomputing...")

    print("Hashing images (one-off)...")
    bits = np.array([imagehash.phash(Image.open(p)).hash.flatten() for p in paths])
    HASH_CACHE.parent.mkdir(exist_ok=True)
    np.save(HASH_CACHE, bits)
    return bits


def hash_groups(paths, bits, thresh=HASH_THRESH, chunk=256):
    """Cluster near-duplicate images into groups via union-find over hash distance."""
    if GROUPS_CACHE.exists():
        groups = np.load(GROUPS_CACHE)
        if len(groups) == len(paths):
            return groups
        print("Group cache doesn't match dataset size, recomputing...")

    n = len(paths)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    # Compare every image against every other, in chunks to save memory
    for start in range(0, n, chunk):
        block = bits[start:start + chunk]
        dists = (block[:, None, :] != bits[None, :, :]).sum(axis=2)
        for i, j in zip(*np.nonzero(dists <= thresh)):
            i += start
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj         # merge the two groups

    groups = np.array([find(i) for i in range(n)])
    GROUPS_CACHE.parent.mkdir(exist_ok=True)
    np.save(GROUPS_CACHE, groups)
    return groups


def purge_val_twins(train_idx, val_idx, bits, thresh=PURGE_THRESH):
    """Move any val image with a close relative in train over to train.

    Repeats until stable: moving an image into train can create new twins
    for val images that were close to *it*.
    """
    moved_total = 0
    while True:
        train_bits = bits[train_idx]
        min_d = np.array([(train_bits != bits[i]).sum(axis=1).min() for i in val_idx])
        twins = min_d <= thresh
        if not twins.any():
            break
        train_idx = np.concatenate([train_idx, val_idx[twins]])
        val_idx = val_idx[~twins]
        moved_total += twins.sum()
    print(f"Purge: moved {moved_total} val images with a train twin "
          f"(distance <= {thresh}) into train")
    return train_idx, val_idx


# =====================================================================
# Models
# =====================================================================
class ChordCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 32),     # 224 -> 112
            self._block(32, 64),    # 112 -> 56
            self._block(64, 128),   # 56  -> 28
            self._block(128, 256),  # 28  -> 14
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # [256, 14, 14] -> [256, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # -> [256]
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),      # -> one score per chord
        )

    @staticmethod
    def _block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet(num_classes, freeze_backbone=True):
    # Load weights learned on ImageNet (1.2M images, 1000 classes)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False    # don't update pretrained layers

    # Replace the 1000-class head with one for our chords (trainable by default)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# =====================================================================
# Training / evaluation
# =====================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()                          # enables dropout, batchnorm updates
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # clear last batch's gradients
        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)
        loss.backward()                    # compute gradients
        optimizer.step()                   # update weights

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()                           # no gradient tracking: faster, less memory
def evaluate(model, loader, criterion):
    model.eval()                           # disables dropout, freezes batchnorm stats
    running_loss = 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        running_loss += criterion(outputs, labels).item() * images.size(0)
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro")
    return running_loss / len(labels), acc, macro_f1, preds, labels


def fit(model, train_loader, val_loader, criterion, optimizer,
        num_epochs, scheduler=None):
    history = {k: [] for k in ["train_loss", "train_acc", "val_loss", "val_acc", "val_f1"]}
    best_f1, best_weights = 0.0, copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion)
        if scheduler:
            scheduler.step()

        for k, v in zip(history, [tr_loss, tr_acc, va_loss, va_acc, va_f1]):
            history[k].append(v)

        marker = ""
        if va_f1 > best_f1:
            best_f1, best_weights = va_f1, copy.deepcopy(model.state_dict())
            marker = " *"

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} F1 {va_f1:.3f} | "
              f"{time.time() - start:.0f}s{marker}")

    model.load_state_dict(best_weights)    # return the best epoch, not the last
    print(f"Best val macro-F1: {best_f1:.3f}")
    return model, history


def merge_histories(*histories):
    """Join histories from several fit() calls into one continuous record."""
    merged = {k: [] for k in histories[0]}
    for h in histories:
        for k in merged:
            merged[k].extend(h[k])
    return merged


# =====================================================================
# Reporting
# =====================================================================
def plot_history(history, title, path, phase_break=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")

    ax2.plot(epochs, history["train_acc"], label="train acc")
    ax2.plot(epochs, history["val_acc"], label="val acc")
    ax2.plot(epochs, history["val_f1"], label="val macro-F1", linestyle="--")
    ax2.set_title("Accuracy / F1")
    ax2.set_xlabel("Epoch")

    for ax in (ax1, ax2):
        if phase_break:
            ax.axvline(phase_break + 0.5, color="grey", linestyle=":", label="unfreeze")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(labels, preds, class_names, title, path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Write the count in each cell
    threshold = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =====================================================================
# Main: everything that *does* something lives here
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train a guitar chord classifier.")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    args = parser.parse_args()

    print(f"Using device: {device} | model: {args.model}")
    OUT_DIR.mkdir(exist_ok=True)

    # --- Reproducibility ---
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --- Preprocess once ---
    if not DATA_DIR.exists():
        preprocess_dataset()

    # --- Data: same images, two transform pipelines ---
    train_full = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    val_full = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

    class_names = train_full.classes
    num_classes = len(class_names)
    print(f"{num_classes} classes: {class_names}")

    # --- Group split: near-duplicates never straddle train/val ---
    paths = [p for p, _ in train_full.samples]
    targets = np.array(train_full.targets)
    bits = hash_bits(paths)
    groups = hash_groups(paths, bits)

    sizes = np.bincount(np.unique(groups, return_inverse=True)[1])
    print(f"{len(sizes)} groups | largest: {sizes.max()} images | "
          f"singletons: {(sizes == 1).sum()}")

    # Groups containing more than one chord label = probable labelling mistakes
    mixed = [g for g in np.unique(groups) if len(np.unique(targets[groups == g])) > 1]
    print(f"Groups spanning multiple chords: {len(mixed)}")
    for g in mixed[:10]:
        members = np.where(groups == g)[0][:4]
        print("   ", [(Path(paths[i]).parent.name, Path(paths[i]).name) for i in members])

    n_splits = round(1 / VAL_SPLIT)          # 0.2 -> 5 folds -> first fold is ~20% val
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(np.arange(len(paths)), targets, groups))

    # Anything in val that still has a close twin in train goes to train instead
    train_idx, val_idx = purge_val_twins(train_idx, val_idx, bits)

    print("Val per class:", {n: int(c) for n, c in
                             zip(class_names, np.bincount(targets[val_idx], minlength=num_classes))})

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,   # don't respawn workers every epoch
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # --- Class-weighted loss (weights from TRAIN split only) ---
    train_targets = np.array(train_full.targets)[train_idx]
    class_counts = np.bincount(train_targets, minlength=num_classes)
    class_weights = len(train_targets) / (num_classes * class_counts)
    class_weights = class_weights ** 0.5   # soften: inverse-sqrt-frequency
    print("Class weights:",
          {n: round(float(w), 2) for n, w in zip(class_names, class_weights)})

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(device),
        label_smoothing=LABEL_SMOOTHING,
    )

    # --- Model + training ---
    phase_break = None

    if args.model == "cnn":
        model = ChordCNN(num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=CNN_LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)
        model, history = fit(model, train_loader, val_loader, criterion,
                             optimizer, CNN_EPOCHS, scheduler)

    else:  # resnet18
        # Phase 1: train only the new head
        print("\n--- Phase 1: training head (backbone frozen) ---")
        model = build_resnet(num_classes, freeze_backbone=True).to(device)
        optimizer = optim.AdamW(model.fc.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        model, hist_head = fit(model, train_loader, val_loader, criterion,
                               optimizer, HEAD_EPOCHS)

        # Phase 2: unfreeze and fine-tune gently
        print("\n--- Phase 2: fine-tuning full network ---")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FT_EPOCHS)
        model, hist_ft = fit(model, train_loader, val_loader, criterion,
                             optimizer, FT_EPOCHS, scheduler)

        history = merge_histories(hist_head, hist_ft)
        phase_break = HEAD_EPOCHS

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    # --- Save weights + class names (the Gradio app needs both) ---
    weights_path = OUT_DIR / f"chord_{args.model}.pt"
    torch.save({"state_dict": model.state_dict(), "class_names": class_names},
               weights_path)
    print(f"Saved model to {weights_path}")

    # --- Final evaluation on the best weights ---
    _, acc, macro_f1, preds, labels = evaluate(model, val_loader, criterion)
    report = classification_report(labels, preds, target_names=class_names, digits=3)
    print(f"\nFinal val accuracy: {acc:.3f} | macro-F1: {macro_f1:.3f}\n")
    print(report)
    (OUT_DIR / f"report_{args.model}.txt").write_text(
        f"accuracy: {acc:.4f}\nmacro_f1: {macro_f1:.4f}\n\n{report}"
    )

    plot_history(history, f"{args.model} training",
                 OUT_DIR / f"history_{args.model}.png", phase_break)
    plot_confusion(labels, preds, class_names, f"{args.model} confusion matrix",
                   OUT_DIR / f"confusion_{args.model}.png")
    print(f"Plots and report saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()