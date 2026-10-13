"""Run the trained chord model on a still image or a video.

Inference has to see the same kind of picture training did. cleanup.py cropped every training
image tight around the fretting hand with MediaPipe, so a raw photo - hand somewhere in a wide
shot - looks nothing like what the model learnt on. preprocess_image repeats that crop, then
hands the result to augment.py's eval transform, so the resize and the normalisation are the
exact ones used during training rather than a second, slightly different version of them.

Usage:
    python predict.py photo.jpg
    python predict.py test.mp4 --every 5      # score every 5th frame
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from PIL import Image
from torch import nn

from augment import build_transforms, dataset_dir

model_path = Path("model2.pth")
padding = 80          # same buffer cleanup.py cropped the training images with
min_confidence = 0.2  # low, as in cleanup.py - angled chord hands detect poorly

_, eval_transform = build_transforms()

mp_hands = mp.solutions.hands
_hands = None  # built on first use; loading MediaPipe costs a second or so


def get_hands():
    """Lazily opens the MediaPipe detector, so importing this module stays cheap."""
    global _hands
    if _hands is None:
        _hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=min_confidence,
        )
    return _hands


def load_classes(root=dataset_dir):
    """Class names in ImageFolder order - sorted folder names, which is how the model indexes."""
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def build_scratch_cnn(num_classes):
    """The from-scratch architecture in model.py, for loading model.pth.

    It has to match layer for layer: those checkpoint keys are positional ("10.weight"), so
    inserting or dropping a layer silently shifts every weight."""
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),
        nn.Linear(64 * 28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
    )


def build_for_state_dict(state, num_classes):
    """Builds whichever architecture the checkpoint came from.

    A state_dict is only weights - nothing in it says which model to pour them into - but the
    key names give it away: the ResNet has "layer1.0.conv1.weight", the scratch Sequential has
    positional names like "10.weight". Deciding here means the CLI and app.py both work with
    either file, so switching model_path is the whole change.
    """
    if any(key.startswith("layer1.") for key in state):
        from model2 import build_model as build_resnet

        # pretrained=False: load_state_dict is about to overwrite every parameter, so fetching
        # the ImageNet weights first would be a download for nothing.
        return build_resnet(num_classes, freeze_backbone=False, pretrained=False)

    return build_scratch_cnn(num_classes)


def load_model(path=model_path, classes=None):
    """Returns (model, classes, device), ready for inference."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {path}. Train one first: "
            f"'python model2.py' for {path.name}, or 'python model.py' for model.pth."
        )

    classes = classes or load_classes()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state = torch.load(path, map_location=device)
    model = build_for_state_dict(state, len(classes))
    model.load_state_dict(state)
    model.to(device)
    model.eval()  # the ResNet has batchnorm, so skipping this genuinely changes the answers
    return model, classes, device


# ==========================================
# PREPROCESSING
# ==========================================
def crop_to_hand(img_bgr, pad=padding):
    """Crops tight around the hand, trying 0/90/180/270 degrees as cleanup.py does.

    Returns the cropped BGR image, or None when no hand is found - the caller decides whether
    that means "skip this frame" or "use the picture as it came".
    """
    rotations = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

    for rot_flag in rotations:
        rotated = img_bgr if rot_flag is None else cv2.rotate(img_bgr, rot_flag)
        results = get_hands().process(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            continue

        landmarks = results.multi_hand_landmarks[0].landmark
        h, w = rotated.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]

        xmin = max(0, min(xs) - pad)
        xmax = min(w, max(xs) + pad)
        ymin = max(0, min(ys) - pad)
        ymax = min(h, max(ys) + pad)

        crop = rotated[ymin:ymax, xmin:xmax]
        if crop.size:
            return crop

    return None


def to_tensor(img_bgr):
    """BGR array -> normalised 1x3x224x224 batch, using the training eval pipeline.

    Going through PIL looks like a detour, but torchvision transforms are what produced the
    tensors the model was trained on; hand-rolling the resize and the /255 here would drop the
    ImageNet normalisation and shift every input away from what the weights expect.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return eval_transform(Image.fromarray(rgb)).unsqueeze(0)


def preprocess_array(img_bgr, require_hand=False):
    """Crop around the hand playing the chord, resize to 224x224, convert to RGB, normalise,
    and return it as a batch-of-one tensor. Returns None if require_hand and none is found.

    Takes an already-loaded BGR array, so a webcam frame or an upload can go through the same
    path as a file on disk - there is only ever one definition of "what the model eats".

    With no hand detected the whole picture is used instead: a photo already cropped to the
    hand (anything out of data/, or capture.py's output) often won't re-detect cleanly, and
    that is exactly the case where the uncropped image is already the right input.
    """
    crop = crop_to_hand(img_bgr)
    if crop is None:
        if require_hand:
            return None
        crop = img_bgr

    return to_tensor(crop)


def preprocess_image(image_path, require_hand=False):
    """preprocess_array for a file on disk."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    return preprocess_array(img, require_hand)


def preprocess_frames(video_path, every=1, require_hand=True):
    """Yields (frame_number, tensor) for the frames of a video.

    A generator rather than a folder of JPEGs: a few seconds of 30fps footage is hundreds of
    files that only exist to be read straight back in. `every` subsamples - consecutive frames
    of a held chord are near-identical, so scoring all of them buys nothing.

    Frames with no detectable hand are skipped by default; mid-strum blur and hand-off-screen
    frames would otherwise be classified on whatever happens to fill the frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_number = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_number % every == 0:
                tensor = preprocess_array(frame, require_hand)
                if tensor is not None:
                    yield frame_number, tensor

            frame_number += 1
    finally:
        cap.release()


# ==========================================
# PREDICTION
# ==========================================
def predict_probs(model, tensor, classes, device):
    """Returns {class name: probability} for one preprocessed batch-of-one."""
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = torch.softmax(logits, dim=1)[0]
    return dict(zip(classes, probs.tolist()))


def predict_tensor(model, tensor, classes, device):
    """Returns (class name, confidence) - the top entry of predict_probs."""
    scores = predict_probs(model, tensor, classes, device)
    name = max(scores, key=scores.get)
    return name, scores[name]


def predict_chord(image_path, model=None, classes=None, device=None):
    """Predicts the chord in a single image."""
    if model is None:
        model, classes, device = load_model()
    return predict_tensor(model, preprocess_image(image_path), classes, device)


def predict_video(video_path, every=5, model=None, classes=None, device=None):
    """Scores the sampled frames of a video and returns (per-frame results, vote counts).

    A per-frame majority is steadier than any single frame: one blurred hand position sends an
    individual prediction anywhere, but it rarely outvotes the rest of the clip.
    """
    if model is None:
        model, classes, device = load_model()

    results = []
    for frame_number, tensor in preprocess_frames(video_path, every=every):
        name, confidence = predict_tensor(model, tensor, classes, device)
        results.append((frame_number, name, confidence))

    return results, Counter(name for _, name, _ in results)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="image or video file")
    ap.add_argument("--every", type=int, default=5,
                    help="for video: score every Nth frame (default 5)")
    ap.add_argument("--model", default=str(model_path), help="checkpoint to load")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    model, classes, device = load_model(Path(args.model))
    print(f"Loaded {args.model} on {device.upper()}  classes: {classes}")

    if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        results, votes = predict_video(path, every=args.every, model=model,
                                       classes=classes, device=device)
        if not results:
            print("No hand detected in any sampled frame.")
            return 1

        for frame_number, name, confidence in results:
            print(f"  frame {frame_number:>5}: {name} ({confidence:.1%})")

        winner, count = votes.most_common(1)[0]
        print(f"\n{len(results)} frames scored -> {winner} ({count}/{len(results)} frames)")
    else:
        name, confidence = predict_chord(path, model, classes, device)
        print(f"\n{path.name} -> {name} ({confidence:.1%})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
