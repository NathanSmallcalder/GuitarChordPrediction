# Guitar Chord Classifier

Recognises which chord (A, Am, B, C, D, E, F, G) a guitarist's fretting hand is playing from a photo.

## Model

A ResNet18 pretrained on ImageNet, with its last layer replaced by one output per chord. It's fine-tuned in two phases:

1. Only the new last layer trains, with the rest frozen (5 epochs).
2. The whole network trains at a low learning rate (8 epochs).

The best validation accuracy is about **92%** (macro-F1 0.92) on photos with no near-duplicates in training. The most common mistakes are F↔G and A→E.

## Pipeline

Before training or predicting, every image is cropped to the hand using MediaPipe.

| Script | What it does |
|---|---|
| `Scraper.py` | Downloads chord photos from the web into `data/<chord>_chord/` |
| `capture.py` | Webcam tool: tracks your hand and saves cropped shots into `data/<chord>_chord/` |
| `cleanup.py` | Removes duplicate images and crops each one to the hand |
| `train.py` | Splits the data, trains the model and saves `outputs/chord_resnet18.pt` plus plots and a report |
| `app.py` | Gradio web app: upload a photo or use the webcam, and get the top 3 chords |
| `check_split.py`, `check_duplicates.py` | Checks that no near-identical images appear in both train and validation |

**Train/validation split:** many images are near-copies, like recoloured versions or frames from the same session. `train.py` groups look-alike images by perceptual hash and keeps each group on one side of the split, so the validation score isn't inflated.

**Augmentation:** training images get a random zoom, a small rotation and lighting changes each epoch. They are never flipped, because a mirrored hand looks like a left-handed player.

## Usage

```bash
python capture.py G               # collect more G-chord images
python train.py                   # train ResNet18 (or: --model cnn for the small baseline)
python app.py                     # launch the web app
```

Requires: `torch`, `torchvision`, `mediapipe`, `opencv-python`, `gradio`, `scikit-learn`, `imagehash`, `matplotlib`.
