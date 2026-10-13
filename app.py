"""Gradio front end for the chord classifier.

Loads outputs/chord_resnet18.pt (made by `python train.py`) once at startup, then every click
reuses it. The cropping and transforms are imported from train.py, so the app prepares images
exactly the way training did.

Usage:
    python app.py
"""

import cv2
import gradio as gr
import gradio.themes as themes
import numpy as np
import torch
from PIL import Image

from train import OUT_DIR, build_resnet, crop_hand, device, mp_hands, val_transforms

MODEL_PATH = OUT_DIR / "chord_resnet18.pt"

# The hand fills about half of each training image, so leave that much room around it here too.
# crop_hand's default (0.35) would zoom in tighter than anything the model was trained on.
CROP_PADDING = 0.9


# ---- Load everything once, before the server starts ----
checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASSES = checkpoint["class_names"]

model = build_resnet(len(CLASSES), freeze_backbone=False)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)
model.eval()  # switch off training-only behaviour (batchnorm updates)

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
print(f"Loaded {MODEL_PATH} on {device}  classes: {CLASSES}")


def classify(image, unmirror):
    """PIL image in, ({chord: probability}, cropped image the model saw) out."""
    if image is None:
        return {}, None

    # MediaPipe/OpenCV work in BGR; PIL gives RGB.
    bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if unmirror:  # undo a mirrored webcam shot - the training images aren't mirrored
        bgr = cv2.flip(bgr, 1)

    # Same steps as preprocess_dataset in train.py: crop to the hand (or keep the whole
    # picture if none is found), resize to 224, then the validation transforms.
    crop = crop_hand(bgr, hands, padding=CROP_PADDING)
    if crop is None or crop.size == 0:
        crop = bgr
    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = val_transforms(Image.fromarray(rgb)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].tolist()

    # Strip the "_chord" suffix so the label reads "G", "Am", ...
    names = [c.removesuffix("_chord").capitalize() for c in CLASSES]
    return dict(zip(names, probs)), rgb


with gr.Blocks(title="Guitar Chord Classifier") as demo:
    gr.Markdown("## Guitar Chord Classifier")
    gr.Markdown(
        "Upload a photo of a fretting hand, or take one with your webcam, then press "
        "**Predict**."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                sources=["upload", "webcam", "clipboard"],
                # Gradio mirrors the webcam by default; ask for the frame the camera actually saw.
                webcam_options=gr.WebcamOptions(mirror=False),
                label="Input Image",
            )
            unmirror = gr.Checkbox(
                value=False,
                label="Flip input horizontally",
                info="If webcam predictions are stuck on one chord, your browser mirrored the "
                     "shot - tick this to undo it.",
            )
            predict_button = gr.Button("Predict", variant="primary")
        with gr.Column():
            output_label = gr.Label(label="Predicted Chord", num_top_classes=3)
            model_view = gr.Image(label="What the model saw", height=224)

    predict_button.click(classify, inputs=[input_image, unmirror],
                         outputs=[output_label, model_view])


if __name__ == "__main__":
    demo.launch(theme=themes.Soft())  # Gradio 6 takes the theme here, not on Blocks
