"""Webcam capture that zooms into the fretting hand and saves training crops.

Tracks your chord hand with MediaPipe, keeps a smoothed square crop locked on
it, and writes 224x224 JPEGs into data/<label>/ - the same layout cleanup.py
produces, so scraped and captured images live side by side.

Usage:
    python capture.py G                 # capture into data/g_chord
    python capture.py G --auto          # start with auto-capture running
    python capture.py Am --hand right   # if it locks onto the wrong hand

Keys:
    SPACE   capture one frame
    A       toggle auto-capture
    H       switch which hand is tracked
    [ / ]   less / more padding around the hand
    Q/ESC   quit
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
OUT_SIZE = 224          # final crop size, matches a standard CNN input
PADDING = 80            # pixel buffer around the hand, as in cleanup.py
SMOOTHING = 0.35        # 0 = frozen crop, 1 = no smoothing (jittery)
AUTO_INTERVAL = 0.4     # seconds between shots in auto mode
MIN_CONFIDENCE = 0.5    # higher than cleanup.py - video tracking is easier

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


# ==========================================
# 2. HELPER: SMOOTHED SQUARE CROP BOX
# ==========================================
def hand_box(landmarks, frame_w, frame_h, padding):
    """Square bounding box around the hand landmarks, clamped to the frame.

    Kept square so the resize to OUT_SIZE never distorts the aspect ratio -
    a stretched hand is a different shape to the CNN.
    """
    xs = [lm.x * frame_w for lm in landmarks]
    ys = [lm.y * frame_h for lm in landmarks]

    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) + 2 * padding
    side = min(side, frame_w, frame_h)  # can't be bigger than the frame

    # Clamp the centre so the square stays fully inside the frame.
    cx = min(max(cx, side / 2), frame_w - side / 2)
    cy = min(max(cy, side / 2), frame_h - side / 2)
    return cx, cy, side


def smooth(prev, current, alpha=SMOOTHING):
    """Exponential moving average, so the zoom glides instead of snapping."""
    if prev is None:
        return current
    return tuple(p + alpha * (c - p) for p, c in zip(prev, current))


# ==========================================
# 3. MAIN CAPTURE LOOP
# ==========================================
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", help="chord name, e.g. G - becomes data/g_chord/")
    ap.add_argument("--out", default="data", help="dataset root")
    ap.add_argument("--hand", choices=["left", "right"], default="left",
                    help="which hand to track (default left = fretting hand "
                         "for a right-handed player)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--auto", action="store_true",
                    help="start with auto-capture already running")
    ap.add_argument("--size", type=int, default=OUT_SIZE)
    args = ap.parse_args()

    out_dir = Path(args.out) / f"{args.label.lower()}_chord"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.jpg")))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    target = args.hand
    padding = PADDING
    auto = args.auto
    last_auto = 0.0
    saved = 0
    box = None

    print(f"Capturing '{args.label}' into {out_dir}/  ({existing} already there)")
    print("SPACE=shoot  A=auto  H=switch hand  [/]=padding  Q=quit")

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                        min_detection_confidence=MIN_CONFIDENCE,
                        min_tracking_confidence=MIN_CONFIDENCE) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break

            # Mirror the frame: gives a natural selfie-style preview, and
            # matches the orientation MediaPipe assumes when it labels
            # handedness, so the Left/Right labels come back correct.
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            preview = frame.copy()

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            landmarks = None
            if results.multi_hand_landmarks:
                for lm, handedness in zip(results.multi_hand_landmarks,
                                          results.multi_handedness):
                    if handedness.classification[0].label.lower() == target:
                        landmarks = lm
                        break

            crop = None
            if landmarks:
                box = smooth(box, hand_box(landmarks.landmark, w, h, padding))
                cx, cy, side = box
                x0, y0 = int(cx - side / 2), int(cy - side / 2)
                x1, y1 = int(cx + side / 2), int(cy + side / 2)
                crop = frame[y0:y1, x0:x1]

                mp_draw.draw_landmarks(preview, landmarks,
                                       mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 255, 0), 2)
            else:
                box = None  # drop the lock so it doesn't drift off a stale box

            # ---- auto-capture ----
            now = time.time()
            if auto and crop is not None and crop.size:
                if now - last_auto >= AUTO_INTERVAL:
                    save(crop, out_dir, args.label, args.size)
                    saved += 1
                    last_auto = now

            # ---- HUD ----
            status = "AUTO" if auto else "manual"
            colour = (0, 200, 255) if auto else (255, 255, 255)
            cv2.putText(preview, f"{args.label}  {status}  saved:{saved}"
                        f"  total:{existing + saved}", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            cv2.putText(preview, f"hand:{target}  pad:{padding}"
                        + ("" if landmarks else "   NO HAND"), (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if landmarks else (0, 0, 255), 2)

            if crop is not None and crop.size:
                thumb = cv2.resize(crop, (160, 160))
                preview[h - 170:h - 10, w - 170:w - 10] = thumb

            cv2.imshow("capture - chord hand", preview)

            # ---- keys ----
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" ") and crop is not None and crop.size:
                save(crop, out_dir, args.label, args.size)
                saved += 1
            elif key == ord("a"):
                auto = not auto
            elif key == ord("h"):
                target = "right" if target == "left" else "left"
                box = None
            elif key == ord("["):
                padding = max(0, padding - 10)
            elif key == ord("]"):
                padding = min(300, padding + 10)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSaved {saved} new crops to {out_dir}/ "
          f"({existing + saved} total)")
    return 0


def save(crop, out_dir: Path, label: str, size: int) -> None:
    """Resize the crop to the CNN input size and write it out."""
    img = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    name = f"{label.lower()}_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    raise SystemExit(main())
