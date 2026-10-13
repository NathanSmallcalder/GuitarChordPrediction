from pathlib import Path
import hashlib
import cv2
import mediapipe as mp


# This script just checks for duplicate images and crops the hand out of each image, overwriting the original file.

dataset_dir = Path("data")  # Main directory containing chord subfolders
padding = 80  # Expanded pixel buffer so fingers/palm aren't cropped out
min_confidence = 0.2  # Lower threshold for complex/angled hand positions

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=min_confidence
)

print("--- Step 1: Scanning for exact duplicate images ---")
hashes = {}
duplicate_count = 0

image_paths = [
    p for p in dataset_dir.rglob("*") if p.suffix.lower() in image_extensions
]

for path in image_paths:
    with open(path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
        
    if file_hash in hashes:
        print(f"[REMOVED DUPLICATE] {path.name} (matches {hashes[file_hash].name})")
        path.unlink()
        duplicate_count += 1
    else:
        hashes[file_hash] = path

print(f"Removed {duplicate_count} duplicates.\n")

image_paths = [
    p for p in dataset_dir.rglob("*") if p.suffix.lower() in image_extensions
]

# ==========================================
# 3. HELPER: HAND DETECTION WITH ROTATION
# ==========================================
def get_hand_landmarks(img_bgr):
    """Attempts MediaPipe detection at 0, 90, 180, and 270 degree rotations."""
    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    
    for angle, rot_flag in rotations:
        rotated = img_bgr if rot_flag is None else cv2.rotate(img_bgr, rot_flag)
        img_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark, rotated
            
    return None, None

# ==========================================
# 4. STEP 2: DETECT, CROP, AND OVERWRITE
# ==========================================
print(f"--- Step 2: Processing {len(image_paths)} unique images ---")

cropped_count = 0
deleted_count = 0

for path in image_paths:
    img = cv2.imread(str(path))
    if img is None:
        print(f"[SKIPPED - CORRUPT] {path.name}")
        continue

    landmarks, processed_img = get_hand_landmarks(img)

    if landmarks:
        h, w, _ = processed_img.shape
        x_coords = [int(lm.x * w) for lm in landmarks]
        y_coords = [int(lm.y * h) for lm in landmarks]

        # Calculate bounding box with safety padding
        xmin = max(0, min(x_coords) - padding)
        xmax = min(w, max(x_coords) + padding)
        ymin = max(0, min(y_coords) - padding)
        ymax = min(h, max(y_coords) + padding)

        # Slice array and overwrite original file
        cropped_img = processed_img[ymin:ymax, xmin:xmax]
        cv2.imwrite(str(path), cropped_img)
        print(f"[CROPPED & SAVED] {path.name}")
        cropped_count += 1
    else:
        # Delete images where no hand structure can be identified
        print(f"[NO HAND DETECTED - DELETED] {path.name}")
        path.unlink()
        deleted_count += 1

hands.close()

# ==========================================
# 5. SUMMARY
# ==========================================
print("\n--- Pipeline Execution Summary ---")
print(f"Duplicates removed:  {duplicate_count}")
print(f"Images cropped:     {cropped_count}")
print(f"Images deleted:     {deleted_count}")