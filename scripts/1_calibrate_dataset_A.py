"""Estimate reference intrinsics from Dataset A's ChArUco observations.

The script detects board corners in every supported Dataset A image, retains
images with enough correspondences, and calls OpenCV's standard calibration.
The resulting ``K`` and ``D`` arrays, board metadata, and image size are saved
to ``Outputs/reference_calibration.npz`` for all downstream stages.
"""

import cv2
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import BOARD, CALIBRATION_PATH, DATASET_A_DIR
from benchmark.io import create_charuco_detector, image_paths, read_image

# ==========================================
# 1. PHYSICAL BOARD SETTINGS (38mm)
# ==========================================
board, detector = create_charuco_detector()

# ==========================================
# 2. PROCESS IMAGES
# ==========================================
image_files = image_paths(DATASET_A_DIR)

all_obj_points = []
all_img_points = []
image_size = None

if not image_files:
    raise SystemExit(f"No .jpg images found in {DATASET_DIR}")

print(f"Processing {len(image_files)} images from Dataset A...")

for image_file in image_files:
    try:
        img = read_image(image_file)
    except ValueError as error:
        print(f"Skipped {error}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = (gray.shape[1], gray.shape[0])

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 5:
        # NEW OPEN CV METHOD: Match the 2D corners to the 3D physical board coordinates
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        
        if obj_points is not None and img_points is not None and len(obj_points) > 5:
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
    else:
        print(f"Skipped {image_file}: Not enough corners detected.")

# ==========================================
# 3. CALCULATE CALIBRATION
# ==========================================
if len(all_obj_points) == 0:
    raise SystemExit("No corners were successfully matched. Check Dataset A images.")

print("\nCalculating K Matrix...")
# Standard OpenCV camera calibration
ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
    all_obj_points, 
    all_img_points, 
    image_size, 
    None, 
    None
)

print(f"\n✅ CALIBRATION SUCCESSFUL ✅")
print(f"Reprojection Error: {ret:.3f} pixels (Lower is better, aim for < 1.0)")
print("\nReference K Matrix:")
print(np.round(K, 2))

# Save for Dataset B
CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez(
    CALIBRATION_PATH,
    K=K,
    D=D,
    image_size=np.asarray(image_size),
    squares_x=BOARD.squares_x,
    squares_y=BOARD.squares_y,
    square_length=BOARD.square_length,
    marker_length=BOARD.marker_length,
    dictionary_id=BOARD.dictionary_id,
    reprojection_error=ret,
)
print(f"\nSaved reference parameters to '{CALIBRATION_PATH}'.")