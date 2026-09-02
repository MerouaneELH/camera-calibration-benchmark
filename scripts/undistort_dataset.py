import cv2
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import CALIBRATION_PATH, DATASET_B_DIR, DATASET_B_UNDISTORTED_DIR
from benchmark.io import image_paths, load_calibration, read_image

# 1. Load your ChArUco ground truth parameters
K, D = load_calibration(CALIBRATION_PATH)

# 2. Set up folders
DATASET_B_UNDISTORTED_DIR.mkdir(parents=True, exist_ok=True)

# 3. Process each image
input_paths = image_paths(DATASET_B_DIR)

for img_path in input_paths:
    filename = img_path.name
    try:
        img = read_image(img_path)
    except ValueError as error:
        print(f"Skipped {error}")
        continue
        
    # Mathematically flatten the curved glass effect
    undistorted_img = cv2.undistort(img, K, D)
    
    # Save to the new folder
    save_path = DATASET_B_UNDISTORTED_DIR / filename
    if not cv2.imwrite(str(save_path), undistorted_img):
        print(f"Skipped {filename}: could not write output image.")
        continue
    print(f"Undistorted: {filename}")

print(f"Done! {len(input_paths)} images safely flattened and saved to {DATASET_B_UNDISTORTED_DIR}.")