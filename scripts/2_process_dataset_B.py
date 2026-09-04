"""Estimate and report ChArUco board distances for Dataset B.

This diagnostic stage uses the reference calibration and the original Dataset
B images. It detects board corners, solves PnP with the measured distortion,
and reports supplementary perpendicular board-plane distance in millimeters.
It does not generate model predictions or physical object dimensions.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import CALIBRATION_PATH, DATASET_B_DIR
from benchmark.geometry import perpendicular_distance_mm
from benchmark.io import create_charuco_detector, image_paths, load_calibration, read_image

# Load the calibration from Dataset A
K, D = load_calibration(CALIBRATION_PATH)

# Same board dimensions
board, detector = create_charuco_detector()

image_files = image_paths(DATASET_B_DIR)

print(f"Calculating supplementary reference distances for {len(image_files)} Dataset B images...\n")
print("-" * 50)
print(f"{'Image Name':<30} | {'Camera Distance (mm)':<20}")
print("-" * 50)

for image_file in image_files:
    try:
        img = read_image(image_file)
    except ValueError as error:
        print(f"Skipped {error}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    
    if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 4:
        # NEW OPEN CV METHOD: Match image points to physical board points
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        
        # Solve PnP gets the 3D position (Translation Vector) and rotation
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, D)
        
        if success:
            # Report perpendicular distance to the board plane, not its origin.
            distance_mm = perpendicular_distance_mm(rvec, tvec)
            
            # Print the file name and the calculated physical distance
            filename = os.path.basename(image_file)
            print(f"{filename:<30} | {distance_mm:.1f} mm")
        else:
            print(f"{os.path.basename(image_file):<30} | Pose estimation failed")
    else:
        print(f"{os.path.basename(image_file):<30} | Board not detected")

print("-" * 50)
print("Dataset B supplementary reference-distance processing complete.")