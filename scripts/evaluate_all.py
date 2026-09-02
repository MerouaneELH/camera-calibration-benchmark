import cv2
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import CALIBRATION_PATH, DATASET_B_UNDISTORTED_DIR, OUTPUT_DIR
from benchmark.io import create_charuco_detector, image_paths, load_calibration, read_image

# Load Ground Truth
K_ref, _ = load_calibration(CALIBRATION_PATH)

# Setup ChArUco
board, detector = create_charuco_detector()

# Load AI Predictions
models = {}
for pred_file in sorted(OUTPUT_DIR.glob("preds_*.npz")):
    model_name = pred_file.stem.removeprefix("preds_")
    models[model_name] = dict(np.load(pred_file))

if not models:
    raise SystemExit(f"No prediction files found in {OUTPUT_DIR}")

image_files = image_paths(DATASET_B_UNDISTORTED_DIR)

results = []

for img_path in image_files:
    filename = img_path.name
    try:
        img = read_image(img_path)
    except ValueError as error:
        print(f"Skipping {error}")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) < 4:
        continue
        
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    
    # Calculate Ground Truth 3D Distance
    success, _, tvec_ref = cv2.solvePnP(obj_points, img_points, K_ref, np.zeros(5))
    if not success: continue
    dist_ref = np.linalg.norm(tvec_ref) * 1000
    
    # Evaluate each available AI model
    row = {"Image": filename, "GT_Distance_mm": dist_ref}
    
    for model_name, preds in models.items():
        if filename in preds:
            K_pred = preds[filename]
            K_pred = np.asarray(K_pred)
            if K_pred.shape != (3, 3) or not np.isfinite(K_pred).all():
                print(f"Skipping {model_name}/{filename}: invalid intrinsic matrix.")
                continue
            success_pred, _, tvec_pred = cv2.solvePnP(
                obj_points, img_points, K_pred, np.zeros(5)
            )
            if not success_pred:
                print(f"Skipping {model_name}/{filename}: pose estimation failed.")
                continue
            dist_pred = np.linalg.norm(tvec_pred) * 1000
            
            error_mm = abs(dist_pred - dist_ref)
            f_error_pct = abs(((K_pred[0,0]+K_pred[1,1])/2) - ((K_ref[0,0]+K_ref[1,1])/2)) / ((K_ref[0,0]+K_ref[1,1])/2) * 100
            
            row[f"{model_name}_Error_mm"] = error_mm
            row[f"{model_name}_Focal_Error_pct"] = f_error_pct
            
    results.append(row)

# Save to CSV and Print Summary
df = pd.DataFrame(results)
if df.empty:
    raise SystemExit("No valid evaluation rows were produced.")

df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)

print("\n--- BENCHMARK SUMMARY (Overall Dataset Averages) ---")
for model_name in models.keys():
    error_column = f"{model_name}_Error_mm"
    if error_column not in df:
        print(f"{model_name.capitalize()}: no valid predictions.")
        continue
    mean_err = df[error_column].mean()
    median_err = df[error_column].median()
    print(f"{model_name.capitalize()}: Mean Error = {mean_err:.2f} mm | Median Error = {median_err:.2f} mm")