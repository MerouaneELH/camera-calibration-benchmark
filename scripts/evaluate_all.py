import cv2
import numpy as np
import glob
import os
import pandas as pd

# Load Ground Truth
gt_data = np.load("reference_calibration.npz")
K_ref, D_ref = gt_data['K'], gt_data['D']

# Setup ChArUco
board = cv2.aruco.CharucoBoard((5, 7), 0.038, 0.029, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
detector = cv2.aruco.CharucoDetector(board)

# Load AI Predictions
models = {}
for pred_file in glob.glob("preds_*.npz"):
    model_name = pred_file.replace("preds_", "").replace(".npz", "")
    models[model_name] = dict(np.load(pred_file))

results = []

for img_path in glob.glob('dataset_B_evaluation/*.jpg'):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) < 4:
        continue
        
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    
    # Calculate Ground Truth 3D Distance
    success, _, tvec_ref = cv2.solvePnP(obj_points, img_points, K_ref, D_ref)
    if not success: continue
    dist_ref = np.linalg.norm(tvec_ref) * 1000
    
    # Evaluate each available AI model
    row = {"Image": filename, "GT_Distance_mm": dist_ref}
    
    for model_name, preds in models.items():
        if filename in preds:
            K_pred = preds[filename]
            _, _, tvec_pred = cv2.solvePnP(obj_points, img_points, K_pred, np.zeros(5))
            dist_pred = np.linalg.norm(tvec_pred) * 1000
            
            error_mm = abs(dist_pred - dist_ref)
            f_error_pct = abs(((K_pred[0,0]+K_pred[1,1])/2) - ((K_ref[0,0]+K_ref[1,1])/2)) / ((K_ref[0,0]+K_ref[1,1])/2) * 100
            
            row[f"{model_name}_Error_mm"] = error_mm
            row[f"{model_name}_Focal_Error_pct"] = f_error_pct
            
    results.append(row)

# Save to CSV and Print Summary
df = pd.DataFrame(results)
df.to_csv("evaluation_results.csv", index=False)

print("\n--- BENCHMARK SUMMARY (Overall Dataset Averages) ---")
for model_name in models.keys():
    mean_err = df[f"{model_name}_Error_mm"].mean()
    median_err = df[f"{model_name}_Error_mm"].median()
    print(f"{model_name.capitalize()}: Mean Error = {mean_err:.2f} mm | Median Error = {median_err:.2f} mm")