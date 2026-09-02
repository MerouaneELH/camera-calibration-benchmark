import sys
import torch
import numpy as np
import torchvision.transforms.functional as TF
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anycalib import AnyCalib
from benchmark.config import DATASET_B_UNDISTORTED_DIR, OUTPUT_DIR, ensure_output_directories
from benchmark.io import image_paths, read_image

ensure_output_directories()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AnyCalib(model_id='anycalib_pinhole').to(device)

predictions = {}
input_paths = image_paths(DATASET_B_UNDISTORTED_DIR)

for img_path in input_paths:
    filename = img_path.name
    try:
        img_bgr = read_image(img_path)
    except ValueError as error:
        print(f"Skipping {error}")
        continue
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model.predict(img_tensor, cam_id="pinhole")

    intrinsics = output["intrinsics"][0].cpu().numpy()
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

    K_pred = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0, 0, 1]])
    predictions[filename] = K_pred

    print(f"AnyCalib processed: {filename}")

if not predictions:
    raise SystemExit("No AnyCalib predictions were produced.")

output_path = OUTPUT_DIR / "preds_anycalib.npz"
np.savez(output_path, **predictions)
print(f"Done! Predictions saved to '{output_path}'.")