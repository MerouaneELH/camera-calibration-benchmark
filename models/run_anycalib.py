"""Run AnyCalib pinhole inference on original Dataset B images.

The adapter converts each OpenCV BGR image to a normalized RGB tensor, asks
AnyCalib for ``fx, fy, cx, cy``, and returns those values as a standard ``3 x 3``
intrinsic matrix.  Shared iteration and persistence are implemented by
``benchmark.model_runner``.
"""

import sys
import torch
import numpy as np
import torchvision.transforms.functional as TF
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from anycalib import AnyCalib
from benchmark.model_runner import run_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AnyCalib(model_id="anycalib_pinhole").to(device)


def predict(_image_path: Path, img_bgr: np.ndarray) -> np.ndarray:
    """Predict a pinhole intrinsic matrix for one BGR image."""

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = TF.to_tensor(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.predict(img_tensor, cam_id="pinhole")
    intrinsics = output["intrinsics"][0].cpu().numpy()
    fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


run_model("AnyCalib", predict, "preds_anycalib.npz")