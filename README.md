# AI Camera Calibration Benchmark vs. ChArUco Ground Truth

Empirical comparison of monocular AI-based camera calibration models (AnyCalib, GeoCalib, Perspective Fields) against traditional ChArUco ground truth for millimeter-level localization accuracy.

## Project Structure
- `scripts/1_calibrate_dataset_A.py`: Computes reference intrinsic matrix ($K_{ref}, D_{ref}$) via OpenCV ChArUco.
- `scripts/2_process_dataset_B.py`: Computes 3D ground truth camera poses via PnP.
- `models/`: Inference scripts generating predicted $K$ matrices for each AI model.
- `scripts/evaluate_all.py`: Compares AI predictions against ground truth and reports 3D localization error (mm).

## Setup
```bash
pip install -r requirements.txt