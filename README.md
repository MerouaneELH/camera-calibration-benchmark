# AI Camera Calibration Benchmark vs. ChArUco Ground Truth

Empirical comparison of monocular AI-based camera calibration models (AnyCalib, GeoCalib, Perspective Fields) against traditional ChArUco ground truth for millimeter-level localization accuracy.

## Project Structure
- `scripts/1_calibrate_dataset_A.py`: Computes reference intrinsic matrix ($K_{ref}, D_{ref}$) via OpenCV ChArUco.
- `scripts/2_process_dataset_B.py`: Computes 3D ground truth camera poses via PnP.
- `models/`: Inference scripts generating predicted $K$ matrices for each AI model.
- `scripts/evaluate_all.py`: Compares AI predictions against ground truth and reports 3D localization error (mm).

## Setup
```bash
python -m pip install -r requirements.txt
python -m pip install -e AnyCalib
python -m pip install -e GeoCalib
```

Use Python 3.10 or newer. Put the private `.jpg` files in
`Data/dataset_A_reference` and `Data/dataset_B_evaluation`, then run these commands
from any working directory:

```bash
python scripts/1_calibrate_dataset_A.py
python scripts/undistort_dataset.py
python models/run_anycalib.py
python models/run_geocalib.py
python models/run_perspective.py
python scripts/evaluate_all.py
```

Calibration and prediction artifacts are written under `Outputs/`. Evaluation uses
the generated undistorted Dataset B images and zero distortion for predicted pinhole
matrices.