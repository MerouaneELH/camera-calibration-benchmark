# AI Camera Calibration Benchmark

This project compares monocular camera-calibration models with a traditional
OpenCV ChArUco calibration. It measures how accurately each predicted pinhole
camera reconstructs the dimensions of a known planar object in millimeters.

The original Dataset A and Dataset B images were preliminary/random images
without documented physical acquisition conditions. They are being replaced by
a newly captured and documented dataset.

This documentation covers only the benchmark-owned files in the repository
root: `benchmark/`, `models/`, `scripts/`, and the root configuration files.
The cloned `AnyCalib/` and `GeoCalib/` model repositories are external model
implementations and are intentionally excluded from this file-by-file guide.

## Architecture

The project is split into three responsibilities:

- `benchmark/` owns shared configuration, validated I/O, ChArUco construction,
	and model-runner orchestration.
- `models/` contains one thin adapter per cloned calibration model. Each adapter
	converts that model's output into a common 3 x 3 intrinsic matrix.
- `scripts/` contains command-line stages for reference calibration,
	undistortion, diagnostic PnP processing, and evaluation.

The shared modules follow single-responsibility principles. Dataset paths and
board dimensions are defined once in `benchmark/config.py`; image discovery and
calibration loading are defined once in `benchmark/io.py`; repeated model-loop
behavior is defined once in `benchmark/model_runner.py`.

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
python models/run_anycalib.py
python models/run_geocalib.py
python models/run_perspective.py
python scripts/evaluate_all.py
```

Run `python scripts/undistort_dataset.py` separately only for diagnostic output;
it is not part of the primary benchmark path.

Calibration and prediction artifacts are written under `Outputs/`. Primary evaluation
uses the original distorted Dataset B images. Undistortion remains available only
for diagnostics and is not part of the primary model-evaluation pipeline.

## Data Contract

Place private images in:

```text
Data/
	dataset_A_reference/       # ChArUco images used to estimate K and D
	dataset_B_evaluation/      # original distorted evaluation images
	dataset_B_undistorted/     # optional diagnostic output
```

Supported image extensions are `.jpg`, `.jpeg`, and `.png`. Dataset A and Dataset
B must use the same physical camera, image resolution, lens configuration, and
printed ChArUco board. The shared board configuration is:

- 5 squares by 7 squares
- square length: 0.038 m
- marker length: 0.029 m
- dictionary: `DICT_5X5_100`

Changing any physical setting requires regenerating the calibration, undistorted
images, predictions, and evaluation CSV.

## Pipeline

Run the stages in this order:

```text
Dataset A -> reference calibration
Dataset B original -> model predictions -> ChArUco PnP -> evaluation CSV
       \-> optional diagnostic undistortion
```

```powershell
python scripts/1_calibrate_dataset_A.py
python models/run_anycalib.py
python models/run_geocalib.py
python models/run_perspective.py
python scripts/evaluate_all.py
```

All paths are resolved from the repository location, so these commands do not
depend on the current working directory.

## Dataset B Metadata

Create `Data/dataset_B_evaluation/metadata.csv` after capturing the new images.
The CSV is the source of truth and must contain exactly one row per Dataset B
image:

```csv
image,experimental_setup_height_mm,true_length_mm,true_width_mm,p1_x,p1_y,p2_x,p2_y,p3_x,p3_y,p4_x,p4_y
B_001.jpg,<measured_height>,<measured_length>,<measured_width>,<p1_x>,<p1_y>,<p2_x>,<p2_y>,<p3_x>,<p3_y>,<p4_x>,<p4_y>
B_002.jpg,<measured_height>,<measured_length>,<measured_width>,<p1_x>,<p1_y>,<p2_x>,<p2_y>,<p3_x>,<p3_y>,<p4_x>,<p4_y>
```

The object corners must be ordered P1 top-left, P2 top-right, P3 bottom-right,
P4 bottom-left. `experimental_setup_height_mm` is only the measured camera or
tripod setup height relative to the supporting surface; it is not camera-to-board
distance and is never used as a geometric constraint. `true_length_mm` and
`true_width_mm` must come from an independent physical measurement. The
evaluator rejects missing rows, duplicate rows, nonexistent filenames,
non-positive measurements, and invalid numeric values. It never invents values
from filenames or image geometry.

## Evaluation Methodology

Primary evaluation uses original Dataset B images and identical manual corners:

```text
original Dataset B -> calibration model -> predicted K
				   -> ChArUco plane + manual object corners
				   -> ray-plane reconstruction -> estimated dimensions
				   -> comparison with true_length_mm and true_width_mm
```

`experimental_setup_height_mm` is a grouping variable only. The physical ground
truth for the primary experiment is the independently measured object size.
Each method obtains its own intrinsic matrix. ChArUco establishes the metric
plane, and calibrated ray-plane intersections reconstruct the object corners in
millimeters. No pixels-to-millimeters shortcut is used.

The evaluator writes signed, absolute, and percentage length/width errors and
calculates MAE, RMSE, median, maximum, 95th percentile, mean percentage error,
and standard deviation globally and for each observed setup height.

## Root-Owned File Guide

### `benchmark/`

- `benchmark/__init__.py`: Defines the shared benchmark package and documents
	that importing it has no command-line side effects.
- `benchmark/config.py`: Defines repository-relative data, output, and
	visualization paths. It also defines the immutable ChArUco board settings and
	creates output directories when requested.
- `benchmark/io.py`: Provides sorted image discovery, validated OpenCV loading,
	shared ChArUco detector construction, and calibration artifact loading.
- `benchmark/model_runner.py`: Provides the common model execution template.
	It reads images, calls a model-specific callback, skips recoverable per-image
	errors, and saves filename-keyed intrinsic matrices to `.npz`.
- `benchmark/geometry.py`: Defines the shared perpendicular camera-center to
  board-plane distance conversion in millimeters.

### `models/`

- `models/run_anycalib.py`: Converts BGR images to normalized RGB tensors,
	invokes AnyCalib's pinhole predictor, and converts `fx, fy, cx, cy` into a
	standard intrinsic matrix.
- `models/run_geocalib.py`: Invokes GeoCalib, extracts its pinhole matrix, and
	saves gravity and latitude diagnostic overlays.
- `models/run_perspective.py`: Converts Perspective Fields' vertical field of
	view and relative principal point into pixel focal length and principal point,
	then saves a perspective-field overlay.

Each model adapter delegates iteration and persistence to
`benchmark.model_runner.py`; model-specific code should stay inside its adapter.

### `scripts/`

- `scripts/1_calibrate_dataset_A.py`: Detects ChArUco corners in Dataset A and
	estimates reference intrinsics `K` and distortion coefficients `D` with
	`cv2.calibrateCamera`.
- `scripts/undistort_dataset.py`: Applies the reference `K` and `D` to Dataset
	B and preserves input filenames in the undistorted directory.
- `scripts/2_process_dataset_B.py`: Performs diagnostic PnP processing on the
	original distorted Dataset B images using the measured distortion model. It
	reports board translation distance and does not create model predictions.
- `scripts/evaluate_all.py`: Detects ChArUco points in original Dataset B,
	estimates reference and predicted PnP distances, computes millimeter and focal
	length errors, and writes the evaluation CSV.

## Generated Outputs

- `Outputs/reference_calibration.npz`: `K`, `D`, image size, and board metadata.
- `Outputs/preds_anycalib.npz`: AnyCalib matrices keyed by filename.
- `Outputs/preds_geocalib.npz`: GeoCalib matrices keyed by filename.
- `Outputs/preds_perspective.npz`: Perspective Fields matrices keyed by filename.
- `Outputs/evaluation_results.csv`: Per-image metadata, distances, signed and
	absolute errors, focal errors, and failure statuses.
- `Outputs/evaluation_summary.csv`: Global, viewpoint-angle, and physical-
	distance grouped metrics.
- `visualizations/evaluation/`: Error-versus-condition, method boxplot, and
	predicted/reference-versus-physical-distance plots.
- `visualizations/geocalib/`: GeoCalib diagnostic images.
- `visualizations/perspective_fields/`: Perspective Fields diagnostic images.

Prediction files use image filenames as keys instead of relying on array order.
This prevents mismatching a prediction with another image when files are added,
removed, or sorted differently.

## Setup

Use Python 3.10 or newer and the existing project virtual environment when
available:

```powershell
.\calib_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e AnyCalib
python -m pip install -e GeoCalib
```

Confirm that the active interpreter is the intended environment:

```powershell
python -c "import sys; print(sys.executable)"
```

The model adapters require the cloned model packages and their dependencies to
be installed in that interpreter. GPU inference is recommended; the adapters
fall back to CPU when CUDA is unavailable.

## Failure Behavior

The shared utilities fail early when a required dataset or calibration artifact
is missing. Individual unreadable images are reported and skipped. A model run
fails if no valid predictions are produced. Evaluation rejects malformed or
non-finite intrinsic matrices and failed PnP results rather than calculating
misleading metrics.

If the reference reprojection error is substantially above one pixel, inspect
board visibility, focus, board measurements, and detected corners
before trusting the benchmark. Large model errors also require checking that all
models consumed the original `dataset_B_evaluation` images and that Dataset A and Dataset B came
from the same camera setup.