"""Evaluate planar object dimensions using independent camera calibrations.

Original Dataset B images and the same manually annotated four object corners
are used for every method. ChArUco detections establish each method's metric
board plane; ray-plane intersection then reconstructs object length and width.
The setup height is a grouping variable only, never a geometric constraint.
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.config import CALIBRATION_PATH, DATASET_B_DIR, DATASET_B_METADATA_PATH, EVALUATION_VISUALIZATION_DIR, OUTPUT_DIR  # noqa: E402
from benchmark.geometry import board_plane_points_mm, object_dimensions_mm  # noqa: E402
from benchmark.io import create_charuco_detector, file_digest, image_paths, load_calibration, load_dataset_metadata, read_image  # noqa: E402

METHODS = {
    "opencv": "OpenCV Reference",
    "anycalib": "AnyCalib",
    "geocalib": "GeoCalib",
    "perspective": "Perspective Fields",
}


def valid_intrinsics(matrix: object) -> bool:
    """Return whether a matrix is a finite, valid pinhole intrinsic matrix."""
    matrix = np.asarray(matrix)
    return matrix.shape == (3, 3) and np.isfinite(matrix).all() and matrix[0, 0] > 0 and matrix[1, 1] > 0 and abs(matrix[2, 2] - 1) < 1e-6


def load_predictions(images: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    """Load fresh model predictions, retaining missing artifacts as failures."""
    current = {path.name: file_digest(path) for path in images}
    loaded = {}
    for model in METHODS:
        if model == "opencv":
            continue
        path = OUTPUT_DIR / f"preds_{model}.npz"
        if not path.exists():
            loaded[model] = {}
            continue
        manifest_path = path.with_suffix(".json")
        if not manifest_path.exists():
            raise ValueError(f"Missing manifest for {path.name}; regenerate predictions")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = {entry["image"]: entry["sha256"] for entry in manifest.get("input_images", [])}
        if recorded != current:
            raise ValueError(f"Stale prediction artifact for {model}; regenerate {path.name}")
        with np.load(path) as data:
            loaded[model] = {name: data[name] for name in data.files}
    return loaded


def estimate_dimensions(obj_points, img_points, matrix, distortion, object_pixels):
    """Estimate object dimensions by solving the board pose and intersecting rays."""
    if not valid_intrinsics(matrix):
        return None
    try:
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, matrix, distortion)
        if not success:
            return None
        plane_points = board_plane_points_mm(object_pixels, matrix, distortion, rvec, tvec)
        return object_dimensions_mm(plane_points)
    except cv2.error:
        return None


def dimension_metrics(results: pd.DataFrame, model: str, dimension: str) -> dict[str, float]:
    """Return aggregate dimensional error and repeatability statistics."""
    errors = results[f"{model}_{dimension}_absolute_error_mm"].dropna().astype(float)
    percentages = results[f"{model}_{dimension}_percentage_error"].dropna().astype(float)
    if errors.empty:
        return {key: float("nan") for key in ("mae_mm", "rmse_mm", "median_absolute_error_mm", "max_absolute_error_mm", "p95_absolute_error_mm", "mean_percentage_error", "median_percentage_error", "std_error_mm")}
    return {
        "mae_mm": float(errors.mean()),
        "rmse_mm": float(np.sqrt(np.mean(errors**2))),
        "median_absolute_error_mm": float(errors.median()),
        "max_absolute_error_mm": float(errors.max()),
        "p95_absolute_error_mm": float(errors.quantile(0.95)),
        "mean_percentage_error": float(percentages.mean()),
        "median_percentage_error": float(percentages.median()),
        "std_error_mm": float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
    }


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Build global and setup-height grouped summaries from actual metadata."""
    groups = [("global", "all", results)]
    groups.extend(("experimental_setup_height_mm", value, group) for value, group in results.groupby("experimental_setup_height_mm", dropna=False))
    rows = []
    for group_by, group_value, group in groups:
        for model, label in METHODS.items():
            for dimension in ("length", "width"):
                rows.append({"group_by": group_by, "group_value": group_value, "method": label, "dimension": dimension, **dimension_metrics(group, model, dimension)})
    return pd.DataFrame(rows)


def save_plot(figure, filename: str) -> None:
    """Save one evaluation plot and release its Matplotlib resources."""
    EVALUATION_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(EVALUATION_VISUALIZATION_DIR / filename, dpi=150)
    plt.close(figure)


def make_plots(results: pd.DataFrame) -> None:
    """Create required dimensional plots grouped by measured setup height."""
    for dimension in ("length", "width"):
        for metric, label in (("absolute_error_mm", "Absolute error (mm)"), ("percentage_error", "Absolute percentage error (%)")):
            figure, axis = plt.subplots()
            for model, method in METHODS.items():
                axis.scatter(results["experimental_setup_height_mm"], results[f"{model}_{dimension}_{metric}"], label=method, alpha=0.7)
            axis.set(xlabel="Experimental setup height (mm)", ylabel=f"{dimension.title()} {label}")
            axis.legend()
            save_plot(figure, f"{dimension}_{metric}_vs_setup_height.png")

        figure, axis = plt.subplots()
        values, labels = [], []
        for model, method in METHODS.items():
            errors = results[f"{model}_{dimension}_signed_error_mm"].dropna()
            if not errors.empty:
                values.append(errors)
                labels.append(method)
        if values:
            axis.boxplot(values, tick_labels=labels)
        else:
            axis.text(0.5, 0.5, "No valid measurements", ha="center", va="center")
        axis.set_ylabel(f"{dimension.title()} signed error (mm)")
        save_plot(figure, f"{dimension}_error_boxplot.png")

        figure, axis = plt.subplots()
        for model, method in METHODS.items():
            axis.scatter(results[f"true_{dimension}_mm"], results[f"{model}_estimated_{dimension}_mm"], label=method, alpha=0.7)
        axis.set(xlabel=f"True {dimension} (mm)", ylabel=f"Estimated {dimension} (mm)")
        axis.legend()
        save_plot(figure, f"estimated_{dimension}_vs_true.png")


def main() -> None:
    """Validate inputs, reconstruct dimensions, and write results and plots."""
    images = image_paths(DATASET_B_DIR)
    metadata = load_dataset_metadata(DATASET_B_METADATA_PATH, images)
    k_ref, distortion = load_calibration(CALIBRATION_PATH)
    predictions = load_predictions(images)
    matrices = {"opencv": {path.name: k_ref for path in images}, **predictions}
    board, detector = create_charuco_detector()
    rows = []

    for image_path in images:
        name = image_path.name
        info = metadata[name]
        row = {"filename": name, "experimental_setup_height_mm": float(info["experimental_setup_height_mm"]), "true_length_mm": float(info["true_length_mm"]), "true_width_mm": float(info["true_width_mm"]), "status": "ok"}
        object_pixels = np.asarray([[float(info[f"p{index}_{axis}"]) for axis in ("x", "y")] for index in range(1, 5)], dtype=np.float64)
        try:
            image = read_image(image_path)
            corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            if corners is None or ids is None or len(corners) < 4:
                raise ValueError("insufficient ChArUco corners")
            obj_points, img_points = board.matchImagePoints(corners, ids)
        except (ValueError, cv2.error) as error:
            row["status"] = str(error)
            obj_points = img_points = None

        for model, model_predictions in matrices.items():
            matrix = model_predictions.get(name)
            dimensions = estimate_dimensions(obj_points, img_points, matrix, distortion, object_pixels) if obj_points is not None and matrix is not None else None
            estimated_length, estimated_width = dimensions if dimensions is not None else (np.nan, np.nan)
            row[f"{model}_estimated_length_mm"] = estimated_length
            row[f"{model}_estimated_width_mm"] = estimated_width
            row[f"{model}_status"] = "ok" if dimensions is not None else ("missing_prediction" if matrix is None else "reconstruction_failed")
        rows.append(row)

    results = pd.DataFrame(rows)
    for model in METHODS:
        for dimension in ("length", "width"):
            true_column = f"true_{dimension}_mm"
            estimate_column = f"{model}_estimated_{dimension}_mm"
            signed_column = f"{model}_{dimension}_signed_error_mm"
            absolute_column = f"{model}_{dimension}_absolute_error_mm"
            results[signed_column] = results[estimate_column] - results[true_column]
            results[absolute_column] = results[signed_column].abs()
            results[f"{model}_{dimension}_percentage_error"] = results[absolute_column] / results[true_column] * 100
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
    build_summary(results).to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)
    make_plots(results)
    print(f"Dataset B images: {len(images)}; metadata rows: {len(metadata)}; evaluated: {len(results)}")
    for model, label in METHODS.items():
        print(f"{label}: length successes={results[f'{model}_estimated_length_mm'].notna().sum()}, width successes={results[f'{model}_estimated_width_mm'].notna().sum()}")


if __name__ == "__main__":
    main()
