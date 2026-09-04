"""Evaluate original Dataset B images against physical distance metadata.

The supplied metadata is independent physical ground truth. The reference
calibration and each predicted intrinsic matrix solve PnP on original,
distorted images. Distance means perpendicular camera-to-board-plane distance.
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

from benchmark.config import (  # noqa: E402
    CALIBRATION_PATH,
    DATASET_B_DIR,
    DATASET_B_METADATA_PATH,
    EVALUATION_VISUALIZATION_DIR,
    OUTPUT_DIR,
)
from benchmark.io import (  # noqa: E402
    create_charuco_detector,
    file_digest,
    image_paths,
    load_calibration,
    load_dataset_metadata,
    read_image,
)
from benchmark.geometry import perpendicular_distance_mm  # noqa: E402

METHODS = {
    "reference": "ChArUco Reference",
    "anycalib": "AnyCalib",
    "geocalib": "GeoCalib",
    "perspective": "Perspective Fields",
}


def valid_intrinsics(matrix: object) -> bool:
    """Return whether a matrix satisfies the minimum pinhole-camera contract."""
    matrix = np.asarray(matrix)
    return (
        matrix.shape == (3, 3)
        and np.isfinite(matrix).all()
        and matrix[0, 0] > 0
        and matrix[1, 1] > 0
        and abs(matrix[2, 2] - 1) < 1e-6
    )


def load_predictions(images: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    """Load fresh filename-keyed predictions; absent artifacts remain explicit."""
    current = {path.name: file_digest(path) for path in images}
    loaded: dict[str, dict[str, np.ndarray]] = {}
    for model in ("anycalib", "geocalib", "perspective"):
        path = OUTPUT_DIR / f"preds_{model}.npz"
        if not path.is_file():
            loaded[model] = {}
            continue
        manifest_path = path.with_suffix(".json")
        if not manifest_path.is_file():
            raise ValueError(f"Missing prediction manifest for {path.name}; regenerate predictions")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = {entry["image"]: entry["sha256"] for entry in manifest.get("input_images", [])}
        if recorded != current:
            raise ValueError(f"Stale prediction artifact for {model}; regenerate {path.name}")
        with np.load(path) as data:
            loaded[model] = {name: data[name] for name in data.files}
    return loaded


def solve_distance(obj_points, img_points, matrix, distortion) -> float | None:
    """Solve PnP and return perpendicular board-plane distance in millimeters."""
    if not valid_intrinsics(matrix):
        return None
    try:
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, matrix, distortion)
    except cv2.error:
        return None
    return perpendicular_distance_mm(rvec, tvec) if success else None


def metric_rows(values: pd.Series) -> dict[str, float]:
    """Calculate summary statistics from absolute errors, ignoring failures."""
    values = values.dropna().astype(float)
    if values.empty:
        return {key: float("nan") for key in ("mae_mm", "rmse_mm", "median_error_mm", "max_error_mm", "p95_error_mm")}
    return {
        "mae_mm": float(values.mean()),
        "rmse_mm": float(np.sqrt(np.mean(values**2))),
        "median_error_mm": float(values.median()),
        "max_error_mm": float(values.max()),
        "p95_error_mm": float(values.quantile(0.95)),
    }


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Create global and observed-angle/distance grouped summaries."""
    groups = [("global", "all", results)]
    for column in ("viewpoint_angle_deg", "physical_distance_mm"):
        groups.extend((column, value, group) for value, group in results.groupby(column, dropna=False))
    rows = []
    for group_by, group_value, group in groups:
        for model, label in METHODS.items():
            rows.append({"group_by": group_by, "group_value": group_value, "method": label, **metric_rows(group[f"{model}_error_mm"])})
    return pd.DataFrame(rows)


def save_plot(figure, filename: str) -> None:
    """Save and close one evaluation plot."""
    EVALUATION_VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(EVALUATION_VISUALIZATION_DIR / filename, dpi=150)
    plt.close(figure)


def make_plots(results: pd.DataFrame) -> None:
    """Generate requested physical-distance plots with missing values ignored."""
    for filename, x_column, xlabel in (
        ("error_vs_viewpoint_angle.png", "viewpoint_angle_deg", "Viewpoint angle (deg)"),
        ("error_vs_physical_distance.png", "physical_distance_mm", "Physical distance (mm)"),
    ):
        figure, axis = plt.subplots()
        for model, label in METHODS.items():
            axis.scatter(results[x_column], results[f"{model}_error_mm"], label=label, alpha=0.7)
        axis.set(xlabel=xlabel, ylabel="Absolute distance error (mm)")
        axis.legend()
        save_plot(figure, filename)

    figure, axis = plt.subplots()
    values, labels = [], []
    for model, label in METHODS.items():
        errors = results[f"{model}_error_mm"].dropna()
        if not errors.empty:
            values.append(errors)
            labels.append(label)
    if values:
        axis.boxplot(values, tick_labels=labels)
    else:
        axis.text(0.5, 0.5, "No valid distance errors", ha="center", va="center")
    axis.set_ylabel("Absolute distance error (mm)")
    save_plot(figure, "method_error_boxplot.png")

    figure, axis = plt.subplots()
    for model, label in METHODS.items():
        axis.scatter(results["physical_distance_mm"], results[f"{model}_distance_mm"], label=label, alpha=0.7)
    axis.set(xlabel="Physical distance (mm)", ylabel="Predicted/reference distance (mm)")
    axis.legend()
    save_plot(figure, "predicted_vs_physical_distance.png")

    figure, axis = plt.subplots()
    axis.scatter(results["physical_distance_mm"], results["reference_distance_mm"], label=METHODS["reference"], alpha=0.7)
    axis.set(xlabel="Physical distance (mm)", ylabel="Reference distance (mm)")
    axis.legend()
    save_plot(figure, "reference_vs_physical_distance.png")


def main() -> None:
    """Validate metadata, evaluate every image, and write CSVs and plots."""
    images = image_paths(DATASET_B_DIR)
    metadata = load_dataset_metadata(DATASET_B_METADATA_PATH, images)
    k_ref, distortion = load_calibration(CALIBRATION_PATH)
    predictions = load_predictions(images)
    board, detector = create_charuco_detector()
    rows = []

    for image_path in images:
        name = image_path.name
        info = metadata[name]
        row = {
            "image": name,
            "physical_distance_mm": float(info["physical_distance_mm"]),
            "viewpoint_angle_deg": float(info["viewpoint_angle_deg"]),
            "horizontal_position": info["horizontal_position"].strip(),
            "vertical_position": (info.get("vertical_position") or "").strip(),
        }
        obj_points = img_points = None
        try:
            image = read_image(image_path)
            corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            if corners is None or ids is None or len(corners) < 4:
                raise ValueError("insufficient ChArUco corners")
            obj_points, img_points = board.matchImagePoints(corners, ids)
            reference = solve_distance(obj_points, img_points, k_ref, distortion)
            row["reference_distance_mm"] = reference
            row["reference_status"] = "ok" if reference is not None else "pnp_failed"
        except (ValueError, cv2.error) as error:
            row["reference_distance_mm"] = np.nan
            row["reference_status"] = str(error)

        for model, values in predictions.items():
            matrix = values.get(name)
            distance = solve_distance(obj_points, img_points, matrix, distortion) if obj_points is not None and matrix is not None else None
            row[f"{model}_distance_mm"] = distance
            row[f"{model}_status"] = "ok" if distance is not None else ("missing_prediction" if matrix is None else "pnp_failed_or_invalid_K")
            row[f"{model}_focal_error_pct"] = (
                abs(np.mean(np.diag(matrix)[:2]) - np.mean(np.diag(k_ref)[:2]))
                / np.mean(np.diag(k_ref)[:2])
                * 100
                if matrix is not None and valid_intrinsics(matrix)
                else np.nan
            )
        rows.append(row)

    results = pd.DataFrame(rows)
    for model in METHODS:
        results[f"{model}_signed_error_mm"] = results[f"{model}_distance_mm"] - results["physical_distance_mm"]
        results[f"{model}_error_mm"] = results[f"{model}_signed_error_mm"].abs()
    results.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
    build_summary(results).to_csv(OUTPUT_DIR / "evaluation_summary.csv", index=False)
    make_plots(results)
    print(f"Dataset B images: {len(images)}; metadata rows: {len(metadata)}; evaluated: {len(results)}")
    for model, label in METHODS.items():
        success_count = int(results[f"{model}_distance_mm"].notna().sum())
        print(f"{label} successful predictions: {success_count}; failures: {len(results) - success_count}")


if __name__ == "__main__":
    main()
