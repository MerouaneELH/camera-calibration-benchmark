"""Common orchestration for image-based intrinsic-calibration models.

Model adapters provide only model-specific inference.  This module owns the
repeatable workflow: discover images, read them safely, collect matrices keyed
by filename, report recoverable failures, and persist one prediction artifact.
"""

from collections.abc import Callable
import json
from pathlib import Path

import cv2
import numpy as np

from .config import DATASET_B_DIR, OUTPUT_DIR, ensure_output_directories
from .io import file_digest, image_paths, read_image


PredictionFunction = Callable[[Path, np.ndarray], np.ndarray]


def run_model(
    name: str,
    predict: PredictionFunction,
    output_filename: str,
    input_dir: Path = DATASET_B_DIR,
) -> None:
    """Run one model adapter over a dataset and save its intrinsic matrices.

    Args:
        name: Human-readable model name used in progress and error messages.
        predict: Callback receiving an image path and BGR image and returning a
            finite ``3 x 3`` intrinsic matrix.
        output_filename: Filename written below the configured output directory.
        input_dir: Dataset directory. Defaults to original, distorted Dataset B.

    Raises:
        FileNotFoundError: If the input directory has no supported images.
        SystemExit: If every image fails and no prediction can be saved.
    """

    ensure_output_directories()
    predictions: dict[str, np.ndarray] = {}
    input_paths = image_paths(input_dir)

    for image_path in input_paths:
        try:
            image = read_image(image_path)
            prediction = np.asarray(predict(image_path, image))
            if prediction.shape != (3, 3) or not np.isfinite(prediction).all():
                raise ValueError("prediction is not a finite 3 x 3 matrix")
            predictions[image_path.name] = prediction
        except (ValueError, RuntimeError, KeyError) as error:
            print(f"Skipping {name}/{image_path.name}: {error}")
            continue
        print(f"{name} processed: {image_path.name}")

    if not predictions:
        raise SystemExit(f"No {name} predictions were produced.")

    output_path = OUTPUT_DIR / output_filename
    np.savez(output_path, **predictions)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "model": name,
                "input_directory": str(input_dir),
                "input_images": [
                    {"image": path.name, "sha256": file_digest(path)} for path in input_paths
                ],
                "prediction_images": sorted(predictions),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done! {name} predictions saved to '{output_path}'.")
