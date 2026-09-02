from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .config import DATASET_B_UNDISTORTED_DIR, OUTPUT_DIR, ensure_output_directories
from .io import image_paths, read_image


PredictionFunction = Callable[[Path, np.ndarray], np.ndarray]


def run_model(
    name: str,
    predict: PredictionFunction,
    output_filename: str,
    input_dir: Path = DATASET_B_UNDISTORTED_DIR,
) -> None:
    ensure_output_directories()
    predictions: dict[str, np.ndarray] = {}

    for image_path in image_paths(input_dir):
        try:
            image = read_image(image_path)
            predictions[image_path.name] = predict(image_path, image)
        except (ValueError, RuntimeError, KeyError) as error:
            print(f"Skipping {name}/{image_path.name}: {error}")
            continue
        print(f"{name} processed: {image_path.name}")

    if not predictions:
        raise SystemExit(f"No {name} predictions were produced.")

    output_path = OUTPUT_DIR / output_filename
    np.savez(output_path, **predictions)
    print(f"Done! {name} predictions saved to '{output_path}'.")
