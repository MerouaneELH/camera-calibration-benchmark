"""Geometry helpers shared by benchmark evaluation stages."""

import cv2
import numpy as np


def perpendicular_distance_mm(rvec: np.ndarray, tvec: np.ndarray) -> float:
    """Return camera-center distance perpendicular to a board plane at Z=0."""
    rotation, _ = cv2.Rodrigues(rvec)
    camera_center = -rotation.T @ tvec.reshape(3, 1)
    return float(abs(camera_center[2, 0]) * 1000.0)
