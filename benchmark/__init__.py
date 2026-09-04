"""Reusable building blocks for the camera-calibration benchmark.

The package intentionally contains no command-line side effects.  Scripts under
``scripts/`` and ``models/`` use these modules to share filesystem conventions,
ChArUco board construction, image validation, calibration loading, and model
execution.  Keeping those policies here prevents the individual entry points
from silently drifting apart.
"""
