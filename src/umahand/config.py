from __future__ import annotations

from pathlib import Path

TRACE_COLUMNS = [
    "Timestamp",
    "Ax",
    "Ay",
    "Az",
    "Gx",
    "Gy",
    "Gz",
    "Mx",
    "My",
    "Mz",
    "P",
]

ACC_COLUMNS = ["Ax", "Ay", "Az"]
GYRO_COLUMNS = ["Gx", "Gy", "Gz"]
MAG_COLUMNS = ["Mx", "My", "Mz"]
SENSOR_COLUMNS = ACC_COLUMNS + GYRO_COLUMNS + MAG_COLUMNS + ["P"]

EXPECTED_SUBJECT_COUNT = 25
EXPECTED_ACTIVITY_COUNT = 29
EXPECTED_TRIAL_COUNT = 752
EXPECTED_SAMPLING_HZ = 100.0
SAMPLING_TOLERANCE_HZ = 5.0


def default_output_dir() -> Path:
    return Path("outputs") / "dataset_summary"


def default_features_output_dir() -> Path:
    return Path("outputs") / "features"
