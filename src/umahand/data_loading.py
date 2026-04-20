from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import TRACE_COLUMNS

TRACE_FILENAME_RE = re.compile(
    r"^user_(?P<user_id>\d+)_activity_(?P<activity_id>\d+)_trial_(?P<trial_id>\d+)\.csv$"
)


class TraceLoadError(RuntimeError):
    """Raised when a trace file cannot be parsed safely."""

    def __init__(self, message: str, *, path: Path, n_columns: int | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.n_columns = n_columns


@dataclass(frozen=True)
class TraceFileInfo:
    path: Path
    relative_path: Path
    user_id: int
    activity_id: int
    trial_id: int


def parse_trace_filename(path: Path, *, data_root: Path | None = None) -> TraceFileInfo:
    match = TRACE_FILENAME_RE.match(path.name)
    if not match:
        raise TraceLoadError(
            (
                "Unexpected trace filename format. Expected "
                "'user_XX_activity_YY_trial_ZZ.csv' but found "
                f"'{path.name}'."
            ),
            path=path,
        )

    relative_path = path.relative_to(data_root) if data_root and path.is_relative_to(data_root) else path
    return TraceFileInfo(
        path=path,
        relative_path=relative_path,
        user_id=int(match.group("user_id")),
        activity_id=int(match.group("activity_id")),
        trial_id=int(match.group("trial_id")),
    )


def find_trace_files(data_root: Path) -> list[Path]:
    traces_root = Path(data_root) / "TRACES"
    if not traces_root.exists():
        raise FileNotFoundError(f"TRACES directory not found under '{data_root}'.")

    files = sorted(traces_root.glob("output_*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV traces found under '{traces_root}'.")
    return files


def load_trace_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, header=None)
    except pd.errors.EmptyDataError as exc:
        raise TraceLoadError("Trace file is empty.", path=path, n_columns=0) from exc

    if df.empty:
        raise TraceLoadError("Trace file is empty.", path=path, n_columns=df.shape[1])

    n_columns = df.shape[1]
    if n_columns != len(TRACE_COLUMNS):
        raise TraceLoadError(
            f"Trace file has {n_columns} columns; expected {len(TRACE_COLUMNS)}.",
            path=path,
            n_columns=n_columns,
        )

    df.columns = TRACE_COLUMNS
    df = df.apply(pd.to_numeric, errors="coerce")

    if df["Timestamp"].isna().all():
        raise TraceLoadError("Timestamp column could not be parsed as numeric.", path=path, n_columns=n_columns)

    if df["Timestamp"].isna().any():
        warnings.warn(f"{path}: timestamp column contains missing or non-numeric values.", stacklevel=2)

    return df
