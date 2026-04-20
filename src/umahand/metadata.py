from __future__ import annotations

from pathlib import Path

import pandas as pd

HANDEDNESS_LABELS = {
    0: "left-handed",
    1: "right-handed",
}

GENDER_LABELS = {
    0: "male",
    1: "female",
    2: "undefined_or_undisclosed",
}


def load_user_metadata(data_root: Path) -> pd.DataFrame:
    path = Path(data_root) / "user_characteristics.txt"
    if not path.exists():
        raise FileNotFoundError(f"User metadata file not found: '{path}'.")

    df = pd.read_csv(
        path,
        header=None,
        names=["user_id", "handedness_code", "gender_code", "weight_kg", "height_cm", "age_years"],
    )

    if df.shape[1] != 6:
        raise ValueError(f"User metadata should contain 6 columns, found {df.shape[1]}.")

    df["user_id"] = pd.to_numeric(df["user_id"], errors="raise").astype(int)
    df["handedness_code"] = pd.to_numeric(df["handedness_code"], errors="raise").astype(int)
    df["gender_code"] = pd.to_numeric(df["gender_code"], errors="raise").astype(int)
    for column in ["weight_kg", "height_cm", "age_years"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    df["handedness_label"] = df["handedness_code"].map(HANDEDNESS_LABELS).fillna("unknown")
    df["gender_label"] = df["gender_code"].map(GENDER_LABELS).fillna("unknown")

    return df[
        [
            "user_id",
            "handedness_code",
            "handedness_label",
            "gender_code",
            "gender_label",
            "weight_kg",
            "height_cm",
            "age_years",
        ]
    ].sort_values("user_id", ignore_index=True)


def load_activity_metadata(data_root: Path) -> pd.DataFrame:
    path = Path(data_root) / "activity_description.txt"
    if not path.exists():
        raise FileNotFoundError(f"Activity metadata file not found: '{path}'.")

    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["activity_id", "activity_name"],
        engine="python",
    )

    if df.shape[1] != 2:
        raise ValueError(f"Activity metadata should contain 2 columns, found {df.shape[1]}.")

    df["activity_id"] = pd.to_numeric(df["activity_id"], errors="raise").astype(int)
    df["activity_name"] = df["activity_name"].astype(str).str.strip()

    return df.sort_values("activity_id", ignore_index=True)
