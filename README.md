# UMAHand Behaviour Analysis

Exploratory Python pipeline for analysing the UMAHand dataset: inertial signals from everyday hand-related activities recorded with a wrist-worn Shimmer sensor.

The initial goal of this repository is to build a clean and reproducible workflow for:

- loading and validating UMAHand trace files;
- summarising dataset structure and signal quality;
- extracting descriptive movement features;
- evaluating activity classification baselines;
- exploring whether hand-related everyday activities can be profiled along habit-like vs goal-directed dimensions.

## Dataset

This project uses the UMAHand dataset:

**UMAHand: A dataset of inertial signals of typical hand activities**

The dataset contains inertial recordings from 25 participants performing 29 hand-related everyday activities. Signals were recorded at 100 Hz using a wrist-worn Shimmer device with accelerometer, gyroscope, magnetometer and barometer channels.

The dataset itself is not included in this repository.

Expected local structure:

```text
data/
└── raw/
    └── UMAHand/
        ├── TRACES/
        ├── VIDEOS/
        ├── readme.txt
        ├── user_characteristics.txt
        ├── activity_description.txt
        └── sensor_orientation.jpg
Repository structure
umahand-behaviour-analysis/
├── data/                  # Local data only; not tracked by Git
├── outputs/               # Generated outputs; not tracked by Git
├── scripts/               # Executable analysis scripts
├── src/
│   └── umahand/           # Reusable project code
├── README.md
├── requirements.txt
└── .gitignore
Current status

This repository is in an early exploratory phase.

Initial planned steps:

Dataset ingestion and validation.
Trial-level summary generation.
Basic descriptive analysis.
Feature extraction from inertial signals.
Activity classification baselines.
Exploratory comparison of habit-like vs goal-directed tasks.
