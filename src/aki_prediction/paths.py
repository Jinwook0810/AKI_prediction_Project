from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path
    data_dir: Path
    reports_dir: Path
    processed_dir: Path
    metrics_dir: Path
    figures_dir: Path
    raw_csv_gz: Path
    raw_csv: Path
    split_json: Path
    hourly_csv: Path
    filtered_csv: Path
    filtered_observed_mask_csv: Path
    cohort_summary_json: Path


def _resolve_data_dir(root_dir: Path) -> Path:
    candidates = [
        root_dir / "data",
        root_dir / "IMEN383_Team_Project_Files",
    ]

    for candidate in candidates:
        if not candidate.exists():
            continue

        has_raw = (candidate / "released_df.csv.gz").exists() or (
            candidate / "released_df.csv"
        ).exists()
        has_split = (candidate / "split_stay_id.json").exists()

        if has_raw or has_split:
            return candidate

    return root_dir / "data"


def build_paths(root_dir: Path | None = None) -> ProjectPaths:
    root = (root_dir or Path.cwd()).resolve()
    data_dir = _resolve_data_dir(root)

    reports_dir = root / "reports"
    processed_dir = reports_dir / "processed"
    metrics_dir = reports_dir / "metrics"
    figures_dir = reports_dir / "figures"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root_dir=root,
        data_dir=data_dir,
        reports_dir=reports_dir,
        processed_dir=processed_dir,
        metrics_dir=metrics_dir,
        figures_dir=figures_dir,
        raw_csv_gz=data_dir / "released_df.csv.gz",
        raw_csv=data_dir / "released_df.csv",
        split_json=data_dir / "split_stay_id.json",
        hourly_csv=processed_dir / "hourly_features.csv",
        filtered_csv=processed_dir / "filtered_cohort.csv",
        filtered_observed_mask_csv=processed_dir / "filtered_observed_mask.csv",
        cohort_summary_json=processed_dir / "cohort_summary.json",
    )
