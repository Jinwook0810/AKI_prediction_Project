from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aki_prediction.paths import build_paths
from aki_prediction.preprocess import run_preprocessing


def main() -> None:
    paths = build_paths(ROOT_DIR)

    print(f"Project root: {paths.root_dir}")
    print(f"Data dir: {paths.data_dir}")
    print(f"Raw csv.gz exists: {paths.raw_csv_gz.exists()}")
    print(f"Raw csv exists: {paths.raw_csv.exists()}")
    print(f"Split json exists: {paths.split_json.exists()}")

    artifacts = run_preprocessing(paths)

    print("")
    print("Generated artifacts")
    print(f"- hourly table: {artifacts.hourly_csv}")
    print(f"- filtered cohort: {artifacts.filtered_csv}")
    print(f"- filtered observed mask: {artifacts.filtered_observed_mask_csv}")
    print(f"- cohort summary: {artifacts.cohort_summary_json}")

    with artifacts.cohort_summary_json.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    print("")
    print("Cohort summary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
