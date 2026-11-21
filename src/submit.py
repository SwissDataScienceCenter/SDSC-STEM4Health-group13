"""Submit a prediction file to the Project 2 leaderboard.

    python src/submit.py <file.csv> --user "Team 13"
    python src/submit.py <file.csv> --dry-run          # check the file only

The file must match BRPMM_template.csv: columns batch_id, project, label; the same
201 batches; labels "accepted" / "rejected". Token (if any) is read from $BRPMM_TOKEN.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import INPUT_DIR  # noqa: E402

BASE_URL = "https://leaderboard-project2-runai-innovation-thibaut.inference.compute.datascience.ch/api/v1"
TEMPLATE = INPUT_DIR / "BRPMM_template.csv"


def load_checked(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    template = pd.read_csv(TEMPLATE, dtype=str)

    errors = []
    if list(df.columns) != list(template.columns):
        errors.append(f"columns {list(df.columns)}, expected {list(template.columns)}")
    else:
        if df["batch_id"].duplicated().any():
            errors.append(f"duplicate batch_id: {df.loc[df['batch_id'].duplicated(), 'batch_id'].tolist()}")
        missing = set(template["batch_id"]) - set(df["batch_id"])
        extra = set(df["batch_id"]) - set(template["batch_id"])
        if missing:
            errors.append(f"{len(missing)} batches missing, e.g. {sorted(missing)[:5]}")
        if extra:
            errors.append(f"{len(extra)} unknown batches, e.g. {sorted(extra)[:5]}")
        bad = set(df["label"].dropna()) - {"accepted", "rejected"}
        if bad or df["label"].isna().any():
            errors.append(f"invalid labels: {sorted(bad)}{' and empty values' if df['label'].isna().any() else ''}")
        merged = df.merge(template[["batch_id", "project"]], on="batch_id", suffixes=("", "_t"))
        if (merged["project"] != merged["project_t"]).any():
            errors.append("project column does not match the template")
    if errors:
        sys.exit(f"{path}: not a valid submission\n  - " + "\n  - ".join(errors))
    return df


def submit(df: pd.DataFrame, user: str, token: str = "") -> dict:
    response = requests.post(f"{BASE_URL}/submit", json={"predictions": df.to_dict(orient="records")},
                             auth=(user, token), timeout=60)
    if response.status_code // 100 != 2:
        sys.exit(f"submission failed (HTTP {response.status_code}): {response.text}")
    return response.json()["metrics"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="CSV with batch_id, project, label")
    p.add_argument("--user", help='team name as on the leaderboard, e.g. "Team 13"')
    p.add_argument("--dry-run", action="store_true", help="validate the file, don't submit")
    args = p.parse_args()

    df = load_checked(args.path)
    print(f"{args.path}: OK")
    print(df.groupby(["project", "label"]).size().to_string())

    if args.dry_run:
        return
    if not args.user:
        p.error("--user is required to submit")
    metrics = submit(df, args.user, os.environ.get("BRPMM_TOKEN", ""))
    for k, v in metrics.items():
        print(f"{k:<28} {v}")


if __name__ == "__main__":
    main()
