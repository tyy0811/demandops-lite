"""Generate frozen test fixture for the MAE regression gate.

One-time script. Output is committed to repo and never regenerated
unless the model/features are intentionally changed.
"""

from pathlib import Path

import polars as pl
import yaml

from demandops.data.splits import split_from_config
from demandops.features import FEATURE_COLUMNS, TARGET_COLUMN


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    features_path = Path(config["data"]["processed_dir"]) / "features.parquet"
    df = pl.read_parquet(features_path)

    _, _, test = split_from_config(df, config)

    # Select 20 zones across all test hours
    zone_ids = sorted(test["zone_id"].unique().to_list())
    selected_zones = zone_ids[:20]

    fixture = test.filter(pl.col("zone_id").is_in(selected_zones))

    # Keep only columns needed: FEATURE_COLUMNS + TARGET_COLUMN
    fixture = fixture.select(FEATURE_COLUMNS + [TARGET_COLUMN])

    out_dir = Path("data/test_fixtures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "regression_test.parquet"
    fixture.write_parquet(out_path)

    print(f"Fixture saved: {out_path}")
    print(f"  Rows: {len(fixture)}")
    print(f"  Zones: {len(selected_zones)}")
    print(f"  Columns: {fixture.columns}")


if __name__ == "__main__":
    main()
