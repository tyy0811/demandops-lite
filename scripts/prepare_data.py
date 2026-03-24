"""Script entrypoint for data preparation."""

from pathlib import Path

import yaml

from demandops.data.prepare import prepare


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    prepare(
        raw_dir=Path(config["data"]["raw_dir"]),
        processed_dir=Path(config["data"]["processed_dir"]),
        zones_path=Path(config["data"]["zones_path"]),
        zone_universe_path=Path(config["artifacts"]["zone_universe_path"]),
        months=config["data"]["months"],
        lag_hours=config["features"]["lag_hours"],
        rolling_windows=config["features"]["rolling_windows"],
    )


if __name__ == "__main__":
    main()
