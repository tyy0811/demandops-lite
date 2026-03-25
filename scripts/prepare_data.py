"""Script entrypoint for data preparation."""
import argparse
from pathlib import Path

import yaml

from demandops.data.adapters import get_adapter
from demandops.data.prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    adapter_name = config.get("dataset", {}).get("adapter", "taxi")
    adapter = get_adapter(adapter_name)

    prepare(
        adapter=adapter,
        raw_dir=Path(config["data"]["raw_dir"]),
        processed_dir=Path(config["data"]["processed_dir"]),
        zone_universe_path=Path(config["artifacts"]["zone_universe_path"]),
        config=config,
    )


if __name__ == "__main__":
    main()
