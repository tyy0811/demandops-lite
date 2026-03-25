"""Script entrypoint for data download."""

import argparse
from pathlib import Path

import yaml

from demandops.data.adapters import get_adapter
from demandops.data.download import download_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    adapter_name = config.get("dataset", {}).get("adapter", "taxi")
    adapter = get_adapter(adapter_name)

    download_all(
        adapter=adapter,
        months=config["data"]["months"],
        raw_dir=Path(config["data"]["raw_dir"]),
        zones_path=Path(config["data"]["zones_path"]) if "zones_path" in config["data"] else None,
    )


if __name__ == "__main__":
    main()
