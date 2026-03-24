"""Script entrypoint for data download."""

from pathlib import Path

import yaml

from demandops.data.download import download_all


def main() -> None:
    config = yaml.safe_load(Path("configs/default.yaml").read_text())
    download_all(
        months=config["data"]["months"],
        raw_dir=Path(config["data"]["raw_dir"]),
        zones_path=Path(config["data"]["zones_path"]),
    )


if __name__ == "__main__":
    main()
