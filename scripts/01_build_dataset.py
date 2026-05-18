import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.preprocessing.build_multidataset import build_multidataset
from src.utils.io import read_yaml


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    build_multidataset(data_config)


if __name__ == "__main__":
    main()
