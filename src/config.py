from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(CONFIG_PATH, "r") as file:
    CONFIG = yaml.safe_load(file)