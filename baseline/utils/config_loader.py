# utils/config_loader.py
import yaml
from pathlib import Path

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config