"""
config_loader.py
Loads and validates the config.json configuration file.
"""

import json
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Optional path to config file. Defaults to config/config.json.

    Returns:
        dict: Configuration dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = json.load(f)

    logger.info(f"Configuration loaded from: {path}")
    _validate_config(config)
    return config


def _validate_config(config: dict):
    """Basic validation to catch obvious misconfigurations."""
    required_keys = ["video", "detection", "recognition", "tracking", "logging", "database"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: '{key}'")

    skip = config["detection"].get("skip_frames", 0)
    if not isinstance(skip, int) or skip < 0:
        raise ValueError("detection.skip_frames must be a non-negative integer")

    threshold = config["recognition"].get("embedding_similarity_threshold", 0.45)
    if not (0.0 < threshold < 1.0):
        raise ValueError("embedding_similarity_threshold must be between 0 and 1")
