"""
Configuration loading and management.
"""

import yaml
import json
import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration manager."""

    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config)

    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        config = {}
        for key, value in os.environ.items():
            if key.startswith("FLASHSERVE_"):
                config_key = key[len("FLASHSERVE_"):].lower()
                config[config_key] = value
        return cls(config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.config
