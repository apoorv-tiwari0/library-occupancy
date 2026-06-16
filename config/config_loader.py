"""
config_loader.py — Load and expose config.yaml as a Python object.

Usage in any module:
    from config.config_loader import cfg
    print(cfg.api.port)
"""

import yaml
from pathlib import Path
from types import SimpleNamespace


_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to SimpleNamespace for dot-access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            setattr(ns, key, [_dict_to_namespace(item) for item in value])
        else:
            setattr(ns, key, value)
    return ns


# def load_config(path: Path = _CONFIG_PATH) -> SimpleNamespace:
#     with open(path, "r") as f:
#         raw = yaml.safe_load(f)
#     return _dict_to_namespace(raw)

def load_config(path: Path = _CONFIG_PATH) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"config.yaml loaded as empty/None. Check the file at: {path}")
    return _dict_to_namespace(raw)


# Module-level singleton — import `cfg` directly everywhere
cfg = load_config()