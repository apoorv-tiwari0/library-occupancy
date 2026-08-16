"""
config_loader.py — Load and expose config.yaml as a Python object with .env support.

Usage in any module:
    from config.config_loader import cfg
    print(cfg.api.port)
"""

import os
import re
import yaml
from pathlib import Path
from types import SimpleNamespace

# Load .env file from project root if present
_PROJECT_ROOT = Path(__file__).parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    if _ENV_PATH.exists():
        with open(_ENV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key not in os.environ:
                        os.environ[key] = val

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def _cast_type(val_str: str):
    """Cast a string value to appropriate Python types (int, float, bool, None)."""
    val_lower = val_str.strip().lower()
    if val_lower in ("null", "none", "~"):
        return None
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    if re.fullmatch(r"-?\d+", val_str.strip()):
        return int(val_str.strip())
    try:
        return float(val_str.strip())
    except ValueError:
        return val_str


def _interpolate_value(value):
    """Interpolate ${VAR:default} pattern in string values."""
    if not isinstance(value, str):
        return value

    match = _ENV_VAR_PATTERN.search(value)
    if not match:
        return value

    # If the string is EXACTLY `${VAR:default}`, preserve type casting
    if _ENV_VAR_PATTERN.fullmatch(value.strip()):
        var_name, default_val = match.group(1), match.group(2)
        env_val = os.environ.get(var_name)
        if env_val is not None:
            return _cast_type(env_val)
        if default_val is not None:
            return _cast_type(default_val)
        return None

    # Partial substitution in a longer string (e.g. RTSP URLs)
    def _subber(m):
        v_name, d_val = m.group(1), m.group(2)
        return os.environ.get(v_name, d_val if d_val is not None else "")

    return _ENV_VAR_PATTERN.sub(_subber, value)


def _interpolate_dict(d: dict) -> dict:
    """Recursively interpolate dict items."""
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = _interpolate_dict(v)
        elif isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    new_list.append(_interpolate_dict(item))
                elif isinstance(item, str):
                    new_list.append(_interpolate_value(item))
                else:
                    new_list.append(item)
            new_dict[k] = new_list
        elif isinstance(v, str):
            new_dict[k] = _interpolate_value(v)
        else:
            new_dict[k] = v
    return new_dict


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


def load_config(path: Path = _CONFIG_PATH) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"config.yaml loaded as empty/None. Check the file at: {path}")

    interpolated = _interpolate_dict(raw)
    return _dict_to_namespace(interpolated)


# Module-level singleton — import `cfg` directly everywhere
cfg = load_config()