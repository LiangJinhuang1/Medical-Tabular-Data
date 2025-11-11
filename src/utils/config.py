import yaml
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent

def load_config(config_path):
    """Load YAML configuration file"""
    config_path = Path(config_path)
    
    # If relative path, resolve it relative to project root
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    
    # Ensure the path exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config_value(config, *keys, default=None):
    """Helper to safely get nested config values"""
    for key in keys:
        config = config.get(key, {})
    return config if isinstance(config, dict) else (config if config is not None else default)


def resolve_path(file_path, base_dir=None):
    """
    Resolve file path flexibly.
    
    Args:
        file_path: Path string (can be absolute, relative, or None)
        base_dir: Base directory for relative paths (defaults to PROJECT_ROOT)
    
    Returns:
        Resolved Path object, or None if file_path is None/empty
    """
    if not file_path:
        return None
    
    file_path = Path(file_path)
    
    # If absolute path, return as is
    if file_path.is_absolute():
        return file_path
    
    # If relative path, resolve relative to base_dir (default: PROJECT_ROOT)
    base_dir = PROJECT_ROOT if base_dir is None else Path(base_dir).absolute()
    return base_dir / file_path