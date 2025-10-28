import yaml

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config_value(config, *keys, default=None):
    """Helper to safely get nested config values"""
    for key in keys:
        config = config.get(key, {})
    return config if isinstance(config, dict) else (config if config is not None else default)