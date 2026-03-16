import yaml
import os


class ConfigLoader:

    def __init__(self, config_path="config/mission_config.yaml"):

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"[CONFIG] File not found: {config_path}"
            )

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f) or {}

        print(f"[CONFIG] Loaded configuration from {config_path}")


    # --------------------------------------
    # Get top level key
    # --------------------------------------

    def get(self, key, default=None):

        return self.config.get(key, default)


    # --------------------------------------
    # Get nested config values
    # Example: get_nested("camera.type")
    # --------------------------------------

    def get_nested(self, key_path, default=None):

        keys = key_path.split(".")

        value = self.config

        for k in keys:

            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


    # --------------------------------------
    # Return entire config dictionary
    # --------------------------------------

    def get_all(self):

        return self.config


    # --------------------------------------
    # Debug print config
    # --------------------------------------

    def print_config(self):

        print("\n[CONFIG] Active Configuration")

        for key, value in self.config.items():
            print(f"{key}: {value}")