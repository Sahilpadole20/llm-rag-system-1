import os
import yaml

class Config:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        with open(self.config_file, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

def load_config(config_file: str):
    """Load configuration from a YAML file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)