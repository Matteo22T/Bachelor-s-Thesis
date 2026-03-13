import yaml
from types import SimpleNamespace

class ConfigLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.data = SimpleNamespace(**self.config['data'])
        self.experiment = SimpleNamespace(**self.config['experiment'])
        self.grid_search = SimpleNamespace(**self.config.get('grid_search'))
        