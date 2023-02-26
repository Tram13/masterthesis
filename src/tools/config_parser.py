from configparser import ConfigParser as BuiltInConfPars
import os
from pathlib import Path


# Extension of built-in Config Parser of Python
class ConfigParser:
    # Location of the .ini file that contains all configuration variables

    def __init__(self):
        self.PATH: os.PathLike = Path("../config.ini")  # Default location
        self.CP_INSTANCE = BuiltInConfPars()
        self.CP_INSTANCE.read(self.PATH)
        self.type_to_function_map = {
            bool: self.CP_INSTANCE.getboolean,
            int: self.CP_INSTANCE.getint,
            float: self.CP_INSTANCE.getfloat,
            str: self.CP_INSTANCE.get
        }

    def get_value(self, section: str, key: str, return_type: type[bool | int | float | str] = str):
        self._assert_config_path()
        # Get the read-function that matches the desired return type
        type_function = self.type_to_function_map.get(return_type, self.CP_INSTANCE.get)
        return type_function(section, key)

    def _assert_config_path(self):
        self.PATH = Path("../config.ini")  # Needs to be updated for Jupyter notebooks
        if not (os.path.exists(self.PATH) and os.path.isfile(self.PATH)):
            raise FileNotFoundError(f"config.ini not found in {self.PATH}")
