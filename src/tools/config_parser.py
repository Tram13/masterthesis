from configparser import ConfigParser as BuiltInConfPars
import os
from pathlib import Path


# Extension of built-in Config Parser of Python
class ConfigParser:
    # Location of the .ini file that contains all configuration variables
    PATH: os.PathLike = Path("../config.ini")  # Default location

    CP_INSTANCE = BuiltInConfPars()
    CP_INSTANCE.read(PATH)
    type_to_function_map = {
        bool: CP_INSTANCE.getboolean,
        int: CP_INSTANCE.getint,
        float: CP_INSTANCE.getfloat,
        str: CP_INSTANCE.get
    }

    @staticmethod
    def get_value(section: str, key: str, return_type: type[bool | int | float | str] = str):
        # Get the read-function that matches the desired return type
        type_function = ConfigParser.type_to_function_map.get(return_type, ConfigParser.CP_INSTANCE.get)
        return type_function(section, key)
