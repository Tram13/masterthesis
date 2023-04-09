import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from tools.config_parser import ConfigParser


class UserProfilesManager:

    __slots__ = ['dir', 'names']

    def __init__(self):
        main_dir = ConfigParser().get_value('cache', 'nlp_cache_dir')
        sub_dir = ConfigParser().get_value('cache', 'user_profiles_dir')
        self.dir = Path(main_dir, sub_dir)
        self.names = [file.name for file in os.scandir(self.dir)]

    def get_user_profiles_names(self) -> list[str]:
        return self.names

    def get_user_profiles(self, name: Union[os.PathLike, str]) -> pd.DataFrame:
        user_profiles = pd.read_parquet(Path(self.dir, name))
        if user_profiles.index.name != 'user_id':
            user_profiles = user_profiles.set_index(user_profiles['user_id'], drop=True)
            user_profiles = user_profiles.drop(columns=['user_id'])
        user_profiles = user_profiles.astype(np.float16)
        return user_profiles
