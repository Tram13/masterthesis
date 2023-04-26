import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from tools.config_parser import ConfigParser


class ProfilesManager:

    __slots__ = ['up_dir', 'up_names', 'bp_dir', 'bp_names']

    def __init__(self):
        main_dir = ConfigParser().get_value('cache', 'nlp_cache_dir')
        up_sub_dir = ConfigParser().get_value('cache', 'user_profiles_dir')
        bp_sub_dir = ConfigParser().get_value('cache', 'business_profiles_dir')
        self.up_dir = Path(main_dir, up_sub_dir)
        self.bp_dir = Path(main_dir, bp_sub_dir)
        self.up_names = [file.name for file in os.scandir(self.up_dir)]
        self.bp_names = [file.name for file in os.scandir(self.bp_dir)]

    def get_user_profiles_names(self) -> list[str]:
        return self.up_names

    def get_user_profiles(self, name: Union[os.PathLike, str] = None) -> pd.DataFrame:
        if name is None:  # load the best model
            name = ConfigParser().get_value("cache", "best_user")
        location = Path(self.up_dir, name)
        user_profiles = pd.read_parquet(location)
        if user_profiles.index.name != 'user_id':
            user_profiles = user_profiles.set_index(user_profiles['user_id'], drop=True)
            user_profiles = user_profiles.drop(columns=['user_id'])
        user_profiles = user_profiles.astype(np.float16)
        return user_profiles

    def get_business_profiles_names(self) -> list[str]:
        return self.bp_names

    def get_business_profiles(self, name: Union[os.PathLike, str] = None) -> pd.DataFrame:
        if name is None:  # load the best model
            name = ConfigParser().get_value("cache", "best_business")
        location = Path(self.bp_dir, name)
        business_profiles = pd.read_parquet(location)
        if business_profiles.index.name != 'business_id':
            business_profiles = business_profiles.set_index(business_profiles['business_id'], drop=True)
            business_profiles = business_profiles.drop(columns=['business_id'])
        business_profiles = business_profiles.astype(np.float16)
        return business_profiles
