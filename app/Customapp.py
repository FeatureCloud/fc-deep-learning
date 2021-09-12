"""
    FeatureCloud Template

    Copyright 2021 Mohammad Bakhtiari. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
from .logic import bcolors
from .Customlogic import CustomLogic
import bios
import os
import shutil




class CustomApp(CustomLogic):
    """
    Attributes
    ----------


    Methods
    -------
    read_config(config_file)
    read_input()
    write_results()
    """

    def __init__(self):
        super(CustomApp, self).__init__()

        #   Configs

        #   Models & Parameters


        #  Update States Functionality
        self.states["Broadcasting Config file and data"] = self.broadcast_data
        self.states["Writing Results"] = self.write_results

    def read_config(self, config_file):
        """ Read Config file

        Parameters
        ----------
        config_file: string
            path to the config.yaml file!

        """
        raise NotImplementedError

    def read_input(self):
        self.progress = "Read data"

        super(CustomApp, self).read_input()

    def write_results(self):
        self.progress = "write results"

        super(CustomApp, self).write_results()

logic = CustomApp()
