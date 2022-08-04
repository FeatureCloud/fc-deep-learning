"""
    FeatureCloud Cross Validation Application
    Copyright 2022 Mohammad Bakhtiari. All Rights Reserved.
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
import bios
import os
import shutil
from FeatureCloud.app.engine.app import AppState, LogLevel


class State(AppState):
    """
    An abstract class to handle paths to in/output files regarding different data splits.
    
    Attributes
    ----------
    app_name: str
        Name of the Utils (used for both docker image and the config file)
    config: dict
        content of config file for the Utils
        config.yml file can contain configs for more than one Utils.
    input_dir: str
        path to directory inside the Utils's docker container for input files
        Default: `/mnt/input`
    output_dir: str
        path to directory inside the Utils's docker container for output files
        Default: `/mnt/output`
    config_file: str
        the path to `config.yml` file inside the input directory
    mode: str
        Choices:
        `directory`: data files have splits.
        `file`: there is only a single file for each data.
    dir: str
        the directory name for input/output data.

    Methods
    -------
    lazy_init()
    read_config()
    finalize_config()
    """

    def __init__(self, app_name, input_dir: str = "/mnt/input", output_dir: str = "/mnt/output"):
        self.config = {}
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config_file = f"{self.input_dir}/config.yml"
        self.mode = 'file'
        self.dir = '.'
        self.app_name = app_name

    def lazy_init(self):
        """ Add new key-value pairs into the `Utils-internal` dictionary, shared memory between states so that they can be accessible for other states.
            
            Keys:
            
            `use_smpc`: User-preference on using SMPC.
            `splits`: names of splits(it should be the same for all data).
            `input_files`: paths to all input files regarding the data splits.
            `output_files`: paths to all output files regarding the data splits.
            
        """
        self.store('smpc_used', False)
        self.store('splits', set())
        self.store('input_files', {})
        self.store('output_files', {})

    def read_config(self):
        """ Read config.yml file
            it looks for `mode` and `dir` in `logic` part of the file,
            if it does not exist, default values will be used.
        """
        self.config = bios.read(self.config_file)[self.app_name]
        if 'debug' in self.config:
            if self.config['debug']:
                self.store('debug', True)
                self.log("Debug mode is ON", LogLevel.DEBUG)
            else:
                self.store('debug', False)

        if 'logic' in self.config:
            self.mode = self.config['logic']['mode']
            self.dir = self.config['logic']['dir']
        else:
            self.log(f"There are no 'logic' options in 'config.yml' file!\n"
                     f"default values will be used:\n"
                     f"mod: 'file'\n"
                     f"dir: '.'", LogLevel.DEBUG)

    def finalize_config(self):
        """  Generates split names, paths to input and output files.
             Regarding the `mode` of the Utils, there should be some splits for data,
             and for each data, different splits should be processed.
        """
        if self.mode == "directory":
            splits = [f.path for f in os.scandir(f'{self.input_dir}/{self.dir}') if f.is_dir()]
        else:
            splits = [self.input_dir, ]
        self.store('splits', set(sorted(splits)))
        self.log(f" Splits order:")
        for i, split in enumerate(self.load('splits')):
            self.log(f"Split {i}: {split}")
        self.store('input_files', {k: [f"{split}/{v}" for split in self.load('splits')]
                                   for k, v in self.config['local_dataset'].items()})
        self.store('output_files', {k: [f"{split.replace('/input', '/output')}/{v}"
                                        for split in self.load('splits')]
                                    for k, v in self.config['result'].items()})
        for split in self.load('splits'):
            os.makedirs(split.replace("/input", "/output"), exist_ok=True)
        shutil.copyfile(self.input_dir + '/config.yml', self.output_dir + '/config.yml')
