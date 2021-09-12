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
from .logic import AppLogic, bcolors
import bios
import os


class CustomLogic(AppLogic):
    """ Subclassing AppLogic for overriding specific methods
        to implement the deep learning application.

    Attributes
    ----------
    parameters: dict
    workflows_states: dict

    Methods
    -------
    init_state()
    read_input()
    final_step()
    write_results():
    """

    def __init__(self):
        super(CustomLogic, self).__init__()

        # Shared parameters and data
        self.parameters = {}
        self.workflows_states = {}

        # Define States
        self.states = {"Initializing": self.init_state,
                       "Writing Results": None,
                       "Finishing": self.final_step
                       }
        self.current_state = 'Initializing'

    def init_state(self):
        raise NotImplementedError
        if self.id is not None:  # Test if setup has happened already
            if self.coordinator:
                self.current_state = "Something"
            else:
                self.current_state = "S.t. else"

    def read_input(self):
        raise NotImplementedError
        if self.coordinator:
            self.current_state = "Something"
        else:
            self.current_state = "S.t. else"

    def write_results(self):
        if self.coordinator:
            self.data_incoming.append('DONE')
            self.current_state = "Finishing"
        else:
            self.data_outgoing = 'DONE'
            self.status_available = True
            self.current_state = None

    def final_step(self):
        self.progress = 'finishing...'
        if len(self.data_incoming) == len(self.clients):
            self.status_finished = True
            self.current_state = None