"""
    FeatureCloud DeepLearning Application

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
    wait_for_init_params()
    broadcasting_init_params()
    local_computation()
    wait_for_aggregation()
    global_aggregation()
    write_results()
    final_step()

    """

    def __init__(self):
        super(CustomLogic, self).__init__()

        # Shared parameters and data
        self.parameters = {}
        self.workflows_states = {}

        # Define States
        self.states = {"Initializing": self.init_state,
                       "Read Input": None,
                       "Wait for Initial Parameters": self.wait_for_init_params,
                       "Broadcasting Initial parameters": None,
                       "Local Update": None,
                       "Wait for Global Aggregation": self.wait_for_aggregation,
                       "Global Aggregation": None,
                       "Writing Results": None,
                       "Finishing": self.final_step
                       }
        self.current_state = 'Initializing'

    def init_state(self):
        if self.id is not None:  # Test if setup has happened already
            self.current_state = "Read Input"

    def read_input(self):
        if self.coordinator:
            self.current_state = "Broadcasting Initial parameters"
        else:
            self.current_state = "Wait for Initial Parameters"

    def wait_for_init_params(self):
        self.progress = 'wait for init parameters from server'
        decoded_data = self.wait_for_server()
        if decoded_data is not None:
            print(f"{bcolors.SEND_RECEIVE} Received Init Params from coordinator. {bcolors.ENDC}")
            self.parameters = decoded_data
            self.current_state = "Local Update"

    def broadcasting_init_params(self):
        self.broadcast(self.parameters)
        self.current_state = "Local Update"

    def local_computation(self):
        """  should be overridden.
            called for clients to update their models
            based on recieved aggregated parameters.

        """
        self.send_to_server(self.parameters)
        if self.coordinator:
            self.current_state = "Global Aggregation"
        else:
            self.current_state = "Wait for Global Aggregation"

    def wait_for_aggregation(self):
        self.progress = 'wait for aggregation'
        decoded_data = self.wait_for_server()
        if decoded_data is not None:
            self.parameters, self.workflows_states = decoded_data[0], decoded_data[1]
            if all(self.workflows_states.values()):
                print(f"{bcolors.WARNING} Workflow of local updates is finished. {bcolors.ENDC}")
                self.current_state = "Writing Results"
            else:
                self.current_state = "Local Update"

    def global_aggregation(self):
        """  should be overridden!
            only called for the coordinator.
            aggregates the communicated parameters from clients.

        """
        self.broadcast([self.parameters, self.workflows_states])
        if all(self.workflows_states.values()):
            print(f"{bcolors.WARNING} Workflow is finished. {bcolors.ENDC}")
            self.current_state = "Writing Results"
        else:
            print(f'{bcolors.WARNING} the workflow is not finished for all splits. {bcolors.ENDC}')
            self.current_state = "Local Update"


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