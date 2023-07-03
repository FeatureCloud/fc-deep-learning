"""
    FeatureCloud Four States App Template
    Copyright 2023 Mohammad Bakhtiari. All Rights Reserved.
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
import os
import shutil
import bios
import ast
import abc
from time import sleep
from FeatureCloud.app.engine.app import App, app
from states import generate_centralized_states, generate_federated_states
from bottle import Bottle
from FeatureCloud.app.api.http_ctrl import api_server
from FeatureCloud.app.api.http_web import web_server
import yaml


def print_configurations(data):
    print("\t\t\tConfigurations...")
    print("\n****************************************\n")
    formatted_yaml = yaml.dump(data, indent=2)
    print(formatted_yaml)
    print("\n****************************************\n")


def is_native():
    path_prefix = os.getenv("PATH_PREFIX")
    if path_prefix:
        return False
    return True


CONFIG_FILE = "config.yml"
GLOBAL_CONFIG = "mnt/input/global_config.yml"


def read_global_config():
    if os.path.exists(GLOBAL_CONFIG):
        return bios.read(GLOBAL_CONFIG)


def get_clients_dirs():
    if is_native():
        return get_dirs()
    if os.path.exists(GLOBAL_CONFIG):
        return get_dirs()
    return "/mnt/input"


def get_dirs():
    config = bios.read(GLOBAL_CONFIG)
    if is_centralized():
        return config["centralized"]["data_dir"]
    # Simulation
    root_dir = config["simulation"]["dir"]
    clients_dir = config["simulation"]["clients_dir"]
    clients_dirs = clients_dir.split(",")
    clients_dirs = [f"{root_dir}/{d}" for d in clients_dirs]
    return clients_dirs


def get_clients_ids():
    config = bios.read(GLOBAL_CONFIG)
    clients_ids = [cid.strip() for cid in config["simulation"]["clients"].split(",")]
    return clients_ids


def is_simulation():
    if os.path.exists(GLOBAL_CONFIG):
        config = bios.read(GLOBAL_CONFIG)
        return config.get("simulation", False)
    return False


def is_centralized():
    if os.path.exists(GLOBAL_CONFIG):
        config = bios.read(GLOBAL_CONFIG)
        return config.get("centralized", False)
    return False


def read_config(dir=None):
    if dir:
        file_path = f"{dir}/{CONFIG_FILE}"
    else:
        file_path = CONFIG_FILE
        if not is_native():
            file_path = f"/mnt/input/{CONFIG_FILE}"
    return bios.read(file_path)


def get_root_dir(input_dir=True, simulation_dir=None):
    simulation_dir = "/" + simulation_dir if simulation_dir else ''

    if input_dir:
        if is_native():
            return f".{simulation_dir}"
        return f"/mnt/input{simulation_dir}"
    if is_native():
        return f"./results{simulation_dir}"
    return f"/mnt/output{simulation_dir}"


def file_has_class(file_path, class_name):
    with open(file_path, 'r') as file:
        source_code = file.read()

    try:
        parsed = ast.parse(source_code)
    except SyntaxError:
        return False

    for node in ast.walk(parsed):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return True

    return False


class Controller:
    def __init__(self, clients_id):
        self.clients = {id.strip(): [] for id in clients_id}

    def register(self, client, app, coordinator):
        app.register()
        app.handle_setup(client_id=client,
                         coordinator=coordinator,
                         clients=list(self.clients.keys()))
        self.clients[client] = app

    def run(self):
        while True:
            for client in self.clients:
                if self.data_available(client):
                    data, dest_client = self.check_outbound(client)
                    if dest_client:
                        print("send_to_participant")
                        self.set_inbound(data, source_client=client, dest_client=dest_client)
                    else:
                        if self.clients[client].coordinator:
                            # broadcast
                            print("broadcast", list(self.clients.keys())[1:])
                            for dest_client in list(self.clients.keys())[1:]:
                                self.set_inbound(data, source_client=client, dest_client=dest_client)
                        else:
                            print("send_to_coordinator")
                            self.set_inbound(data, source_client=client, dest_client=list(self.clients.keys())[0])

            sleep(1)
            if self.finished():
                break

    def check_outbound(self, client):
        dest = self.status(client)['destination']

        data = self.clients[client].handle_outgoing()
        return data, dest

    def set_inbound(self, data, source_client, dest_client):
        self.clients[dest_client].handle_incoming(data, source_client)

    def finished(self):
        finished = [self.status(app)['finished'] for app in self.clients]
        return all(finished)

    def status(self, client):
        app = self.clients[client]
        client_status = {
            'available': app.status_available,
            'finished': app.status_finished,
            'message': app.status_message if app.status_message else (
                app.current_state.name if app.current_state else None),
            'progress': app.status_progress,
            'state': app.status_state,
            'destination': app.status_destination,
            'smpc': app.status_smpc,
        }
        return client_status

    def data_available(self, client):
        return self.status(client)['available']


def simulate():
    clients_dirs = get_clients_dirs()
    clients_ids = get_clients_ids()
    controller = Controller(clients_ids)
    for i, (client_id, client_dir) in enumerate(zip(clients_ids, clients_dirs)):
        app = App()
        kwargs = {"input_dir": client_dir,
                  "output_dir": client_dir.replace("/input/", "/output/")}
        app = generate_federated_states(app, **kwargs)
        controller.register(client_id, app, coordinator=i == 0)
    controller.run()


def run_app():
    server = Bottle()
    server.mount('/api', api_server)
    server.mount('/web', web_server)
    server.run(host='localhost', port=5000)


def centralized():
    input_dir = get_clients_dirs()
    output_dir = input_dir.replace("/input/", "/output/")
    # app = App()
    global app
    kwargs = {"input_dir": input_dir,
              "output_dir": output_dir}
    app = generate_centralized_states(app, **kwargs)
    app.register()
    if is_native():
        app.handle_setup(client_id='1', coordinator=True, clients=['1'])
    else:
        run_app()


def federated():
    global app
    # kwargs = {"input_dir": get_root_dir(),
    #           "output_dir": get_root_dir(input_dir=False)}
    kwargs = {"input_dir": "mnt/input",
              "output_dir": "mnt/output"}
    app = generate_federated_states(app, **kwargs)
    app.register()
    run_app()
