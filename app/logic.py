"""
    FeatureCloud Template
    mohammad.bakhtiari@uni-hamburg.de
"""
import threading
import time

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pd

jsonpickle_numpy.register_handlers()
jsonpickle_pd.register_handlers()


class AppLogic:
    """ Implementing the workflow for FeatureCloud platform

    Attributes
    ----------
    status_available: bool
    status_finished: bool
    id:
    coordinator: bool
    clients:
    data_incoming: list
    data_outgoing: list
    thread:
    iteration: int
    progress: str
    INPUT_DIR: str
    OUTPUT_DIR: str
    mode: str
    dir: str
    splits: dict
    test_splits: dict
    states: dict
    current_state: str

    Methods
    -------
    handle_setup(client_id, coordinator, clients)
    handle_incoming(data)
    handle_outgoing()
    app_flow()
    send_to_server(data_to_send)
    wait_for_server()
    broadcast(data)
    lazy_initialization(mode, dir)
    """

    def __init__(self):

        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'

        # === Custom ===
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        self.mode = None
        self.dir = None
        self.splits = {}
        # self.test_splits = {}

        self.states = {}
        self.current_state = None

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....")
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...")
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        print(f"{bcolors.STATE}States:{bcolors.ENDC}")
        for i, state in enumerate(self.states):
            print(f"{bcolors.STATE}{i}: {state}{bcolors.ENDC}")

        # Initial state
        self.progress = 'initializing...'
        previous_states = [self.current_state]
        while True:
            if self.current_state != previous_states[-1]:
                previous_states.append(self.current_state)
            msg = ""
            if len(previous_states) < 5:
                for state in previous_states:
                    msg += state + "$#@"
            else:
                msg = "... "
                for state in previous_states[-5:]:
                    msg += state + "$#@"
            print(f"{bcolors.STATE}{msg[:-3].strip().replace('$#@', ' ---> ')}{bcolors.ENDC}")
            print(f"{bcolors.STATE}Current State: {self.current_state}{bcolors.ENDC}")
            self.states[self.current_state]()
            if self.current_state is None:
                break

            time.sleep(1)

    def send_to_server(self, data_to_send):
        """  Will be called only for clients
            to send their parameters or local statistics for the coordinator

        Parameters
        ----------
        data_to_send: list

        """
        data_to_send = jsonpickle.encode(data_to_send)
        if self.coordinator:
            self.data_incoming.append(data_to_send)
        else:
            self.data_outgoing = data_to_send
            self.status_available = True
            print(f'{bcolors.SEND_RECEIVE} [CLIENT] Sending data to coordinator. {bcolors.ENDC}', flush=True)

    def get_clients_data(self):
        """ Will be called only for the coordinator
            to get all the clients parameters or statistics.
            For each split, corresponding clients' data will be yield back.

        Returns
        -------
        clients_data: list
        split: str
        """
        print(f"{bcolors.SEND_RECEIVE} Received data of all clients. {bcolors.ENDC}")
        data = [jsonpickle.decode(client_data) for client_data in self.data_incoming]
        self.data_incoming = []
        for split in self.splits.keys():
            print(f'{bcolors.SPLIT} Get {split} {bcolors.ENDC}')
            clients_data = []
            for client in data:
                clients_data.append(client[split])
            yield clients_data, split

    def wait_for_server(self):
        """ Will be called only for clients
            to wait for server to get
            some globally shared data.

        Returns
        -------
        None or list
            in case no data received None will be returned
            to signal the state!
        """
        if len(self.data_incoming) > 0:
            data_decoded = jsonpickle.decode(self.data_incoming[0])
            self.data_incoming = []
            return data_decoded
        return None

    def broadcast(self, data):
        """ will be called only for the coordinator after
            providing data that should be broadcast to clients

        Parameters
        ----------
        data: list

        """
        data_to_broadcast = jsonpickle.encode(data)
        self.data_outgoing = data_to_broadcast
        self.status_available = True
        print(f'{bcolors.SEND_RECEIVE} [COORDINATOR] Broadcasting data to clients. {bcolors.ENDC}', flush=True)

    def lazy_initialization(self, mode, dir):
        """

        Parameters
        ----------
        mode: str
        dir: str
        """
        self.mode = mode
        self.dir = dir


class TextColor:
    def __init__(self, color):
        if color:
            self.SEND_RECEIVE = '\033[95m'
            self.STATE = '\033[94m'
            self.SPLIT = '\033[96m'
            self.VALUE = '\033[92m'
            self.WARNING = '\033[93m'
            self.FAIL = '\033[91m'
            self.ENDC = '\033[0m'
            self.BOLD = '\033[1m'
            self.UNDERLINE = '\033[4m'
        else:
            self.SEND_RECEIVE = ''
            self.STATE = ''
            self.SPLIT = ''
            self.VALUE = ''
            self.WARNING = ''
            self.FAIL = ''
            self.ENDC = ''
            self.BOLD = ''
            self.UNDERLINE = ''


bcolors = TextColor(color=False)
