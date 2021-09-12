# FeatureCloud Template
### Implementing FeatureCloud Applications

FeatureCloud provide an advantageous platform which is accessible at [FeatureCloud.ai](https://featurecloud.ai/) 

In an OO fashion, just by extending two classes, developers can use FeatureCloud Template for
implementing one-shot or iterative applications. This template consists of three main classes to interact with FC Controller and execute the app-level tasks. Generally, two types of clients are used in FeatureCloud Template:

- Client: Every participant in the FeatureCloud platform is considered a client who should perform local tasks and communicate some intermediary results with the coordinator. No raw data are supposed to be exchanged among clients and the coordinator.
- Coordinator: One of the clients who can receive results of other clients, aggregate, and broadcast them.


## AppLogic
Using the `AppLogic` class, users can define different states and make a flow to move from one to another. Each state should be added to the `states` attributes, while there is no predefined order for executing states, the flow direction will be handled using `CustomLogic` class. With `current_state`, developers know the flow and determine which state they desire to move in.

### Attributes
We categorize attributes in the `AppLogic` class as follows:
- Controlling the flow:
  - `states`: Python dictionary that keeps names of states, as keys, and methods, as values.
  - `current_state` Name of the current state, or the next state that a developer wants.
  - `status_available`: Boolean attribute to signal the availability of data to the FeatureCloud Controller to share it. 
  - `status_finished`: Boolean variable to signal the end of app's execution to the FeatureCloud Controller.
  - `thread`:
  - `iteration`: Number of executed iterations.
  - `progress`: Short descriptor of internal progress of app instance for the FeatureCloud Controller.
- General 
  - `id`: ID of each participant, regardless of being client or coordinator.
  - `coordinator`: Boolean flag indicating whether the running container is a coordinator or not.
  - `clients`: Contains IDs of all participating clients.
- Data management:
  - For communicating data:
    - `data_incoming`: list of data that was received.
    - `data_outgoing`: list of data that should be shared.
  - For I/O from the docker container:
    - `INPUT_DIR`: path to the directory inside the docker container for reading the input files.
    - `OUTPUT_DIR`: path to the directory inside the docker container for writing the results.
    - `mode`: Primarily used for indicating whether input files are stored in one folder or multiple folders.
    - `dir`: The folder containing the input files. 
    - `splits`: A dictionary of possible splits(folder names containing the input data that are used for training) 
  
### Methods
Using `lazy_initializing`, Developers can initialize some attributes in an arbitrary time. `app_flow` is the method in `AppLogic` class that contains a state machine for the client and the coordinator. 
It calls corresponding methods to each state. These are the four methods in `AppLogic` class that facilitate communicating data between coordinator and clients.

- `send_to_server`: should be called only for clients to send their data to the coordinator.
- `get_clients_data`: Should be called only for the coordinator to wait for the clients until receiving their data.
  For each split, corresponding clients' data will be yield back.
- `wait_for_server`: Should be called only for clients to wait for coordinator until receiving broadcasted data.
- `broadcast`: should be called only for the coordinator to broadcast the same date to all clients.


## CustomLogic
`CustomLogic` is an extension class of `AppLogic`, which defines all the states, determines the first state, and, more importantly, 
implements the flow between states. Besides controlling the flow, generally, we categorize states' tasks as
operational and/or communicational. For communicational states responsible for sharing or receiving data,
the method will be fully implemented and assigned to the state in `CustomLogic` class. For others, only the flow 
related part will be implemented here, and the operation happens in `CustomApp` class. All the data-related
attributes, shared among clients, should be introduced in `CustomLogic`.  

### Attributes
- `parameters`: A dictionary that can contain any data that should be shared.
- `workflows_states`: A dictionary that can signal any messages to the coordinator or vice versa.

### Methods
Methods are highly diverse regarding the target application; however, almost every application
should include initializing and finalizing state and method. 
- `init_state`
- `read_input`
- `final_step`

## CustomApp
`CustomApp` is an extension of `CustomLogic` that introduces all the required attributes and methods to execute the app's task. Each state's method call its corresponding superclass method in `CustomLogic` to change the flow to the next state, which was previously implemented.
