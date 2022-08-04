# ConfigState

Many of FeatureCloud apps expect to have input files, `config.yml` file, and output files. All input files
should be copied into the same directory inside the docker container to be accessed by the app. To facilitate the I/O process 
`ConfigState` defines methods to automatically find the paths to all input and output files considering the logic(data splits).
Accordingly, it stores values in app internal dictionary for  `smpc_used`, `splits`', `input_files`, and `output_files`.
It is meant to use `ConfigState` for defining just one state, the `initial` one, first one, to set up all the required 
values for all following states in the app. `ConfigState` should be used in an app that all other states are defined 
based on `GeneralState` because it provides paths and splits required in other states to process the data.

#### def lazy_init()
After defining the state, for adding new key-value pairs into the app internal, this method should be called inside the run method
of the extended state. It ensures that these key-values inside the app internal exist and have the correct value:
- `smpc_used`: It's a flag to remember whether SMPC was used previously or not. It will be handled internally by the `GneralState` methods.
- `splits`: names of different splits for each data part that can be used to keep the order of processing them.
- `input_files`: paths to of different input data with considering several splits.  
- `output_files`: paths to of different output data with considering several splits.

#### read_config()
Read config.yml file it looks for `mode` and `dir` in `logic` part of the file, 
if it does not exist, default values will be used

#### finalize_config()
Generates split names, paths to input and output files. Regarding the `mode` of the app, there should be some splits for data
and for each data, different splits should be processed