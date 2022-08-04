# [FeatureCloud Deep Learning](https://featurecloud.ai/app/deep-learning)
### FeatureCloud Deep Learning Application

Practitioners and researchers can use the FeatureCloud Deep Learning app to run different deep neural networks
in a federated fashion inside the FeatureCloud platform. The Deep learning app is implemented using the PyTorch library. It supports other architectural and training elements to provide FeatureCloud users with enough flexibility to experiment with diverse architecture while shying away from making 
the app more complicated.

## Architecture
One of the significant advantages of Deep Neural networks is the universality of the models to be applied 
to highly diversified fields and tasks. Such a helpful capability lies in the flexibility of designing DNN models
by incorporating different layers and stacking them above each other. The same federated platform with the same
setting can be used to train DNNs with different architectures. The deep learning app provides three options for end-users to deploy deep learning app with desired architecture,
:
### Using the existing models
The simplest option to use different DNN model architectures is choosing between the provided options in
`models.pytorch` class. For this purpose, one should use `name` key to provide the model name and required arguments
with corresponding values.
```python
model:
    name: 'CNN'
    n_classes: 10
    in_features: 1
```
### loading a predefined model class
Instead of using existing models, one can define the desired model class, provided as generic data to clients, 
which later will be imported by the app. For this purpose, one should use `name` key to give the model class name and required arguments
with corresponding values. the model class should be named `Model` and contain the forward method, for example,
you can check [models/pytorch/models](/data/sample_data/generic/cnn.py) 

```python
model:
    name: 'cnn.py'
    n_classes: 10
    in_features: 1
```
### Defining a model in the config file 
Alternatively, end-users can list their desired stacked network, where different layers can be listed, to introduce a DNN model architecture to the app. Each layer has different parameters that should be mentioned in the layer's parameters list.
In listing the architecture, users should use the same module names as [`torch.nn`](https://pytorch.org/docs/stable/nn.html).
The same rule is applied for listing layer parameters.

For the sake of simplicity, there are simplifying rules that users can take advantage of 
when they list their model architecture:
1. First in, first stacked: layers will be added to the network based on the list order. 
2. Default values: When no value is provided for any layer, default values will be used.
    If parameters' values are required, providing no values will result in an error.
    Users should check the required parameters for each layer to prevent such errors. 
3. Same as previous: For each specific layer type, once a parameter value is provided, to preclude
   repetition, that value will be used for the next layer's usage. This rule remains as long as 
   no parameter value is given for the new layer. Once a new value is provided, the same will be true for it. 
```python
model:
    - type: 'Conv2d'
      param:
        in_channels: None
        out_channels: 32
        kernel_size: 3
        stride: 1
        padding: 1
        bias: True
    - type: 'MaxPool2d'
      param:
        kernel_size: 2
        stride: 2
    - type: 'ReLU'
    - type: 'Conv2d'
      param:
        in_channels: 32
        out_channels: 64
    - type: 'MaxPool2d'
    - type: 'ReLU'
    - type: 'Flatten'
    - type: 'Linear'
      param:
        in_features: None
        out_features: 128
        bias: True
    - type: 'ReLU'
    - type: 'Linear'
      param:
        in_features: 128
        out_features: 10
```
## Optimizers
In many cases, to change the data, one needs to use a specific optimizer. In that regard, deep learning app
enables end-users to specify their desired optimizer with suitable hyperparameters. PyTorch's optimizers can
be used in the training phase of the Deep Learning app, where 
all the optimizers are imported from [`torch.nn.functional`](https://pytorch.org/docs/stable/nn.functional.html).
FeatureCloud users can mention the desired optimizer and its parameters in the config file.
Same as for layers, in listing the optimizer parameters, in case of omitting the parameters,
default values will be used.
```python
train_config:
    optimizer:
      name: 'SGD'
      param:
        lr: 0.1
```

## Loss Functions
Like optimizers, end-users can specify a loss function and its arguments in the deep learning app in the config file.
PyTorch's loss functions can be imported from [`torch.nn`](https://pytorch.org/docs/stable/nn).
Same as for layers and optimizer parameters, default values will be used for parameters unless it is mentioned in the config file.
```python
train_config:
    loss:
      name: 'CrossEntropyLoss'
```


## Config Settings

```python
fc_deep:
  local_dataset:
    train: "train.npy"
    test: "test.npy"


  logic:
    mode: "file"
    dir: "."


  results:
    pred: "y_pred.csv"
    target: "y_test.csv"


  federated_hyper_parameters:
    max_iter: 10
    n_classes: 10
    federated_model: 'FedMMB'
    batch_count: 1



  train_config:
    torch_loader: False # True: using torchvision Dataloader, False using custom Dataloader
    batch_size: 32
    test_batch_size: 32
    epochs: 1
    optimizer:
      name: 'SGD'
      param:
        lr: 0.1
    loss:
      name: 'CrossEntropyLoss'


  model:
    name: 'CNN'
    n_classes: 10
    in_features: 1

```



### Config file options
#### Local dataset
Currently, only NumPy files are supported as input data. Input data should contain 'train' file, while the 'test' file
is optional. When no test data is provided, the corresponding FeatureCloud user will not
get any evaluation results without the test data.

#### logic
`logic` key handles the case of using cross-validation. For the 'mode' and 'dir' settings, you can refer to the
[Companion apps](https://github.com/FeatureCloud/fc-companion-apps/tree/master/CustomStates).


#### Federated Hyper-Parameters
- `max_iter`: Number of communication rounds.
- `n_classes`: number of classes in the target dataset.
- `federated_model`:
  - `FedMMB`: [Federated Multi-Mini-Batch](https://arxiv.org/abs/2011.07006)
    - `batch_count`: number of batches in a local update.  
  - `FedAvg`: [Federated Averaging](https://arxiv.org/abs/1602.05629)

#### Training config
In the local training of the base DNN model, FeatureCloud users should decide which optimizer and loss function they want to use
and give some values for the training parameters. There are some specific training parameters to determine as follows:
- `torch_loader`: Is a boolean value, where `True` means default Pytorch `DataLoader` should be used for loading 
 the data set and feeding it to the model. In the case of `False` value, the custom class of NumpyLoader will be used.
 Using `torchvision.DataLoader` demands more memory than the custom DataLoader implemented in the app.
- `batch_size`: Number of samples in each training batch.
- `test_batch_size`: Number of samples in each test batch.
- `epochs`: The number of training epochs, which in the case of using `FedMMB` for federated aggregation, will be ignored.

##### Optimizer
For optimizing the network, users should decide which optimizers they will use and which parameters.
Regarding the optional parameters' value, if not provided, the default value will be used. 
- name: 'SGD'
- param:
  - lr: 0.1
  
##### Loss function
Also, for the loss function, users can determine it the same as the optimizer. 
- name: 'CrossEntropyLoss'

### Run deep learning app

#### Prerequisite

To run the deep learning app, you should install Docker and FeatureCloud pip package:

```shell
pip install featurecloud
```

Then either download the deep learning app image from the FeatureCloud docker repository:

```shell
featurecloud app download featurecloud.ai/fc_deep_networks
```

Or build the app locally:

```shell
featurecloud app build featurecloud.ai/fc_deep_networks
```

Please provide example data so others can run the deep learning app with the desired settings in the `config.yml` file.

#### Run YOUR_APPLICATION in the test-bed

You can run the deep learning app as a standalone app in the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also run the app using CLI:

```shell
featurecloud test start --app-image featurecloud.ai/fc_deep_networks --client-dirs './sample_data/c1,./sample_data/c2' --generic-dir './sample_data/generic'
```
