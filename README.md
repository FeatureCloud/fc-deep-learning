# FeatureCloud Deep Learning
### FeatureCloud Deep Learning Application

Practitioners and researchers can use FeatureCloud Deep Learning app to run different deep neural networks
in a federated fashion inside FeatureCloud platform. The Deep learning app is implemented using
Pytorch library and supports different architectural and training elements to provide FeatureCloud
users with enough flexibility to experiment with diverse architecture while shying away from making 
the app more complicated.

## Architecture
One of the great advantages of Deep Neural networks is Universality of the models to be applied 
on highly diversified fields and tasks. Such a useful capability lies in the flexibility of designing DNN models
with incorporating different layers, and stacking them above each other. For employing Deep Learning app, users can 
simply list their desired stacked network, where different layers can be listed, to introduce an
architecture to the app. Each layer has different parameters, that should be mentioned in layer's parameters list.
In listing the architecture, users should use same module names as [`torch.nn`](https://pytorch.org/docs/stable/nn.html).
Same rule is applied for listing layer parameters.

For the sake of simplicity there are simplifying rules that users can take advantages of 
when they list their model architecture:
1. First in, first stacked: layers will be added to the network based of the order of being listed. 
2. Default values: When no value is provided for any layer, default values will be used.
    In case parameters' values are required, providing no values will result in an error.
    To prevent such errors, users should check the required parameters for each layer. 
3. Same as previous: For each specific layer type, once a parameter value is provided, to preclude
   repetition, that value will be used for the next usage of the layer. This rule remains as long as 
   no parameter value is given for the new layer. Once a new value is provided, same will be true for it. 

## Optimizers
Pytorch optimizers can be used in the training phase of Deep Learning app, where 
all the optimizers are imported from [`torch.nn.functional`](https://pytorch.org/docs/stable/nn.functional.html).
FeatureCloud users can mention the desired optimizer, and its parameters in the config file.
Same as for layers, in listing the optimizer parameters, in case of omitting the parameters,
default values will be used.

## Loss Functions



## Config Settings
```
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


### Config file options
#### Local dataset
Currently, only NumPy files are supported as input data. input data should contain 'train' file, while the 'test' file
is optional. When there is no test data is provided, corresponding FeatureCloud user without the test data, will not
get any evaluation results.

#### logic
For the 'mode' and 'dir' settings you can refer to [FeatureCloud Template](https://github.com/FeatureCloud/fc-template).


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
 the data set and feeding it to the model. In the case of `False` value, custom class of NumpyLoader will be used.
 Using `torchvision.DataLoader` demands more memory than custom DataLoader implemented in the app.
- `batch_size`: Number of samples in each training batch.
- `test_batch_size`: Number of samples in each test batch.
- `epochs`: Number of training epochs, which in the case of using `FedMMB` for federated aggregation will be ignored.

##### Optimizer
For optimizing the network, users should decide which optimizers they are going to use , and with which parameters.
Regarding the optional parameters' value, if not provided the default value will be used. 
- name: 'SGD'
- param:
  - lr: 0.1
  
##### Loss function
Also, for the loss function users can determine it same as optimizer. 
- name: 'CrossEntropyLoss'