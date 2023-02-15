# Plugins
For different use-cases different architecture and training components are needed. For the deep learning aop, one can implement
and use different plugins. Here I explain how a plugin is developed and works in the deep learning app.

## Supported plugins
A wide range of plugins will be supported in the deep learning app. Currently the following are supported and exemplified in this repository:
* DataLoaders
* Loss functions
* Architectures
* Optimizers
* Federated Optimizers

## How to develop a plugin
Even though plugins are geared for specific functionalities, I tried to keep the format simple compatible, so by developing one plugin the experience helps for new plugins.
* Any plugin should be implemented as a class, with desired name and couple of methods to cover the target functionality.
* The class in most cases inherits the `torch.nn.module`
* `__init__` method receives the input arguments from the app, which is restricted and custom for a plugin type.
* Input arguments for the plugin can be extended or changes using the config file.
```python
import torch.nn as nn
class CustomPlugin(nn.module):
    def __init__(self, **kwargs):
        for k, v in kwargs:
            setattr(self, k, v)

```
However, there are minor issues that differ based on the plugin type, which I will cover in the corresponding plugin sections:
