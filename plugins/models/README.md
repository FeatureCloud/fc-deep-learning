# DNN model architecture Plugins
For model plugins, one should follow the same steps that are required for developing custom PyTorch models.
The custom model receives the all keywords under the `model` in the config file as `**kwargs`
```yaml
   model:
    name: 'cnn.py'
    n_classes: 10
    in_features: 1
```
## Current Dataloader plugins
* [CNN](cnn.py)
* [MLP](mlp.py)
