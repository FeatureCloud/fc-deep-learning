# Loss function Plugins
For loss function plugins, one should follow the same steps that are required for developing custom PyTorch loss functions.
The loss function receives the all keywords under the `train_config.loss.param` in the config file as `**kwargs`
```yaml
  train_config:
    loss:
      name: 'focal_loss.py'
      param:
        alpha:
          - 0.2
          - 0.3
          - 0.3
          - 0.2
        gamma: 1.1
```
## Current Dataloader plugins
* [focal_loss](focal_loss.py)
