# DataLoader Plugins
DatLoader Plugins are designed to deal with specific data types or/and formats. In some cases DataLoader can support some level of data cleaning and preprocessing at local level (not federated).

## How to
The DataLoader plugin will receive all keywords under the `local_dataset.detail` in the config file as `**kwargs` from the app beside `path`.

```yaml
  local_dataset:
    train: "train.npz"
    test: "test.npz"
    detail:
      sep: ','
      label: "Group"
    init_model: "model.pt"
```
### `load` method
To load the data into PyTorch Dataloader based on the `path` to data (including the dataformat), and `batch_size`
```python
from torch.utils.data import DataLoader
def load(self, path, batch_size):
    dl = DataLoader
    return dl
```
The load method should return an object of `torch.utils.data.DataLoader` type.


* `sample_data` property: to provide a small dataset sample which is necessary for covering Custom models that are provided in config file.

## Current Dataloader plugins
* [ImageLoader](ImageLoader.py): Load image data from `.npz` file with `data` and `target` keys.
* [RNALoader](RnaLoader.py): Load sc-RNA seq data from `.csv` files.
