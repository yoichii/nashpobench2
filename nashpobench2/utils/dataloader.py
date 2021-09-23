"""Load datasets"""
import time
from types import ModuleType
from typing import Optional, List, Tuple

from hydra.experimental import initialize, compose
import inspect
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, Sampler, DataLoader, SubsetRandomSampler
import torchvision
from torchvision.transforms import Compose
import yaml

from nashpobench2.utils.config import subcfg2instance


def get_dataloader(
    cfg: DictConfig,
    idxcfg: dict,
    codes: dict
) -> Tuple[DataLoader]:
    """Return an operation object corresponding to the class name.

    Examples:
        >>> start = time.time()
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        ...     with open('../../config/idx/split.yaml') as f:
        ...         idxcfg = yaml.safe_load(f)
        ...     train_loader, valid_loader, test_loader = get_dataloader(cfg, idxcfg, {'dataloaders': 0, 'datasets': 0, 'batch_size': 2})
        ...     for i, (inputs, labels) in enumerate(train_loader):
        ...         if i > 10:
        ...             break
        ...     for i, (inputs, labels) in enumerate(test_loader):
        ...         if i > 10:
        ...             break
        Files already downloaded and verified
        Files already downloaded and verified
        Files already downloaded and verified
    """
    dataloaders = []
    subcfg = cfg['datasets']
    for datatype in list(subcfg.keys()):
        kwargs = {}
        if 'options' in list(subcfg[datatype].keys()):
            argcfg = subcfg[datatype]['options']
            for argname in list(argcfg.keys()):
                kwargs[argname] = subcfg2instance(argcfg, argname, codes[argname])
        # get transform
        transformlist = []
        for i in range(len(subcfg[datatype]['transforms'])):
            transformlist.append(subcfg2instance(subcfg[datatype], 'transforms', i, [torchvision.transforms], **kwargs))
        transform = Compose(transformlist)
        # get dataset
        kwargs.update({'transform': transform})
        dataset = subcfg2instance(subcfg[datatype], 'datasets', codes['datasets'], [torchvision.datasets], **kwargs)
        if datatype == 'train' or datatype == 'valid':
            sampler = SubsetRandomSampler(idxcfg[datatype])
        else:
            sampler = None
        kwargs.update({'dataset': dataset, 'sampler': sampler})
        dataloaders.append(subcfg2instance(subcfg[datatype], 'dataloaders', codes['dataloaders'], [torch.utils.data], **kwargs))
    return tuple(dataloaders)



if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
