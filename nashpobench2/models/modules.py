"""Modules for training models like optimizers or schedulers"""
from typing import Optional, List
from types import ModuleType

from hydra.experimental import initialize, compose
import inspect
from omegaconf import DictConfig
import torch
from torch import optim
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim import lr_scheduler, Optimizer

from nashpobench2.utils.config import *


def get_modules(
    cfg: DictConfig,
    modulename: str,
    codes: dict,
    namespace: list,
    **kwargs
):
    """Return an operation object corresponding to the class name.

    Examples:
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        >>> model = torch.nn.Linear(10, 1)
        >>> opt = get_modules(cfg, 'optimizers', {'optimizers':0, 'lr':3, 'momentum':2}, [optim], params=model.parameters())
        >>> sche = get_modules(cfg, 'schedulers', {'schedulers': 0}, [lr_scheduler], optimizer=opt)
        >>> opt.zero_grad()
        >>> inputs = torch.ones(1, 10)
        >>> outputs = model(inputs)
        >>> outputs.backward()
        >>> opt.step()
        >>> sche.step()
        >>> criterion = get_modules(cfg, 'criteria', {'criteria': 0}, [nn])
    """
    # get the class name and arguments
    if 'options' in list(cfg[modulename].keys()):
        argcfg = cfg[modulename]['options']
        for argname in list(argcfg.keys()):
            kwargs[argname] = subcfg2instance(argcfg, argname, codes[argname], namespace)
    return subcfg2instance(cfg[modulename], modulename, codes[modulename], namespace, **kwargs)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
