"""Configure Utilities"""
import inspect
from itertools import product
from typing import List
from types import ModuleType

from hydra.experimental import compose, initialize
from omegaconf import DictConfig
from omegaconf.listconfig import DictConfig, ListConfig
import torch
from torch import optim as optim
from torch import Tensor
import torch.nn as nn

from nashpobench2.models.operations import ReLUConvBN, Pooling, Identity, Zero, Serialize
from nashpobench2.models import operations


def subcfg2instance(
    cfg: DictConfig,
    keyname: str,
    valueidx: int,
    namespace: list = [],
    **kwargs: dict
):
    """Return a list of objects corresponding to the keyname from namespace.

    Examples:
        >>> with initialize(config_path='.'):
        ...     cfg = compose(config_name='sample')
        ...     model = nn.Linear(100, 10)
        ...     m = subcfg2instance(cfg, 'sample1', 0)
        ...     m = subcfg2instance(cfg, 'sample2', 0, [nn, optim])
        ...     m = subcfg2instance(cfg, 'sample3', 0)
        ...     m = subcfg2instance(cfg, 'sample4', 0, [nn, optim], **{'params': model.parameters()})
        ...     m = subcfg2instance(cfg, 'sample5', 0, [nn, optim], **{'params': model.parameters()})
        ...     m = subcfg2instance(cfg, 'sample5', 1, [nn, optim], **{'params': model.parameters()})
        ...     m = subcfg2instance(cfg, 'sample5', 2, [nn, optim], **{'params': model.parameters()})
        ...     m = subcfg2instance(cfg, 'sample5', 3, [nn, optim], **{'params': model.parameters()})
        ...     m = subcfg2instance(cfg, 'sample5', 4, [nn, optim], **{'params': model.parameters()})
        ...     m = subcfg2instance(cfg, 'sample5', 5, [nn, optim], **{'params': model.parameters()})
    """
    # kwargs to variables
    for k,v in kwargs.items():
        globs = globals()
        locs = locals()
        exec(f'{k} = v', globs, locs)
    # multiple or one option
    if type(cfg[keyname]) == ListConfig:
        assert type(valueidx) == int, f'{keyname} has multiple options, but no index is givin.'
        c = cfg[keyname][valueidx]
    else:
        raise ValueError(f'expected ListConfig, but got {type(cfg[keyname])} in cfg {keyname} value')
    # if int, return directly
    if type(c) == int or type(c) == float:
        return c
    elif type(c) == ListConfig:
        return list(c)
    # if str enclosed in ", return as str
    elif type(c) == str and c.startswith("'") and c.endswith("'"):
        return c[1:-1]
    # if str, regarded as classname
    elif type(c) == str:
        classname = c
        arguments = {}
    elif type(c) == DictConfig:
        classname = list(c.keys())[0]
        assert type(classname) == str, f'expected str, but got {type(classname)} in {keyname} classname'
        arguments = list(c.values())[0]
        assert type(arguments) == DictConfig, f'expected DictConfig, but got {type(arguments)} in {keyname} arguments'
        globs = globals()
        locs = locals()
        tmp = {}
        for k,v in arguments.items():
            assert type(k) == str, f'expected str, but got {type(k)} in {keyname} argument key: {k}'
            if type(v) == int or type(v) == float or type(v) == bool:
                tmp[k] = v
            elif type(v) == str and v.startswith("'") and v.endswith("'"):
                tmp[k] = v[1:-1]
            elif type(v) == str:
                tmp[k] = eval(v, globs, locs)
            else:
                raise ValueError(f'expected ListConfig, DictConfig or str, but got {type(v)} in {keyname} argument value: {v}')
        arguments = tmp
    else:
        raise ValueError(f'expected DictConfig or str, but got {type(c)} in {c}')
    # get the operation corresponding to the class name
    return _classname2instance(classname, arguments, namespace)


def _classname2instance(
    classname: str,
    arguments: dict,
    namespace: list,
):
    """Return an object corresponding to the class name.

    Examples:
        >>> m = _classname2instance('AdaptiveAvgPool2d', {'output_size': 1}, [nn])
    """
    # find the corresponding operation
    basename = None
    for name in namespace:
        if classname in _get_all_classnames(name):
            basename = name
            break
    assert basename != None, f'cannot found {classname} in {namespace}'
    # return the corresponding module
    return eval(f'basename.{classname}(**arguments)')


def _get_all_classnames(
    module: ModuleType
) -> List[str]:
    """Return a list of all the class names in the module.

    Examples:
        >>> _get_all_classnames(operations)
        ['Identity', 'Pooling', 'ReLUConvBN', 'Serialize', 'Tensor', 'Zero']
    """
    return list(map(lambda x: x[0], inspect.getmembers(module, inspect.isclass)))



def get_nth_idx(
        cfg: DictConfig,
        idx: int
    ) -> dict:
    """Return a list of configurations corresponding to the idx.

    Examples:
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        ... cfgidx = get_nth_idx(cfg, 100):
    """

    codekeys = []
    prodlist = []
    # datasets
    _append_one(cfg['datasets']['train'], codekeys, prodlist, 'datasets')
    _append_one(cfg['datasets']['train'], codekeys, prodlist, 'dataloaders')
    _append_options(cfg['datasets']['train'], codekeys, prodlist)
    # optimizers
    _append_one(cfg['optimizers'], codekeys, prodlist, 'optimizers')
    _append_options(cfg['optimizers'], codekeys, prodlist)
    # criteria 
    _append_one(cfg['criteria'], codekeys, prodlist, 'criteria')
    # schedulers
    _append_one(cfg['schedulers'], codekeys, prodlist, 'schedulers')
    # arch info
    _append_one(cfg['architecture'], codekeys, prodlist, 'num_nodes')
    _append_one(cfg['architecture'], codekeys, prodlist, 'num_cells')
    _append_one(cfg['architecture'], codekeys, prodlist, 'num_blocks')
    _append_one(cfg['architecture'], codekeys, prodlist, 'channels')
    # get codes
    codevalues = list(product(*prodlist))[codesidx]
    return {k:v for k,v in zip(codekeys, codevalues)}


def _append_one(
    subcfg: DictConfig,
    codekeys: list,
    prodlist: list,
    name: str
):
    codekeys.append(name)
    prodlist.append(list(range(len(subcfg[name]))))


def _append_options(
    subcfg: DictConfig,
    codekeys: list,
    prodlist: list,
):
    codekeys += list(subcfg['options'].keys())
    prodlist += [list(range(len(l))) for l in list(subcfg['options'].values())]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
