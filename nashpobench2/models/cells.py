"""Cells for creating models"""
import sys
from functools import reduce
from types import ModuleType
from typing import Union, Tuple, List, Optional, Type

from hydra.experimental import compose, initialize
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from nashpobench2.models.operations import ReLUConvBN, Pooling, Identity, Zero, Serialize
from nashpobench2.models import operations
from nashpobench2.utils.config import subcfg2instance


class NormalCell(nn.Module):
    """A cell to be stacked normally.

    Examples:
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        ...     c = NormalCell(cfg, {'cellcode': '2|22|222', 'num_nodes': 0}, **{'in_channels': 32})
        >>> inputs = torch.ones(3, 32, 16, 16)
        >>> outputs = c(inputs)
        >>> (outputs == torch.ones(3, 32, 16, 16)*4).all()
        tensor(True)
    """
    def __init__(
        self,
        cfg: DictConfig,
        codes: dict,
        **kwargs: dict
    ):
        super().__init__()
        self.num_nodes = len(cfg['operations'])
        self.modulelist = _cellcode2op(cfg, codes, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        nodes = [x]
        for i in range(1,self.num_nodes):
            inter_nodes = []
            for j in range(i):
                inter_nodes.append(self.modulelist[int((i-1)*i/2) + j](nodes[j]))
            nodes.append(reduce(torch.add, inter_nodes))
        return nodes[-1]


class AnyCell(nn.Module):
    """cells like FirstCell or LastCell

    Examples:
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        >>> c = AnyCell(cfg, 'FirstCell', **{'out_channels': 16})
        >>> inputs = torch.ones(3, 3, 16, 16)
        >>> outputs = c(inputs)
        >>> outputs.shape
        torch.Size([3, 16, 16, 16])
        >>> c = AnyCell(cfg, 'ReductionCell', **{'in_channels':32, 'out_channels':64})
        >>> inputs = torch.ones(3, 32, 16, 16)
        >>> outputs = c(inputs)
        >>> outputs.shape
        torch.Size([3, 64, 8, 8])
        >>> c = AnyCell(cfg, 'LastCell', **{'in_channels': 64})
        >>> inputs = torch.ones(3, 64, 4, 4)
        >>> outputs = c(inputs)
        >>> outputs.shape
        torch.Size([3, 10])
    """
    def __init__(self,
        cfg: DictConfig,
        cellname: str,
        **kwargs: dict
    ):
        super().__init__()
        self.ops = _cfg2ops(cfg, cellname, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        return self.ops(x)


class ResidualBlock(nn.Module):
    """Return the operation list corresponding to the cellcode.

    Examples:
        >>> b = ResidualBlock(16, 32, 2, 1)
        >>> inputs = torch.ones(3, 16, 100, 100)
        >>> outputs = b(inputs)
        >>> outputs.shape
        torch.Size([3, 32, 50, 50])
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        assert stride == 1 or stride == 2
        self.conv1 = ReLUConvBN(in_channels, out_channels, 3, stride=stride, padding=padding)
        self.conv2 = ReLUConvBN(out_channels, out_channels, 3, 1, padding=padding)
        if stride == 2:
            self.downsample : Optional[Union[nn.Sequential, ReLUConvBN]] = nn.Sequential(
                nn.AvgPool2d(2, stride=stride),
                nn.Conv2d(in_channels, out_channels, 1)
            )
        elif in_channels != out_channels:
            self.downsample = ReLUConvBN(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        basic = self.conv1(x)
        basic = self.conv2(basic)
        if self.downsample is None:
            residual = x
        else:
            residual = self.downsample(x)
        return basic + residual


def _cellcode2op(
    cfg: DictConfig,
    codes: dict,
    **kwargs: Optional[dict]
) -> nn.ModuleList:
    """Return the operation list corresponding to the archcode.

    Examples:
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        ...     o = _cellcode2op(cfg, {'cellcode': '1|32|202', 'num_nodes': 0}, **{'in_channels': 64, 'out_channels': 128})
    """
    cellcode_list = list( map(lambda x: int(x), reduce(lambda x,y:x+y, codes['cellcode'].split('|')) ) )
    num_nodes = subcfg2instance(cfg['architecture'], 'num_nodes', codes['num_nodes'])
    assert len(cellcode_list) == int(num_nodes * (num_nodes-1) / 2),\
            f'expected {int(num_nodes * (num_nodes-1) / 2)} length, but got {len(cellcode_list)} in cellcode'
    edges = []
    for edgecode in cellcode_list:
        edges.append(subcfg2instance(cfg, 'operations', edgecode, [nn, sys.modules[__name__], operations], **kwargs))
    return nn.ModuleList(edges)


def _cfg2ops(
    cfg: DictConfig,
    subname: str,
    **kwargs: Optional[dict]
) -> nn.Sequential:
    """Return a sequence of operations corresponding to the config file.

    Examples:
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        ...     o = _cfg2ops(cfg, 'FirstCell', **{'out_channels': 16})
        ...     o = _cfg2ops(cfg, 'ReductionCell', **{'in_channels':32, 'out_channels':64})
        ...     o = _cfg2ops(cfg, 'LastCell', **{'in_channels': 64})
    """
    model_cfg = cfg['architecture']
    oplist = []
    for idx in range(len(model_cfg[subname])):
        oplist.append( subcfg2instance(model_cfg, subname, idx, [nn, sys.modules[__name__], operations], **kwargs) )
    return nn.Sequential(*oplist)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
