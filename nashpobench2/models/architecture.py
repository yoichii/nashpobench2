"""The whole architecture for the benchmark"""
from hydra.experimental import initialize, compose
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from nashpobench2.models.cells import *
from nashpobench2.utils.config import *


class NASHPOModel(nn.Module):
    """A model for NAS-HPO-Bench-II.

    Examples:
        >>> with initialize(config_path='../../config/hydra'):
        ...     cfg = compose(config_name='config')
        ...     c = NASHPOModel(
        ...         cfg,
        ...         {'cellcode': '0|31|213',
        ...         'num_nodes': 0,
        ...         'num_blocks': 0,
        ...         'num_cells': 0,
        ...         'channels': 0
        ...         }
        ...     )
        >>> inputs = torch.ones(3, 3, 16, 16)
        >>> outputs = c(inputs)
        >>> outputs.shape
        torch.Size([3, 10])
    """
    def __init__(
        self,
        cfg: DictConfig,
        codes: dict
    ):
        super().__init__()
        num_blocks = subcfg2instance(
            cfg['architecture'], 'num_blocks', codes['num_blocks']
        )
        num_cells = subcfg2instance(
            cfg['architecture'], 'num_cells', codes['num_cells']
        )
        channels = subcfg2instance(
            cfg['architecture'], 'channels', codes['channels']
        )
        # stack cells
        cells = [AnyCell(cfg, 'FirstCell', **{'out_channels': channels[0]})]
        for i in range(num_blocks):
            for j in range(num_cells):
                cells.append(
                    NormalCell(cfg, codes, **{'in_channels': channels[i]})
                )
            if i != (num_blocks - 1):
                cells.append(
                    AnyCell(cfg,
                        'ReductionCell',
                        **{'in_channels': channels[i],
                            'out_channels': channels[i+1]}
                    )
                )
            else:
                cells.append(
                    AnyCell(cfg,
                        'LastCell',
                        **{'in_channels': channels[i]}
                    )
                )
        self.ops = nn.Sequential(*cells)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        return self.ops(x)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
