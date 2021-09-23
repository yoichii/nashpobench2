"""Operations for creating a cell."""

from typing import Union, Tuple

import torch
from torch import Tensor
import torch.nn as nn


class ReLUConvBN(nn.Module):
    """A operation sequence of ReLU, convolution, and batch normalization.

    Examples:
        >>> m = ReLUConvBN(64, 64, 3)
        >>> inputs = torch.ones((3, 64, 16, 16))
        >>> outputs = m(inputs)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int,Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias = True
    ):
        super().__init__()
        self.ops = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        return self.ops(x)


class Pooling(nn.Module):
    """A pooling operation.

    If the number of input channels doesn't equal to the number of output channels, apply ReLUConvBN first to adjust the number.

    Examples:
        >>> m = Pooling(64, 64, padding=1)
        >>> inputs = torch.ones((3, 64, 16, 16))
        >>> outputs = m(inputs)
        >>> outputs.shape
        torch.Size([3, 64, 16, 16])

        >>> m = Pooling(64, 64, 'max', padding=1)
        >>> inputs = torch.ones((3, 64, 16, 16))
        >>> outputs = m(inputs)
        >>> outputs.shape
        torch.Size([3, 64, 16, 16])
        
        >>> m = Pooling(64, 128, stride=2)
        >>> inputs = torch.ones((3, 64, 16, 16))
        >>> outputs = m(inputs)
        >>> assert list(outputs.shape) == [3, 128, 7, 7]
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: str = 'avg',
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
    ):
        super().__init__()
        if mode == 'avg':
            self.ops : Union[nn.AvgPool2d, nn.MaxPool2d] = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
        elif mode == 'max':
            self.ops = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        else:
            raise ValueError(f'Invalid mode {mode} in Pooling')
        if in_channels == out_channels:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        if self.preprocess:
            x = self.preprocess(x)
        return self.ops(x)


class Identity(nn.Module):
    """A identity operation. No transformation is applied.

    Example:
        >>> m = Identity()
        >>> inputs = torch.ones((3, 64, 16, 16))
        >>> torch.equal(m(inputs), inputs)
        True
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        return x


class Zero(nn.Module):
    """A zero operation. Output tensor is zeros.

    Examples:
        >>> m = Zero(64, 64)
        >>> inputs = torch.ones((3, 64, 16, 16))
        >>> torch.equal(m(inputs), torch.zeros((3, 64, 16, 16)))
        True

        >>> m = Zero(64, 64, 2)
        >>> inputs = torch.ones((3, 64, 16, 16))
        >>> torch.equal(m(inputs), torch.zeros((3, 64, 8, 8)))
        True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation"""
        if self.in_channels == self.out_channels:
            if self.stride == 1:
                #return x.mul(0.)
                return torch.zeros_like(x)
            else:
                #return x[:,:,::self.stride,::self.stride].mul(0.)
                return torch.zeros_like(x[:,:,::self.stride,::self.stride])
        else:
            shape = list(x.shape)
            shape[1] = self.out_channels
            return x.new_zeros(shape, dtype=x.dtype, device=x.device)


class Serialize(nn.Module):
    """ a 3d-to-1d operation.

    Examples:
        >>> inputs = torch.ones(3, 64, 16, 16)
        >>> m = Serialize()
        >>> outputs = m(inputs)
        >>> assert list(outputs.shape) == [3, 64*16*16]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
