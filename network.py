import torch.nn as nn
from torch.nn import Module
from torch import Tensor

class ConvBlock(Module):
    """Convolution block used in the discriminator of DCGAN."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

class ConvBlockTranspose(Module):
    """Transposed Convolution block used in the generator of DCGAN."""
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
        """
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=stride-1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Discriminator(Module):
    """Discriminator of the DCGAN"""
    def __init__(self, layers: list[int]) -> None:
        """

        Args:
            layers (list[int]): List, in which each elements corresponds to the number of channels in the given layer.
                                length of `layers` is the depth of the network.
        """
        super().__init__()

        self.D = nn.Sequential(
            *nn.ModuleList([ConvBlock(i, j) for i, j in zip(layers[:-1], layers[1:])]),
            nn.Conv2d(layers[-1], 1, kernel_size=4, stride=1, bias=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Sigmoid()
        )


    def forward(self, x: Tensor) -> Tensor:
        return self.D(x).view(-1)


class Generator(Module):
    """Generator of the DCGAN"""
    def __init__(self, layers: list[int]) -> None:
        """
        Args:
            layers (list[int]): List, in which each elements corresponds to the number of channels in the given layer.
                                length of `layers` is the depth of the network.
        """
        super().__init__()

        self.G = nn.Sequential(
            *nn.ModuleList([ConvBlockTranspose(i, j, 1 if l_id == 0 else 2) for l_id, (i, j) in enumerate(zip(layers[:-1], layers[1:]))]),
            nn.ConvTranspose2d(in_channels=layers[-1], out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.G(x)


def weights_init(module: Module):
    """Initialize convolutional kernels and batch norms.

    Args:
        module (Module): Module, which should be initialized.
    """
    try:
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    except AttributeError:
        return
