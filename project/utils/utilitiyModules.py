__author__ = "Jan Scheffczyk, Oliver Leuschner"
__date__ = "August 2020"

from typing import Any, Iterator
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision import models
from pytorch_lightning.core.lightning import LightningModule

from project.utils.dssim import DSSIM
from project.utils.convGRU2 import ConvGRU
from project.utils.loactionAwareConv import LocationAwareConv2d


class ResNet(nn.Module):
    def __init__(self, freeze=True):
        super(ResNet, self).__init__()
        self.pretrained_resnet = models.resnet18(pretrained=True)
        self.layers = list(self.pretrained_resnet.children())
        self.layers = self.layers[:-1]
        self.ff = nn.Linear(512, 10)
        for param in self.pretrained_resnet.parameters():
            param.requires_grad = not freeze
        self.layer_outputs = []

    def forward(self, x):
        self.layer_outputs = []
        for ii, layer in enumerate(self.layers):
            # only layer 0,4,5,6,7 are relevant for us
            x = layer(x)
            if ii in [0, 4, 5, 6, 7]:
                self.layer_outputs.append(x)
        # x = self.ff(torch.flatten(x, 1))
        return self.layer_outputs

    def unfreeze(self, unfreeze):
        for param in self.pretrained_resnet.parameters():
            param.requires_grad = unfreeze


class CenterBlock(nn.Module):
    def __init__(self, *args: Any):
        super(CenterBlock, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def __iter__(self) -> Iterator[nn.Module]:
        return iter(self._modules.values())

    def forward(self, x):
        results = []
        gru_out = []
        for module in self:
            if isinstance(module, ConvGRU):
                res, hid = module(x)  # [torch.Size([8, 64, 64, 80])], [torch.Size([8, 64, 64, 80])] identical
                results.append(hid[-1])
                gru_out.append(hid[-1])
            else:
                results.append(module(x))
        return torch.cat(results, 1), gru_out
