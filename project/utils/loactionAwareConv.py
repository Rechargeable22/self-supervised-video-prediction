import torch
import torch.nn as nn

"""
Copied and adapted under 
NO LICENCE but implied by lab task to be used
https://github.com/AIS-Bonn/LocDepVideoPrediction
"""



class LocationAwareConv2d(nn.Conv2d):
    def __init__(self, w, h, in_channels, out_channels, gradient=True,
                 locationAware=True, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        w=int(w+0.5)
        h=int(h+0.5)
        if locationAware:
            self.locationBias = nn.Parameter(torch.zeros(w, h, 3))
            self.locationEncode = torch.autograd.Variable(torch.ones(w, h, 3))
            if gradient:
                for i in range(w):
                    self.locationEncode[i, :, 1] = self.locationEncode[:, i, 0] = i / float(w - 1)

        self.up = torch.nn.Upsample(size=(w, h), mode='bilinear', align_corners=False)
        self.w = w
        self.h = h
        self.locationAware = locationAware

    def forward(self, inputs):
        if self.locationAware:
            if self.locationBias.device != inputs.device:
                self.locationBias = self.locationBias.to(inputs.get_device())
            if self.locationEncode.device != inputs.device:
                self.locationEncode = self.locationEncode.to(inputs.get_device())

            b = self.locationBias * self.locationEncode

        convRes = super().forward(inputs)

        if convRes.shape[2] != self.w and convRes.shape[3] != self.h:
            convRes = self.up(convRes)

        if self.locationAware:
            return convRes + b[:, :, 0] + b[:, :, 1] + b[:, :, 2]
        else:
            return convRes
