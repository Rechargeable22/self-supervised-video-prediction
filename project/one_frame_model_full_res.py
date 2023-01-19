__author__ = "Jan Scheffczyk, Oliver Leuschner"
__date__ = "Oktober 2020"

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
from project.utils.utilitiyModules import ResNet, CenterBlock

class SingleFrameFullResVideoPredictorModel(LightningModule):
    def __init__(self, image_height, image_width, mse_loss=1, l1_loss=1, ssim_loss=18,normalize=True):
        super(SingleFrameFullResVideoPredictorModel, self).__init__()
        self.height = image_height
        self.width = image_width
        self.lr = 3e-03
        self.normalize=normalize

        sum= mse_loss + l1_loss + ssim_loss
        self.mse_loss = mse_loss/sum
        self.l1_loss = l1_loss/sum
        self.ssim_loss= ssim_loss/sum
        self.resNet = ResNet(freeze=True)
        w = image_width
        h = image_height
        self.cBlock1 = CenterBlock(
            nn.Conv2d(64, 16, kernel_size=1),
            ConvGRU([h, w], 64, 16, 3, 1),
            ConvGRU([h, w], 64, 16, 5, 1),
            ConvGRU([h, w], 64, 16, 7, 1),
        )
        w = w // 2
        h = h // 2
        self.cBlock2 = CenterBlock(
            nn.Conv2d(64, 32, kernel_size=1),
            ConvGRU([h, w], 64, 32, 3, 1),
            ConvGRU([h, w], 64, 32, 5, 1),
            ConvGRU([h, w], 64, 32, 7, 1),
        )
        w = w // 2
        h = h // 2
        self.cBlock3 = CenterBlock(
            LocationAwareConv2d(w=h, h=w,in_channels=64, out_channels=64),
            ConvGRU([h, w], 64, 64, 3, 1),
            ConvGRU([h, w], 64, 64, 5, 1),
            ConvGRU([h, w], 64, 64, 7, 1),
        )
        w = w // 2
        h = h // 2
        self.cBlock4 = CenterBlock(
            LocationAwareConv2d(w=h, h=w, in_channels=128, out_channels=64),
            ConvGRU([h, w], 128, 64, 3, 1),
            ConvGRU([h, w], 128, 64, 5, 1),
            ConvGRU([h, w], 128, 64, 7, 1),
        )

        self.lBlock1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(192),
            nn.Conv2d(192, 1024, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
        )
        self.lBlock2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(320),
            nn.Conv2d(320, 1024, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
        )
        self.lBlock3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(320),
            nn.Conv2d(320, 1024, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
        )
        self.lBlock4 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.PixelShuffle(4),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.last_pixel_shuff = nn.PixelShuffle(2)
        self.firstConv = nn.Conv2d(3, 64, 7,1,3)
        self.finalConvDown = nn.Conv2d(128, 3, 1)
        #self.finalConvDown = nn.Conv2d(80, 3, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.DSSIMcrit = DSSIM(window_size=11)
        self.MSEcrit = nn.MSELoss()
        self.L1crit = nn.L1Loss()
        self.relu6=nn.ReLU6()

        self.inv_normalize = torchvision.transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255]
)

    def _internal_forward(self, x, image_index, hidden=None):
        res_out = self.resNet(x)

        c1_res, gru_1 = self.cBlock1.forward(self.firstConv(x))
        c2_res, gru_2 = self.cBlock2.forward(res_out[0])
        c3_res, gru_3 = self.cBlock3.forward(res_out[1])
        c4_res, gru_4 = self.cBlock4.forward(res_out[2])

        #list(self.cBlock1.modules())[2].save_hidden(f"convgru1_{image_index}")

        l4_res = self.lBlock4.forward(res_out[4])
        l3_res = self.lBlock3.forward(torch.cat((c4_res, l4_res), 1))
        l2_res = self.lBlock2.forward(torch.cat((c3_res, l3_res), 1))
        l1_res = self.lBlock1.forward(torch.cat((c2_res, l2_res), 1))

        out = torch.cat((c1_res, l1_res), 1)
        out = self.finalConvDown(out)
        #out = self.upsample(out)
        # out = self.last_pixel_shuff(out)
        return out, [gru_1, gru_2, gru_3, gru_4]

    def forward(self, x):  # torch.Size([bs, seq_len, 3, 128, 160])
        result = []
        out = None
        # hidden = [None] * 12  # since we have 12 convgru
        self._reset_hidden(x.size()[0])
        for image_index in range(x.size()[1]):
            if image_index <= x.size()[1] - 2:
                in_img = x[:, image_index, :, :, :]
                out, gru = self._internal_forward(in_img, image_index)
            else:
                out, gru = self._internal_forward(out, image_index)
                result.append(out)
        self.hidden = gru  # we're only interested in the gru states after the whole sequence is passed
        return result

    def training_step(self, batch, batch_idx):
        batch = batch[0]  # only images, no label
        images = batch.float()

        if torch.cuda.is_available():
            images = images.float().cuda()  # do we need that with lighnting?
            out = self(images)
        else:
            out = self(images)

        loss = self._loss(out, images)

        if self.global_step % 50 == 0 or self.global_step < 15:
            self._log_samples(images, out, names=['Images/training_input', 'Images/training_predic'])

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def _loss(self, out, images):
        out_tensor = out[0]

        loss = self.DSSIMcrit(out_tensor, images[:, 5, :, :, :])
        loss2 = self.MSEcrit(out_tensor, images[:, 5, :, :, :])
        loss3 = self.L1crit(out_tensor, images[:, 5, :, :, :])

        loss += self.mse_loss * loss2 + self.l1_loss * loss3
        return loss

    def _log_samples(self, images, out, names):
        in_frames = images[0, :, :, :, :].data

        if self.normalize:
            for i in range(len(in_frames)):
                in_frames[i]=torch.clamp(self.inv_normalize(in_frames[i]),0,1)

        grid = torchvision.utils.make_grid(in_frames)
        self.logger.experiment.add_image(names[0], grid, self.global_step)
        if self.normalize:
            out = [self.inv_normalize(out[0][0].data),]
        else:
            out = [out[0][0].data, ]
        out = torch.clamp(torch.stack(out),0,1)
        grid = torchvision.utils.make_grid(out)
        self.logger.experiment.add_image(names[1], grid, self.global_step)
        self.logger.experiment.add_scalar('Lr',self.trainer.optimizers[0].param_groups[0]['lr'],self.global_step)

    def validation_step(self, batch, batch_nb):
        batch = batch[0]  # only images, no label
        images = batch.float()
        if torch.cuda.is_available():
            images = images.float().cuda()
            out = self(images)
        else:
            out = self(images)

        loss = self._loss(out, images)

        self._log_samples(images, out, names=['Images/val_goal', 'Images/val_output'])

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def _reset_hidden(self, bs):
        for m in self.modules():
            if isinstance(m, ConvGRU):
                m.reset_hidden(bs)

    def configure_optimizers(self):
        opti = [torch.optim.AdamW([p for p in self.parameters()], lr=self.lr, weight_decay=1e-5)]
        seq = [{
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opti[0],1),
            'monitor': 'val_recall',  # Default: val_loss
            'interval': 'epoch',
            'frequency': 1
        }]
        return opti,seq



