__author__ = "Jan Scheffczyk, Oliver Leuschner"
__date__ = "August 2020"

from typing import Any, Iterator
from collections import OrderedDict
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import loggers as pl_log, seed_everything

from project.utils.dssim import DSSIM
from project.utils.convGRU import ConvGRU
from project.utils.toyDataset import moving_ball


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
        for module in self:
            if isinstance(module, ConvGRU):
                res = module(x)  # [torch.Size([8, 64, 64, 80])], [torch.Size([8, 64, 64, 80])] identical
                results.append(res[0])
            else:
                results.append(module(x))
        return torch.cat(results, 1)


class ToyNet(LightningModule):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.lr = 1e-03

        self.initial_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.encode1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.cBlock1 = CenterBlock(
            nn.Conv2d(64, 64, kernel_size=1),
            ConvGRU(64, 64, 3, 1),
            ConvGRU(64, 64, 5, 1),
            ConvGRU(64, 64, 7, 1),
        )
        self.cBlock2 = CenterBlock(
            nn.Conv2d(64, 64, kernel_size=1),
            ConvGRU(64, 64, 3, 1),
            ConvGRU(64, 64, 5, 1),
            ConvGRU(64, 64, 7, 1),
        )
        self.decode1 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 1024, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
        )
        self.finalConvDown = nn.Conv2d(320, 1, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.DSSIMcrit = DSSIM(window_size=11)
        self.MSEcrit = nn.MSELoss()
        self.L1crit = nn.L1Loss()

    def _internal_forward(self, x):
        out = self.initial_conv(x)
        enc_1 = self.encode1(out)

        c1_gru = self.cBlock1(out)
        c2_gru = self.cBlock2(enc_1)

        dec_1 = self.decode1(c2_gru)
        out = self.finalConvDown(torch.cat((dec_1, c1_gru), 1))
        out = self.upsample(out)
        return out

    def forward(self, x):
        result = []
        out = None
        for image_index in range(x.size()[1]):
            if image_index <= x.size()[1] - 4:
                in_img = x[:, image_index, :, :, :]
                out = self._internal_forward(in_img)
            else:
                out = self._internal_forward(out)
                result.append(out)
        return result

    def training_step(self, batch, batch_idx):
        batch = batch[0]  # only images, no label
        images = batch.float()

        out = self(images)

        loss = self._loss(out, images)

        if self.global_step % 20 == 0 or self.global_step < 15:
            self._log_samples(images, out, names=['Images/training_input', 'Images/training_sample'])

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def _loss(self, out, images):
        out1, out2, out3 = out
        images = torch.cat((images[:, 3, :, :, :], images[:, 4, :, :, :], images[:, 5, :, :, :]), 0)
        output = torch.cat((out1, out2, out3), 0)
        loss = self.MSEcrit(output, images)
        return loss

    def _log_samples(self, images, out, names):
        in_frames = images[0, :, :, :, :]
        grid = torchvision.utils.make_grid(in_frames)
        self.logger.experiment.add_image(names[0], grid, self.global_step)
        out = [images[0, 0, :, :, :], images[0, 1, :, :, :], images[0, 2, :, :, :],
               out[0][0], out[1][0], out[2][0]]
        out = torch.clamp(torch.stack(out), 0, 1)
        grid = torchvision.utils.make_grid(out)
        self.logger.experiment.add_image(names[1], grid, self.global_step)

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters()], lr=self.lr, weight_decay=1e-5)


if __name__ == '__main__':
    seed_everything(42)
    # Parameters
    batch_size = 1

    trainloader, testloader = moving_ball(batch_size)
    # Model
    model = ToyNet()
    log = pl_log.TensorBoardLogger('runs/' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    trainer = pl.Trainer(gpus=1 if torch.cuda.is_available() else 0, log_gpu_memory=False, logger=log)
    trainer.fit(model, trainloader)
