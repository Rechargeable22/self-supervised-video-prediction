__author__ = 'Jan Scheffczyk, Oliver Leuschner'
__date__ = 'August 2020'

import torch
import torch.backends.cudnn
import torchvision.transforms as transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_log
from datetime import datetime
import torchvision

from project.LVLN_model import VideoPredictorModel
from project.utils.UCFLoader import UCF101
from project.utils.toyDataset import toy_dataset
from utils.generateGif import generate_gif

h = 128
w = 160
batch_size = 16
# test
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((h, w)),
    transforms.ToTensor(),
])
trainloader, testloader = toy_dataset(batch_size, transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0,
# pin_memory=True)
model = VideoPredictorModel(h, w)
model = model.load_from_checkpoint("model_checkpoints/adam_0.25.chkp", image_width=w, image_height=h)
if torch.cuda.is_available():
    model.cuda()
model.eval()
for i, images in enumerate(trainloader):
    if i == 0:
        if torch.cuda.is_available():
            images = images[0].float().cuda()
        else:
            images = images[0].float()
        out = model(images)
        in_frames = images[0, 3:, :, :, :]
        in_frames = images[0, :, :, :, :]
        # grid = torchvision.utils.make_grid(in_frames)
        # torchvision.utils.save_image(grid,"results/in.png")
        out = [out[0][0], out[1][0], out[2][0]]
        out = torch.stack(out)
        # grid = torchvision.utils.make_grid(out)
        # torchvision.utils.save_image(grid,"results/out.png")
        generate_gif(in_frames, out, "results/comparison_regular2.gif")
