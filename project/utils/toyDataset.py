import os

import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video

from project.utils.UCFLoader import UCF101


class MovingBall(Dataset):
    """Loads the moving ball dataset for debugging."""

    def __init__(self, left_to_right=True, transform=None):
        self.left_to_right = left_to_right
        self.transform = transform
        self.frames = [Image.open(f"data/moving_ball/ball_l2r_{i}.jpg") for i in range(1, 7)]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        fake_label = 1
        if self.transform:
            frames_post = []
            for idx in range(6):
                #frames_post.append(self.transform(self.frames[i]))
                if idx==0:
                    for i in range(6):
                        frames_post.append(self.transform(self.frames[i]))
                else:
                    for i in range(5,-1,-1):
                        frames_post.append(self.transform(self.frames[i]))
        frames_out = torch.stack(frames_post, 0)
        return frames_out, fake_label

def moving_ball(batchsize, transform=None):
    if not transform:
        transform = transforms.Compose([
            #transforms.Grayscale(),
            transforms.ToTensor(),
        ])
    """Returns a dataloader that wraps the first image of the UCF-101 dataset"""
    dataset = MovingBall(transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1)
    return dataloader, dataloader


class ToyUCF101(Dataset):
    """Loads one video"""

    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.vframes, _, _ = read_video(self.video_path, pts_unit="sec")  # torch.Size([131, 240, 320, 3])

    def __len__(self):
        return len(self.vframes) // 24

    def __getitem__(self, idx):
        fake_label = 1
        idx = idx * 12
        sample = self.vframes[idx: idx + 24: 4]
        sample = sample.permute(0, 3, 1, 2)

        if self.transform:
            sample_2 = []
            for i in range(6):
                sample_2.append(self.transform(sample[i]))
            sample = torch.stack(sample_2, 0)

        return sample, fake_label


def toy_dataset(batchsize, transform=None):
    """Returns a dataloader that wraps the first image of the UCF-101 dataset"""
    dataset = ToyUCF101("data/UCF-101/Archery/v_Archery_g01_c01.avi", transform=transform)
    # dataset = ToyUCF101("data/UCF-101/Fencing/v_Fencing_g01_c03.avi", transform=transform)
    # dataset = ToyUCF101("data/validation/hands.mp4", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=1)
    return dataloader, dataloader


def load_ucf_dataset(batch_size, transform, step_between_clips=20, frame_rate=3, frames_per_clip=6,percent=0.01):
    # Load datasets
    trainset = UCF101("data/UCF-101", "data/UCF101_Action_detection_splits", transform=transform, frames_per_clip=6,
                      step_between_clips=step_between_clips, train=True, num_workers=16, frame_rate=frame_rate)
    testset = UCF101("data/UCF-101", "data/UCF101_Action_detection_splits", frames_per_clip=6, train=False,
                     step_between_clips=step_between_clips, transform=transform, num_workers=16, frame_rate=frame_rate)
    # writer = SummaryWriter()
    # Data Loader
    # trainloader = toy_dataset(batch_size, transform)
    print(f"Taking {len(trainset)*percent} training samples")
    subset_indices = [i for i in torch.randint(0,len(trainset),((int)(len(trainset)*percent),))]
    trainset = torch.utils.data.Subset(trainset, subset_indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16,
                                              pin_memory=True)

    subset_indices = [i for i in torch.randint(0, len(testset), ((int)(len(testset)*percent),))]
    testset = torch.utils.data.Subset(testset, subset_indices)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
    return trainloader, testloader


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataloader, _ = moving_ball(batchsize=1, transform=transform)
    for i, images in enumerate(dataloader):
        grid = torchvision.utils.make_grid(images[0])
        torchvision.utils.save_image(grid, "results/one_frame_predict_3days_in.png")
