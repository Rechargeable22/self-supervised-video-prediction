__author__ = 'Jan Scheffczyk, Oliver Leuschner'
__date__ = 'August 2020'

import torch.backends.cudnn
import torchvision.transforms as transforms
import torchvision

from project.one_frame_model import SingleFrameVideoPredictorModel
from project.one_frame_model_full_res import SingleFrameFullResVideoPredictorModel
from project.utils.toyDataset import toy_dataset, load_ucf_dataset
from utils.generateGif import generate_gif

h = 64
w = 96
batch_size = 16
# test
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((h, w)),
    transforms.ToTensor(),
])
# trainloader, testloader = toy_dataset(batch_size, transform)
trainloader, testloader = load_ucf_dataset(batch_size, transform, step_between_clips=1, frame_rate=10,
                                           frames_per_clip=50, percent=0.01)
model = SingleFrameVideoPredictorModel(h, w)
# model = SingleFrameFullResVideoPredictorModel(h, w)
model = model.load_from_checkpoint("model_checkpoints/one_frame_bak2.chkp", image_width=w, image_height=h)
if torch.cuda.is_available():
    model.cuda()
model.eval()
name = "comp2/random"
for i, images in enumerate(trainloader):
    if torch.cuda.is_available():
        images = images[0].float().cuda()
    else:
        images = images[0].float()
    size = images.size()
    in_frames = images[:, :, :, :, :]
    out1 = torch.unsqueeze(model(in_frames)[0], 1)
    in_frames = torch.cat([in_frames, out1], dim=1)
    out2 = torch.unsqueeze(model(in_frames)[0], 1)
    in_frames = torch.cat([in_frames, out2], dim=1)
    out3 = torch.unsqueeze(model(in_frames)[0], 1)

    # in_frames = images[:, :, :, :, :]
    # # in_frames = in_frames.view([-1, size[2], size[3], size[4]])
    # out = [out1[:, 0], out2[:, 0], out3[:, 0]]
    # out = torch.stack(out).permute(1, 0, 2, 3, 4)
    # for bat in range(batch_size-1):
    #     generate_gif(in_frames[bat], out[bat], f"results/gifs/gif_batch_{i}_{bat}.gif")

    in_frames = images[:, :, :, :, :].data
    in_frames = in_frames.view([-1, size[2], size[3], size[4]])
    grid = torchvision.utils.make_grid(in_frames, nrow=6)
    torchvision.utils.save_image(grid, f"results/{name}_{i}_in.png")

    out = [out1[:, 0], out2[:, 0], out3[:, 0]]
    out = torch.stack(out).permute(1, 0, 2, 3, 4)
    size = out.size()
    out = out.reshape([-1, size[2], size[3], size[4]])
    grid = torchvision.utils.make_grid(out, nrow=3)
    torchvision.utils.save_image(grid, f"results/{name}_{i}_out.png")

    in_frames = images[:, 3:, :, :, :]
    out = [out1[:, 0], out2[:, 0], out3[:, 0]]
    out = torch.stack(out).permute(1, 0, 2, 3, 4)
    out = torch.cat([in_frames, out], dim=1)
    size = out.size()
    out = out.reshape([-1, size[2], size[3], size[4]])
    grid = torchvision.utils.make_grid(out, nrow=6)
    torchvision.utils.save_image(grid, f"results/{name}_{i}_comb.png")

# for i, images in enumerate(trainloader):
#     if i == 0:
#         if torch.cuda.is_available():
#             images = images[0].float().cuda()
#         else:
#             images = images[0].float()
#         in_frames = images[:, 3:, :, :, :]
#         out1 = torch.unsqueeze(model(in_frames)[0], 1)
#         in_frames = torch.cat([in_frames, out1], dim=1)
#         out2 = torch.unsqueeze(model(in_frames)[0], 1)
#         in_frames = torch.cat([in_frames, out2], dim=1)
#         out3 = torch.unsqueeze(model(in_frames)[0], 1)
#
#         in_frames = images[0, :, :, :, :].data
#
#         # grid = torchvision.utils.make_grid(in_frames)
#         # torchvision.utils.save_image(grid, "results/one_frame_predict_3days_art_in.png")
#         out = [out1[0][0], out2[0][0], out3[0][0]]
#         out = torch.stack(out)
#         # grid = torchvision.utils.make_grid(out)
#         # torchvision.utils.save_image(grid, "results/one_frame_predict_3days_art_out.png")
#         generate_gif(in_frames, out, "results/comparison_one_frame_3.gif")
