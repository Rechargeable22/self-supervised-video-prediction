import os

import PIL
import PIL.Image
import imageio
from PIL.ImageDraw import ImageDraw
from torchvision import transforms

from utils.toyDataset import toy_dataset


def generate_gif(src_images, gen_images, name="comparison.gif"):
    """Generates gif to compare generated images with ground truth and saves it.

    :param src_images:  ground truth torch.Size([6, 3, 240, 320])
    :param gen_images:  generated images torch.Size([3, 3, 240, 320])
    :param name:    filename
    """
    animation_images = []
    for i in range(len(src_images)):
        im = transforms.ToPILImage()(src_images[i]).convert("RGB")
        draw = ImageDraw(im)
        bottom = tuple(src_images.size()[2:])[::-1]
        draw.rectangle(((0, 0), bottom), outline='#f00', width=6)
        if i < 3:
            dst = PIL.Image.new('RGB', (im.width + im.width, im.height))
            dst.paste(im, (0, 0))
            dst.paste(im, (im.width, 0))
            animation_images.append(dst)
        else:
            im2 = transforms.ToPILImage()(gen_images[i - 3]).convert("RGB")
            draw2 = ImageDraw(im2)
            bottom = tuple(gen_images.size()[2:])[::-1]
            draw2.rectangle(((0, 0), bottom), outline='#0f0', width=6)
            dst = PIL.Image.new('RGB', (im.width + im2.width, im.height))
            dst.paste(im, (0, 0))
            dst.paste(im2, (im.width, 0))
            animation_images.append(dst)
    imageio.mimsave(name, animation_images, fps=2)


def convgru_filter_sequence(x=1, y=3, path="../../report/pic/after_training"):
    """x and y denote which filter cell we want to plot over the training steps"""
    result = PIL.Image.new("RGB", (84 * 6, 68))
    for i, image_name in enumerate(os.listdir(path)):
        if i < 6:
            im = PIL.Image.open(path + "/" + image_name)
            width, height = im.size
            cropped = im.crop((x, y * (height // 8), x + width // 8 + 2, (y + 1) * (height // 8) + 2))  # (84, 68)
            result.paste(cropped, (i * 84, 0))
    # result.show()
    result.save("gru_filter_after_training.png")


if __name__ == '__main__':
    # images = next(iter(toy_dataset(1)))
    # images = images.squeeze()
    # generate_gif(images, images[3:])
    convgru_filter_sequence()
