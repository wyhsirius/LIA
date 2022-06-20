import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def data_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]

    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def save_video(vid_gen, path, fps):
    vid = vid_gen.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type('torch.ByteTensor')

    torchvision.io.write_video(path, vid[0], fps=fps)


class Demo(nn.Module):
    def __init__(self, args):
        super(Demo, self).__init__()

        if args.model == 'vox':
            model_path = 'checkpoints/vox.pt'
        elif args.model == 'taichi':
            model_path = 'checkpoints/taichi.pt'
        elif args.model == 'ted':
            model_path = 'checkpoints/ted.pt'
        else:
            raise NotImplementedError

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(model_path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.save_folder = os.path.join(args.save_folder, args.model, Path(args.img_path).stem)
        os.makedirs(self.save_folder, exist_ok=True)
        self.img_source = data_preprocessing(args.img_path, args.size).cuda()

    def run(self, args):

        with torch.no_grad():

            h_start = self.gen.enc.enc_motion(self.img_source)
            wa, _, feat = self.gen.enc(self.img_source, None)

            for i in tqdm(range(args.latent_dim_motion)):

                alpha_zero = torch.zeros(1, args.latent_dim_motion).cuda()
                alpha = torch.zeros(1, args.latent_dim_motion).cuda()

                vid_target_recon = []
                for j in range(args.range):
                    delta = -args.degree + j * (2 * args.degree / args.range)
                    alpha[:, i] = 1.0 * delta

                    img_recon = self.gen.synthesis(wa, [h_start, alpha, alpha_zero], feat)
                    vid_target_recon.append(img_recon.unsqueeze(2))

                vid_target_recon = torch.cat(vid_target_recon, dim=2)

                save_path = os.path.join(self.save_folder, '%02d.mp4' % i)
                save_video(vid_target_recon, save_path, args.save_fps)


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--model", type=str, choices=['vox', 'taichi', 'ted'], default='')
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--degree", type=float, default=3.0)
    parser.add_argument("--range", type=float, default=16)
    parser.add_argument("--save_fps", type=float, default=10)
    parser.add_argument("--img_path", type=str, default='')
    parser.add_argument("--save_folder", type=str, default='./res_manipulation')
    args = parser.parse_args()

    demo = Demo(args)
    demo.run(args)
