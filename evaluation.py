import torch
import torch.nn as nn
from networks.generator import Generator
import argparse
import numpy as np
import torchvision
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from dataset import Vox256_eval, Taichi_eval, TED_eval
from torch.utils import data
from PIL import Image
import lpips


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def load_image(filename, size):
    img = Image.open(filename).convert('RGB')
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def save_video(save_path, name, vid_target_recon, fps=10.0):
    vid = (vid_target_recon.permute(0, 2, 3, 4, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    torchvision.io.write_video(save_path + '%s.mp4' % name, vid[0].cpu(), fps=fps)


def data_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


class Eva(nn.Module):
    def __init__(self, args):
        super(Eva, self).__init__()

        self.args = args

        transform = torchvision.transforms.Compose([
            transforms.Resize((args.size, args.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
        )

        if args.dataset == 'vox':
            path = 'checkpoints/vox.pt'
            dataset = Vox256_eval(transform)
        elif args.dataset == 'taichi':
            path = 'checkpoints/taichi.pt'
            dataset = Taichi_eval(transform)
        elif args.dataset == 'ted':
            path = 'checkpoints/ted.pt'
            dataset = TED_eval(transform)
        else:
            raise NotImplementedError

        os.makedirs(os.path.join(self.save_path, args.dataset), exist_ok=True)

        print('==> loading model')
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).cuda()
        weight = torch.load(path, map_location=lambda storage, loc: storage)['gen']
        self.gen.load_state_dict(weight)
        self.gen.eval()

        print('==> loading data')
        self.loader = data.DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            drop_last=False,
        )

        self.loss_fn = lpips.LPIPS(net='alex').cuda()

    def run(self):

        loss_list = []
        loss_lpips = []
        for idx, (vid_name, vid) in tqdm(enumerate(self.loader)):

            with torch.no_grad():

                vid_real = []
                vid_recon = []
                img_source = vid[0].cuda()
                for img_target in vid:
                    img_target = img_target.cuda()
                    img_recon = self.gen(img_source, img_target)
                    vid_recon.append(img_recon.unsqueeze(2))
                    vid_real.append(img_target.unsqueeze(2))

                vid_recon = torch.cat(vid_recon, dim=2)
                vid_real = torch.cat(vid_real, dim=2)

                loss_list.append(torch.abs(0.5 * (vid_recon.clamp(-1, 1) - vid_real)).mean().cpu().numpy())
                vid_real = vid_real.permute(0, 2, 1, 3, 4).squeeze(0)
                vid_recon = vid_recon.permute(0, 2, 1, 3, 4).squeeze(0)
                loss_lpips.append(self.loss_fn.forward(vid_real, vid_recon.clamp(-1, 1)).mean().cpu().detach().numpy())

        print("reconstruction loss: %s" % np.mean(loss_list))
        print("lpips loss: %s" % np.mean(loss_lpips))


if __name__ == '__main__':
    # training params
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--latent_dim_style", type=int, default=512)
    parser.add_argument("--latent_dim_motion", type=int, default=20)
    parser.add_argument("--dataset", type=str, choices=['vox', 'taichi', 'ted'], default='')
    parser.add_argument("--save_path", type=str, default='./evaluation_res')
    args = parser.parse_args()

    demo = Eva(args)
    demo.run()
