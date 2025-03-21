import torch
from networks.discriminator import Discriminator
from networks.generator import Generator
import torch.nn.functional as F
from torch import nn, optim
import os
from vgg19 import VGGLoss
from torch.nn.parallel import DistributedDataParallel as DDP


def requires_grad(net, flag=True):
    for p in net.parameters():
        p.requires_grad = flag


def batch_crop_and_resize(img_batch, mask_batch, target_size=(64, 64)):
    """
    对于形状为 [B, C, H, W] 的 img_batch 和形状为 [B, H, W] 或 [B, 1, H, W] 的 mask_batch，
    对每个样本利用 mask 裁剪出最小包围框，再 resize 到 target_size，
    返回 tensor，形状为 [B, C, target_size[0], target_size[1]]。
    """
    B, C, H, W = img_batch.shape
    cropped_list = []
    for i in range(B):
        img = img_batch[i]  # [C, H, W]
        mask = mask_batch[i]  # [H, W] 或 [1, H, W]
        if mask.dim() == 3:
            mask = mask.squeeze(0)  # 转为 [H, W]
        foreground = (mask > 0)
        coords = torch.where(foreground)
        if coords[0].numel() == 0:
            # 如果 mask 全为0，则直接使用原图
            cropped = img
        else:
            y_min, y_max = coords[0].min(), coords[0].max()
            x_min, x_max = coords[1].min(), coords[1].max()
            cropped = img[:, y_min:y_max+1, x_min:x_max+1]
        # resize cropped 到 target_size，注意 F.interpolate 要求4D张量
        cropped = cropped.unsqueeze(0)  # [1, C, h, w]
        resized = F.interpolate(cropped, size=target_size, mode='bilinear', align_corners=False)
        cropped_list.append(resized.squeeze(0))  # [C, target_size[0], target_size[1]]
    return torch.stack(cropped_list, dim=0)  # [B, C, target_size[0], target_size[1]]

class Trainer(nn.Module):
    def __init__(self, args, device, rank):
        super(Trainer, self).__init__()

        self.args = args
        self.batch_size = args.batch_size

        # 定义多个判别器：全图和局部（face, hands, lips, eyes）
        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).to(device)
        self.dis = Discriminator(args.size, args.channel_multiplier).to(device)
        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.dis = DDP(self.dis_main, device_ids=[rank], find_unused_parameters=True)
        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        self.d_optim = optim.Adam(
            self.dis.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
        )

        self.criterion_vgg = VGGLoss().to(device)

    def g_nonsaturating_loss(self, fake_pred):
        return F.softplus(-fake_pred).mean()

    def d_nonsaturating_loss(self, fake_pred, real_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()

    def gen_update(self, img_source, img_target, face_mask, hands_mask, lips_mask, eyes):
        self.gen.train()
        self.gen.zero_grad()

        requires_grad(self.gen, True)
        requires_grad(self.dis, False)

        # 生成器前向得到全图重建
        img_target_recon = self.gen(img_source, img_target)
        img_recon_pred = self.dis(img_target_recon)

        # VGG Loss (全图及各局部乘掩码方式)
        vgg_loss_base = self.criterion_vgg(img_target_recon, img_target).mean()
        face_vgg_loss  = self.criterion_vgg(img_target_recon * face_mask, img_target * face_mask).mean()
        hands_vgg_loss = self.criterion_vgg(img_target_recon * hands_mask, img_target * hands_mask).mean()
        lips_vgg_loss  = self.criterion_vgg(img_target_recon * lips_mask,  img_target * lips_mask).mean()
        eyes_vgg_loss  = self.criterion_vgg(img_target_recon * eyes,   img_target * eyes).mean()
        vgg_loss = vgg_loss_base + 10 * (face_vgg_loss + 2.0 * hands_vgg_loss + 4.0 * lips_vgg_loss + 2.0 * eyes_vgg_loss)

        # L1 Loss
        l1_loss = F.l1_loss(img_target_recon, img_target)

        gan_main_loss  = self.g_nonsaturating_loss(img_recon_pred)
        gan_g_loss = gan_main_loss

        # 总生成器损失
        g_loss = gan_g_loss + vgg_loss + l1_loss

        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, img_target_recon

    def dis_update(self, img_real, img_recon):
        self.dis.zero_grad()

        requires_grad(self.gen, False)
        requires_grad(self.dis, True)

        real_img_pred = self.dis(img_real)
        recon_img_pred = self.dis(img_recon.detach())

        d_loss = self.d_nonsaturating_loss(recon_img_pred, real_img_pred)
        d_loss.backward()
        self.d_optim.step()

        return d_loss

    def sample(self, img_source, img_target):
        with torch.no_grad():
            self.gen.eval()
            img_recon = self.gen(img_source, img_target)
            img_source_ref = self.gen(img_source, None)
        return img_recon, img_source_ref

    def resume(self, resume_ckpt):
        print("load model:", resume_ckpt)
        ckpt = torch.load(resume_ckpt)
        ckpt_name = os.path.basename(resume_ckpt)
        start_iter = int(os.path.splitext(ckpt_name)[0])

        self.gen.module.load_state_dict(ckpt["gen"])
        self.dis.module.load_state_dict(ckpt["dis"])
        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter
    
    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "dis": self.dis.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
