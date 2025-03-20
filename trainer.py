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
        self.dis_main = Discriminator(args.size, args.channel_multiplier).to(device)
        self.dis_face = Discriminator(args.size // (args.size // 64), args.channel_multiplier).to(device)
        self.dis_hands = Discriminator(args.size // (args.size // 64), args.channel_multiplier).to(device)
        self.dis_lips = Discriminator(args.size // (args.size // 64), args.channel_multiplier).to(device)
        self.dis_eyes = Discriminator(args.size // (args.size // 64), args.channel_multiplier).to(device)

        self.gen = Generator(args.size, args.latent_dim_style, args.latent_dim_motion, args.channel_multiplier).to(device)

        # distributed computing
        self.gen = DDP(self.gen, device_ids=[rank], find_unused_parameters=True)
        self.dis_main = DDP(self.dis_main, device_ids=[rank], find_unused_parameters=True)
        self.dis_face = DDP(self.dis_face, device_ids=[rank], find_unused_parameters=True)
        self.dis_hands = DDP(self.dis_hands, device_ids=[rank], find_unused_parameters=True)
        self.dis_lips = DDP(self.dis_lips, device_ids=[rank], find_unused_parameters=True)
        self.dis_eyes = DDP(self.dis_eyes, device_ids=[rank], find_unused_parameters=True)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        self.g_optim = optim.Adam(
            self.gen.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
        )

        # 合并所有判别器的参数
        all_dis_params = list(self.dis_main.parameters()) + \
                         list(self.dis_face.parameters()) + \
                         list(self.dis_hands.parameters()) + \
                         list(self.dis_lips.parameters()) + \
                         list(self.dis_eyes.parameters())

        self.d_optim = optim.Adam(
            all_dis_params,
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
        requires_grad(self.gen, True)
        requires_grad(self.dis_main, False)
        requires_grad(self.dis_face, False)
        requires_grad(self.dis_hands, False)
        requires_grad(self.dis_lips, False)
        requires_grad(self.dis_eyes, False)

        self.gen.zero_grad()

        # 生成器前向得到全图重建
        img_target_recon = self.gen(img_source, img_target)

        # 全图判别器
        pred_main = self.dis_main(img_target_recon)

        # 对 batch 内每个样本，根据各 mask 裁剪后统一 resize 到固定尺寸（如 64×64）
        face_cropped = batch_crop_and_resize(img_target_recon, face_mask, target_size=(64, 64))
        pred_face = self.dis_face(face_cropped)

        hands_cropped = batch_crop_and_resize(img_target_recon, hands_mask, target_size=(64, 64))
        pred_hands = self.dis_hands(hands_cropped)

        lips_cropped = batch_crop_and_resize(img_target_recon, lips_mask, target_size=(64, 64))
        pred_lips = self.dis_lips(lips_cropped)

        eyes_cropped = batch_crop_and_resize(img_target_recon, eyes, target_size=(64, 64))
        pred_eyes = self.dis_eyes(eyes_cropped)

        # 对抗损失
        gan_main_loss  = self.g_nonsaturating_loss(pred_main)
        gan_face_loss  = self.g_nonsaturating_loss(pred_face)
        gan_hands_loss = self.g_nonsaturating_loss(pred_hands)
        gan_lips_loss  = self.g_nonsaturating_loss(pred_lips)
        gan_eyes_loss  = self.g_nonsaturating_loss(pred_eyes)

        # 局部对抗损失权重：10:5:10:5 对应 face, hands, lips, eyes
        gan_local_loss = 2.0 * gan_face_loss + 2.5 * gan_hands_loss + 2.5 * gan_lips_loss + 2.5 * gan_eyes_loss
        gan_g_loss = gan_main_loss + gan_local_loss

        # VGG Loss (全图及各局部乘掩码方式)
        vgg_loss_base = self.criterion_vgg(img_target_recon, img_target).mean()
        face_vgg_loss  = self.criterion_vgg(img_target_recon * face_mask, img_target * face_mask).mean()
        hands_vgg_loss = self.criterion_vgg(img_target_recon * hands_mask, img_target * hands_mask).mean()
        lips_vgg_loss  = self.criterion_vgg(img_target_recon * lips_mask,  img_target * lips_mask).mean()
        eyes_vgg_loss  = self.criterion_vgg(img_target_recon * eyes,   img_target * eyes).mean()
        vgg_loss = 0.6 * vgg_loss_base + 0.6 * (face_vgg_loss + 2.0 * hands_vgg_loss + 4.0 * lips_vgg_loss + 2.0 * eyes_vgg_loss)

        # L1 Loss
        l1_loss = F.l1_loss(img_target_recon, img_target)

        # 总生成器损失
        g_loss = gan_g_loss + vgg_loss + l1_loss

        g_loss.backward()
        self.g_optim.step()

        return vgg_loss, l1_loss, gan_g_loss, img_target_recon

    def dis_update(self, img_real, img_recon, face_mask, hands_mask, lips_mask, eyes):
        requires_grad(self.gen, False)
        requires_grad(self.dis_main, True)
        requires_grad(self.dis_face, True)
        requires_grad(self.dis_hands, True)
        requires_grad(self.dis_lips, True)
        requires_grad(self.dis_eyes, True)

        # 清零所有判别器梯度
        self.dis_main.zero_grad()
        self.dis_face.zero_grad()
        self.dis_hands.zero_grad()
        self.dis_lips.zero_grad()
        self.dis_eyes.zero_grad()

        # 全图判别器损失
        real_pred_main = self.dis_main(img_real)
        fake_pred_main = self.dis_main(img_recon.detach())
        loss_main = self.d_nonsaturating_loss(fake_pred_main, real_pred_main)

        # 局部判别器：依次对每个区域裁剪，再输入对应的判别器
        real_face = batch_crop_and_resize(img_real, face_mask, target_size=(64, 64))
        fake_face = batch_crop_and_resize(img_recon.detach(), face_mask, target_size=(64, 64))
        loss_face = self.d_nonsaturating_loss(self.dis_face(fake_face), self.dis_face(real_face))

        real_hands = batch_crop_and_resize(img_real, hands_mask, target_size=(64, 64))
        fake_hands = batch_crop_and_resize(img_recon.detach(), hands_mask, target_size=(64, 64))
        loss_hands = self.d_nonsaturating_loss(self.dis_hands(fake_hands), self.dis_hands(real_hands))

        real_lips = batch_crop_and_resize(img_real, lips_mask, target_size=(64, 64))
        fake_lips = batch_crop_and_resize(img_recon.detach(), lips_mask, target_size=(64, 64))
        loss_lips = self.d_nonsaturating_loss(self.dis_lips(fake_lips), self.dis_lips(real_lips))

        real_eyes = batch_crop_and_resize(img_real, eyes, target_size=(64, 64))
        fake_eyes = batch_crop_and_resize(img_recon.detach(), eyes, target_size=(64, 64))
        loss_eyes = self.d_nonsaturating_loss(self.dis_eyes(fake_eyes), self.dis_eyes(real_eyes))

        # 合并判别器损失（可以设置不同权重，此处简单相加）
        d_loss = 0.1 * loss_main + 0.2 * (loss_face + 2 * loss_hands + 3 * loss_lips + 2 * loss_eyes)

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
        self.dis_main.module.load_state_dict(ckpt["dis_main"])
        self.dis_face.module.load_state_dict(ckpt["dis_face"])
        self.dis_hands.module.load_state_dict(ckpt["dis_hands"])
        self.dis_lips.module.load_state_dict(ckpt["dis_lips"])
        self.dis_eyes.module.load_state_dict(ckpt["dis_eyes"])

        self.g_optim.load_state_dict(ckpt["g_optim"])
        self.d_optim.load_state_dict(ckpt["d_optim"])

        return start_iter

    def save(self, idx, checkpoint_path):
        torch.save(
            {
                "gen": self.gen.module.state_dict(),
                "dis_main": self.dis_main.module.state_dict(),
                "dis_face": self.dis_face.module.state_dict(),
                "dis_hands": self.dis_hands.module.state_dict(),
                "dis_lips": self.dis_lips.module.state_dict(),
                "dis_eyes": self.dis_eyes.module.state_dict(),
                "g_optim": self.g_optim.state_dict(),
                "d_optim": self.d_optim.state_dict(),
                "args": self.args
            },
            f"{checkpoint_path}/{str(idx).zfill(6)}.pt"
        )
