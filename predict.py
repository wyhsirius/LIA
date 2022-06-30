import tempfile
import torch
from tqdm import tqdm
from PIL import Image
import torchvision
import numpy as np
from cog import BasePredictor, Path, Input

from networks.generator import Generator


class Predictor(BasePredictor):
    def setup(self):

        self.size = 256
        channel_multiplier = 1
        latent_dim_style = 512
        latent_dim_motion = 20

        model_weights = {
            "vox": torch.load(
                "checkpoints/vox.pt", map_location=lambda storage, loc: storage
            )["gen"],
            "taichi": torch.load(
                "checkpoints/taichi.pt", map_location=lambda storage, loc: storage
            )["gen"],
            "ted": torch.load(
                "checkpoints/ted.pt", map_location=lambda storage, loc: storage
            )["gen"],
        }
        self.gen_models = {
            k: Generator(
                self.size, latent_dim_style, latent_dim_motion, channel_multiplier
            ).cuda()
            for k in model_weights.keys()
        }

        for k, v in self.gen_models.items():
            v.load_state_dict(model_weights[k])
            v.eval()

    def predict(
        self,
        img_source: Path = Input(
            description="Input source image.",
        ),
        driving_video: Path = Input(
            description="Choose a driving video.",
        ),
        model: str = Input(
            choices=["vox", "taichi", "ted"],
            default="vox",
            description="Choose a dataset.",
        ),
    ) -> Path:
        gen = self.gen_models[model]
        print("==> loading data")

        img_source = img_preprocessing(str(img_source), self.size).cuda()
        vid_target, fps = vid_preprocessing(str(driving_video))
        vid_target = vid_target.cuda()

        out_path = Path(tempfile.mkdtemp()) / "output.mp4"

        with torch.no_grad():
            vid_target_recon = []

            if model == "ted":
                h_start = None
            else:
                h_start = gen.enc.enc_motion(vid_target[:, 0, :, :, :])

            for i in tqdm(range(vid_target.size(1))):
                img_target = vid_target[:, i, :, :, :]
                img_recon = gen(img_source, img_target, h_start)
                vid_target_recon.append(img_recon.unsqueeze(2))

            vid_target_recon = torch.cat(vid_target_recon, dim=2)
            save_video(vid_target_recon, str(out_path), fps)

        return out_path


def load_image(filename, size):
    img = Image.open(filename).convert("RGB")
    img = img.resize((size, size))
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # 3 x 256 x 256

    return img / 255.0


def img_preprocessing(img_path, size):
    img = load_image(img_path, size)  # [0, 1]
    img = torch.from_numpy(img).unsqueeze(0).float()  # [0, 1]
    imgs_norm = (img - 0.5) * 2.0  # [-1, 1]

    return imgs_norm


def vid_preprocessing(vid_path):
    vid_dict = torchvision.io.read_video(vid_path, pts_unit="sec")
    vid = vid_dict[0].permute(0, 3, 1, 2).unsqueeze(0)
    fps = vid_dict[2]["video_fps"]
    vid_norm = (vid / 255.0 - 0.5) * 2.0  # [-1, 1]

    return vid_norm, fps


def save_video(vid_target_recon, save_path, fps):
    vid = vid_target_recon.permute(0, 2, 3, 4, 1)
    vid = vid.clamp(-1, 1).cpu()
    vid = ((vid - vid.min()) / (vid.max() - vid.min()) * 255).type("torch.ByteTensor")

    torchvision.io.write_video(save_path, vid[0], fps=fps)
