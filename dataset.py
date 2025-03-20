import pandas as pd
import random
from PIL import Image
from torch.utils.data import Dataset
import glob
import os
import cv2
from augmentations import AugmentationTransform
from PIL import ImageFile



ImageFile.LOAD_TRUNCATED_IMAGES = True


class Vox256(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/vox/train'
        elif split == 'test':
            self.ds_path = './datasets/vox/test'
        else:
            raise NotImplementedError

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

            return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Vox256_vox2german(Dataset):
    def __init__(self, transform=None):
        self.source_root = './datasets/german/'
        self.driving_root = './datasets/vox/test/'

        self.anno = pd.read_csv('pairs_annotations/german_vox.csv')

        self.source_imgs = os.listdir(self.source_root)
        self.transform = transform

    def __getitem__(self, idx):
        source_name = str('%03d' % self.anno['source'][idx])
        driving_name = self.anno['driving'][idx]

        source_vid_path = self.source_root + source_name
        driving_vid_path = self.driving_root + driving_name

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.source_imgs)


class Vox256_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class Vox256_cross(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/vox/test/'
        self.videos = os.listdir(self.ds_path)
        self.anno = pd.read_csv('pairs_annotations/vox256.csv')
        self.transform = transform

    def __getitem__(self, idx):
        source_name = self.anno['source'][idx]
        driving_name = self.anno['driving'][idx]

        source_vid_path = os.path.join(self.ds_path, source_name)
        driving_vid_path = os.path.join(self.ds_path, driving_name)

        source_frame_path = sorted(glob.glob(source_vid_path + '/*.png'))[0]
        driving_frames_path = sorted(glob.glob(driving_vid_path + '/*.png'))[:100]

        source_img = self.transform(Image.open(source_frame_path).convert('RGB'))
        driving_vid = [self.transform(Image.open(p).convert('RGB')) for p in driving_frames_path]

        return source_img, driving_vid, source_name, driving_name

    def __len__(self):
        return len(self.videos)


class Taichi(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/taichi/train/'
        else:
            self.ds_path = './datasets/taichi/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(True, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):

        video_path = self.ds_path + self.videos[idx]
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)


class Taichi_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/taichi/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)


class TED(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/ted/train/videos'
            self.face_mask_path = './datasets/ted/train/face_mask'
            self.hands_mask_path = './datasets/ted/train/hands_mask'
            self.lips_mask_path = './datasets/ted/train/lips_mask'
            self.eye_mask_path = './datasets/ted/train/eyes_mask'
            # TODO 手部的结构性损失
        else:
            self.ds_path = './datasets/ted/test/'
            self.face_mask_path = './datasets/ted/test/face_mask'
            self.hands_mask_path = './datasets/ted/test/hands_mask'
            self.lips_mask_path = './datasets/ted/test/lips_mask'
            self.eye_mask_path = './datasets/ted/test/eye_mask'
        
        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, video_name)
        face_mask_video_path = os.path.join(self.face_mask_path, video_name)
        hands_mask_video_path = os.path.join(self.hands_mask_path, video_name)
        lips_mask_video_path = os.path.join(self.lips_mask_path, video_name)
        eye_mask_video_path = os.path.join(self.eye_mask_path, video_name)

        # TODO 打开四段视频
        cap_rgb = cv2.VideoCapture(video_path)
        cap_face = cv2.VideoCapture(face_mask_video_path)
        cap_hands = cv2.VideoCapture(hands_mask_video_path)
        cap_lips = cv2.VideoCapture(lips_mask_video_path)
        cap_eye = cv2.VideoCapture(eye_mask_video_path)

        nframes = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        items = random.sample(range(nframes), 2)

        def get_frame(cap, frame_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx} from video.")
            # OpenCV 默认读取为 BGR，这里转换为 RGB 再给 PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame) 
        
        # 分别读取 source 帧、target 帧
        img_source = get_frame(cap_rgb, items[0])
        img_target = get_frame(cap_rgb, items[1])

        # 对 target 帧对应的三种 mask 也分别读取
        face_mask = get_frame(cap_face, items[1])
        hands_mask = get_frame(cap_hands, items[1])
        lips_mask = get_frame(cap_lips, items[1])
        eye_mask = get_frame(cap_eye, items[1])
            

        if self.augmentation:
            img_source, img_target, face_mask, hands_mask, lips_mask, eye_mask \
            = self.aug(img_source, img_target,
                       face_mask, hands_mask, lips_mask, eye_mask)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)
            face_mask = self.transform(face_mask)
            hands_mask = self.transform(hands_mask)
            lips_mask = self.transform(lips_mask)
            eye_mask = self.transform(eye_mask)

        return img_source, img_target, face_mask, hands_mask, lips_mask, eye_mask

    def __len__(self):
        return len(self.videos)

class TED_TEST(Dataset):
    def __init__(self, split, transform=None, augmentation=False):

        if split == 'train':
            self.ds_path = './datasets/ted/train/'
        else:
            self.ds_path = './datasets/ted/test/'

        self.videos = os.listdir(self.ds_path)
        self.augmentation = augmentation

        if self.augmentation:
            self.aug = AugmentationTransform(False, True, True)
        else:
            self.aug = None

        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.ds_path, self.videos[idx])
        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        nframes = len(frames_paths)

        items = random.sample(list(range(nframes)), 2)

        img_source = Image.open(frames_paths[items[0]]).convert('RGB')
        img_target = Image.open(frames_paths[items[1]]).convert('RGB')

        if self.augmentation:
            img_source, img_target = self.aug(img_source, img_target)

        if self.transform is not None:
            img_source = self.transform(img_source)
            img_target = self.transform(img_target)

        return img_source, img_target

    def __len__(self):
        return len(self.videos)

class TED_eval(Dataset):
    def __init__(self, transform=None):
        self.ds_path = './datasets/ted/test/'
        self.videos = os.listdir(self.ds_path)
        self.transform = transform

    def __getitem__(self, idx):
        vid_name = self.videos[idx]
        video_path = os.path.join(self.ds_path, vid_name)

        frames_paths = sorted(glob.glob(video_path + '/*.png'))
        vid_target = [self.transform(Image.open(p).convert('RGB')) for p in frames_paths]

        return vid_name, vid_target

    def __len__(self):
        return len(self.videos)
