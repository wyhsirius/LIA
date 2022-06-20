import torch
import torchvision
import numpy as np
import random
import PIL
from PIL import Image
from skimage import img_as_ubyte, img_as_float
import warnings
import numbers


class RandomCrop(object):
    """Extract random crop at the same location for a list of videos
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size

    def __call__(self, img_source, img_target):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of videos to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of videos
        """
        h, w = self.size
        if isinstance(img_source, np.ndarray):
            im_h, im_w, im_c = img_source.shape
        elif isinstance(img_source, PIL.Image.Image):
            im_w, im_h = img_source.size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got {0}'.format(type(img_source)))

        x1 = 0 if h == im_h else random.randint(0, im_w - w)
        y1 = 0 if w == im_w else random.randint(0, im_h - h)

        if isinstance(img_source, np.ndarray):
            img_source_crop = img_source[y1:y1 + h, x1:x1 + w, :]
            img_target_crop = img_target[y1:y1 + h, x1:x1 + w, :]

        elif isinstance(img_source, PIL.Image.Image):
            img_source_crop = img_source.crop((x1, y1, x1 + w, y1 + h))
            img_target_crop = img_target.crop((x1, y1, x1 + w, y1 + h))
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image but got {0}'.format(type(img_source)))

        return img_source_crop, img_target_crop


class RandomFlip(object):
    def __init__(self, horizontal_flip=True):
        self.horizontal_flip = horizontal_flip

    def __call__(self, img_source, img_target):
        if random.random() < 0.5 and self.horizontal_flip:
            # return np.fliplr(img_source), np.fliplr(img_target)
            return img_source.transpose(PIL.Image.FLIP_LEFT_RIGHT), img_target.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        return img_source, img_target


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, img_source, img_target):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """

        if isinstance(img_source, np.ndarray):
            brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation,
                                                                    self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)
            img_transforms = [img_as_ubyte, torchvision.transforms.ToPILImage()] + img_transforms + [np.array,
                                                                                                     img_as_float]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for func in img_transforms:
                    jittered_source = func(img_source).astype('float32')
                    jittered_target = func(img_target).astype('float32')

        elif isinstance(img_source, PIL.Image.Image):
            brightness, contrast, saturation, hue = self.get_params(self.brightness, self.contrast, self.saturation,
                                                                    self.hue)

            # Create img transform function sequence
            img_transforms = []
            if brightness is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
            if saturation is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
            if hue is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
            if contrast is not None:
                img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
            random.shuffle(img_transforms)

            # Apply to all videos
            for func in img_transforms:
                jittered_source = func(img_source)
                jittered_target = func(img_target)

        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(img_source)))
        return jittered_source, jittered_target


class AugmentationTransform:
    def __init__(self, crop, flip, jitter):

        self.transforms = []

        if crop:
            self.transforms.append(RandomCrop(224))
        if flip:
            self.transforms.append(RandomFlip(True))
        if jitter:
            self.transforms.append(ColorJitter(0.1, 0.1, 0.1, 0.1))

    def __call__(self, img_source, img_target):

        for t in self.transforms:
            img_source, img_target = t(img_source, img_target)

        return img_source, img_target
