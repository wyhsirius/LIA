from torch import nn
import torch
from torchvision import models
import numpy as np
from networks.utils import AntiAliasInterpolation2d


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss. See Sec 3.3
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):

        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)

        return out_dict


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss. See Sec 3.3.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()

        vgg_model = models.vgg19(pretrained=True)
        # vgg_model.load_state_dict(torch.load('./vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = vgg_model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):

        X = X.clamp(-1, 1)
        X = X / 2 + 0.5
        X = (X - self.mean) / self.std
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()

        self.scales = [1, 0.5, 0.25, 0.125]
        self.pyramid = ImagePyramide(self.scales, 3).cuda()

        # vgg loss
        self.vgg = Vgg19().cuda()
        self.weights = (10, 10, 10, 10, 10)

    def forward(self, img_recon, img_real):

        # vgg loss
        pyramid_real = self.pyramid(img_real)
        pyramid_recon = self.pyramid(img_recon)

        vgg_loss = 0
        for scale in self.scales:
            recon_vgg = self.vgg(pyramid_recon['prediction_' + str(scale)])
            real_vgg = self.vgg(pyramid_real['prediction_' + str(scale)])

            for i, weight in enumerate(self.weights):
                value = torch.abs(recon_vgg[i] - real_vgg[i].detach()).mean()
                vgg_loss += value * self.weights[i]

        return vgg_loss
