
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from torchvision import datasets, transforms
from ..utils import log_dict


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, cond_dim=0, num_classes=0, embed_class=True):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.num_groups = 2
        self.Embedding = nn.Embedding if embed_class else nn.Linear
        if cond_dim == 0 and num_classes == 0:
            self.bn = nn.BatchNorm2d(num_features, affine=True)
        if cond_dim > 0:
            self.bn = nn.BatchNorm2d(num_features, affine=False)
            self.cond_map = nn.Linear(cond_dim, 2 * num_features)
            self.cond_map.weight.data[:, :num_features].normal_(1, 0.02)
            self.cond_map.weight.data[:, num_features:].zero_()
        if num_classes > 0:
            self.bn = nn.BatchNorm2d(num_features, affine=False)
            self.label_map = self.Embedding(num_classes, 2 * num_features)
            self.label_map.weight.data[:, :num_features].normal_(1, 0.02)
            self.label_map.weight.data[:, num_features:].zero_()
        if cond_dim > 0 and num_classes > 0:
            cond_features = num_features // 2
            self.bn1 = nn.BatchNorm2d(cond_features, affine=False)
            self.cond_map = nn.Linear(cond_dim, 2 * cond_features)
            self.cond_map.weight.data[:, :cond_features].normal_(1, 0.02)
            self.cond_map.weight.data[:, cond_features:].zero_()
            label_features = num_features - cond_features
            self.bn2 = nn.BatchNorm2d(label_features, affine=False)
            self.label_map = self.Embedding(num_classes, 2 * label_features)
            self.label_map.weight.data[:, :label_features].normal_(1, 0.02)
            self.label_map.weight.data[:, label_features:].zero_()
            self.cond_features = cond_features
            self.label_features = label_features

    def forward(self, x, cond=None, label=None):
        # assert cond is None or self.cond_dim > 0
        # assert label is None or self.num_classes > 0
        if self.cond_dim == 0 and self.num_classes == 0:
            out = self.bn(x)
        if self.cond_dim > 0 and self.num_classes == 0:
            out = self.bn(x)
            if cond is not None:
                gamma, beta = self.cond_map(cond).chunk(2, dim=1)
                gamma = gamma.view(-1, self.num_features, 1, 1)
                beta = beta.view(-1, self.num_features, 1, 1)
                out = gamma * out + beta
        elif self.cond_dim == 0 and self.num_classes > 0:
            out = self.bn(x)
            if label is not None:
                gamma, beta = self.label_map(label).chunk(2, dim=1)
                gamma = gamma.view(-1, self.num_features, 1, 1)
                beta = beta.view(-1, self.num_features, 1, 1)
                out = gamma * out + beta
        elif self.cond_dim > 0 and self.num_classes > 0:
            x1, x2 = x.chunk(2, dim=1)
            out1 = self.bn1(x1)
            out2 = self.bn2(x2)
            if cond is not None:
                gamma1, beta1 = self.cond_map(cond).chunk(2, dim=1)
                gamma1 = gamma1.view(-1, self.cond_features, 1, 1)
                beta1 = beta1.view(-1, self.cond_features, 1, 1)
                out1 = gamma1 * out1 + beta1
            if label is not None:
                gamma2, beta2 = self.label_map(label).chunk(2, dim=1)
                gamma2 = gamma2.view(-1, self.label_features, 1, 1)
                beta2 = beta2.view(-1, self.label_features, 1, 1)
                out2 = gamma2 * out2 + beta2
            out = torch.cat([out1, out2], dim=1)

        return out


class ChannelNoise(nn.Module):
    """
    Channel noise injection module.
    Adds a linearly transformed noise to a convolution layer.
    """

    def __init__(self, num_channels, std=0.02):
        super().__init__()
        self.std = std
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x):
        noise_size = [x.size()[0], 1, *x.size()[2:]]  # single channel
        noise = self.std * torch.randn(noise_size).to(x)

        return x + self.scale * noise


# resnet code based on:
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/model_resnet.py
class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, cond_dim=0, num_classes=0, embed_class=True):
        super().__init__()
        cond_opts = {'cond_dim': cond_dim, 'num_classes': num_classes, 'embed_class': embed_class}

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.cn1 = ChannelNoise(in_channels)
        self.bn1 = ConditionalBatchNorm2d(in_channels, **cond_opts)
        self.upsample = nn.Upsample(scale_factor=2)
        self.cn2 = ChannelNoise(out_channels)
        self.bn2 = ConditionalBatchNorm2d(out_channels, **cond_opts)

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x, cond=None, label=None):
        cond_args = {'cond': cond, 'label': label}
        h = F.relu(self.bn1(self.cn1(x), **cond_args))
        h = self.conv1(self.upsample(h))
        h = F.relu(self.bn2(self.cn2(h), **cond_args))
        h = self.conv2(h)
        return self.bypass(x) + h


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_sn=True):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        if use_sn:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(inplace=True),
                self.conv1,
                nn.ReLU(inplace=True),
                self.conv2,
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(inplace=True),
                self.conv1,
                nn.ReLU(inplace=True),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            if use_sn:
                self.bypass_conv = spectral_norm(self.bypass_conv)

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         spectral_norm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_sn=True):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
        if use_sn:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.bypass_conv = spectral_norm(self.bypass_conv)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            self.conv2,
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


############################################
class ResNetGenerator(nn.Module):
    def __init__(self, z_dim, num_features=128, image_size=32, channels=3, cond_dim=0, num_classes=0, embed_class=True):
        super().__init__()
        self.z_dim = z_dim
        self.num_features = num_features
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        cond_opts = {'cond_dim': cond_dim, 'num_classes': num_classes, 'embed_class': embed_class}

        self.dense = nn.Linear(self.z_dim, 4 * 4 * num_features)
        self.final = nn.Conv2d(num_features, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.block1 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)
        self.block2 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)
        self.block3 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)
        self.block4 = None
        if image_size == 64:
            self.block4 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)

        self.cn = ChannelNoise(num_features)
        self.bn = ConditionalBatchNorm2d(num_features, **cond_opts)

    def forward(self, z, cond=None, label=None):
        cond_args = {'cond': cond, 'label': label}
        h = self.dense(z).view(-1, self.num_features, 4, 4)
        h = self.block1(h, **cond_args)
        h = self.block2(h, **cond_args)
        h = self.block3(h, **cond_args)
        if self.block4 is not None:
            h = self.block4(h, **cond_args)
        h = F.relu(self.bn(self.cn(h), **cond_args))
        out = torch.tanh(self.final(h))
        return out


class ResNetDiscriminator(nn.Module):
    def __init__(self, num_features=128, image_size=32, channels=3, cond_dim=0, num_classes=0, use_sn=True, embed_class=True):
        super().__init__()
        self.num_features = num_features
        self.image_size = image_size
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.use_sn = use_sn
        self.Embedding = nn.Embedding if embed_class else nn.Linear

        def maybe_sn(layer):
            return spectral_norm(layer) if use_sn else layer

        self.hidden_dim = num_features
        if image_size == 64:
            self.hidden_dim = num_features * 4

        self.model = [
            FirstResBlockDiscriminator(channels, num_features, stride=2, use_sn=use_sn),
            ResBlockDiscriminator(num_features, num_features, stride=2, use_sn=use_sn),
            ResBlockDiscriminator(num_features, num_features, use_sn=use_sn),
            ResBlockDiscriminator(num_features, num_features, use_sn=use_sn),
        ]
        if image_size == 64:
            self.model += [ResBlockDiscriminator(num_features, num_features, use_sn=use_sn)]

        self.model = nn.Sequential(*self.model, nn.ReLU(inplace=True), nn.AvgPool2d(8))

        self.fc = maybe_sn(nn.Linear(self.hidden_dim, 1))
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)

        if cond_dim > 0:
            self.cond_map = maybe_sn(nn.Linear(cond_dim, self.hidden_dim))
            nn.init.xavier_uniform_(self.cond_map.weight.data, 1.)

        if num_classes > 0:
            self.label_map = maybe_sn(self.Embedding(num_classes, self.hidden_dim))
            nn.init.xavier_uniform_(self.label_map.weight.data, 1.)

    def forward(self, x, cond=None, label=None, return_h=False):
        # assert cond is None or self.cond_dim > 0
        # assert label is None or self.num_classes > 0
        h = self.model(x)
        h = h.view(-1, self.hidden_dim)
        output = self.fc(h)
        if cond is not None:
            output += torch.sum(self.cond_map(cond) * h, dim=1, keepdim=True)
        if label is not None:
            output += torch.sum(self.label_map(label) * h, dim=1, keepdim=True)

        return (output, h) if return_h else output


#################################
# Only supports 32x32 or 64x64 images
class ConditionalResNetGAN(nn.Module):
    def __init__(self, num_latents=64,
                 D_features=32, G_features=32,
                 image_size=32, channels=3,
                 cond_dim=0, num_classes=10,
                 use_sn=True):
        super().__init__()
        self.num_latents = num_latents
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.D = ResNetDiscriminator(num_features=D_features,
                                     image_size=image_size,
                                     channels=channels,
                                     cond_dim=cond_dim,
                                     num_classes=num_classes,
                                     use_sn=use_sn)
        self.G = ResNetGenerator(num_latents,
                                 num_features=G_features,
                                 image_size=image_size,
                                 channels=channels,
                                 cond_dim=cond_dim,
                                 num_classes=num_classes)


class ResNetGAN(ConditionalResNetGAN):
    def __init__(self, *args, **kwargs):
        kwargs['cond_dim'] = 0
        kwargs['num_classes'] = 0
        super().__init__(*args, **kwargs)


def mnist32(
    data_dir,
    train,
    download,
    batch_size,
    shuffle=None,
    sampler_callback=None,
    dataset_cls=datasets.MNIST,
    drop_last=True,
    **loader_kwargs
):
    # force set sampler to be None
    # sampler_callback = None

    dataset = dataset_cls(
        data_dir,
        train=train,
        download=download,
        transform=transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    sampler = sampler_callback(dataset) if sampler_callback else None
    log_dict(
        {
            "Type": "Setup",
            "Dataset": "mnist",
            "data_dir": data_dir,
            "train": train,
            "download": download,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler.__str__() if sampler else None,
        }
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        drop_last=drop_last,
        **loader_kwargs,
    )
