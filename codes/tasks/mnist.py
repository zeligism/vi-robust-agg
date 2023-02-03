import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from ..utils import log_dict


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class _NetWithAdversary(nn.Module):
    """
    Adversary on the activation prior to last linear layer.
    """

    def __init__(self, adv_strength):
        super().__init__()
        self.pre_fc = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(9216, 128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )
        self.fc = nn.Linear(128, 10)
        self.model = nn.Sequential(self.pre_fc, self.fc)
        self.adversary = DataShiftAdversary((128,), adv_strength)

    def forward(self, x):
        x = self.pre_fc(x)
        x = self.adversary(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class NetWithAdversary(nn.Module):
    """
    Adversary on the input data.
    """

    def __init__(self, adv_strength):
        super().__init__()
        self.model = Net()
        self.adversary = DataShiftAdversary((1, 28, 28), adv_strength)

    def forward(self, x):
        return self.model(self.adversary(x))


class DataShiftAdversary(nn.Module):
    def __init__(self, shape, strength):
        super().__init__()
        self.parameter = nn.Parameter(torch.randn(1, *shape))
        self.strength = strength

    def forward(self, x):
        return x + self.strength * self.parameter


def mnist(
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
    # if sampler_callback is not None and shuffle is not None:
    #     raise ValueError

    dataset = dataset_cls(
        data_dir,
        train=train,
        download=download,
        transform=transforms.Compose(
            [
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
