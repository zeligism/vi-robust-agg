
import torch
import torch.nn as nn
from torchvision import datasets
from ..utils import log_dict


def random_psd_matrix(N, dim, mu=0., ell=None):
    # https://github.com/hugobb/sgda/blob/main/gamesopt/games/quadratic_games.py
    M = torch.randn(N, dim, dim)
    PSD_M = torch.einsum("bik,bjk->bij", M, M)  # batched version of `M @ M.T`
    eigs, V = torch.linalg.eig(PSD_M)
    eigs.real = abs(eigs.real)
    if ell is not None:
        R_0 = ((1 / eigs).real).min(-1, keepdim=True)[0]
        eigs = eigs * R_0 * ell
    matrix = (V @ torch.diag_embed(eigs) @ torch.linalg.inv(V)).real
    return matrix


def random_vector(N, dim):
    return torch.randn(N, dim)


def generate_quadratic_game_dataset(N, dim):
    A11 = random_psd_matrix(N, dim)
    A12 = random_psd_matrix(N, dim)
    A22 = random_psd_matrix(N, dim)
    a1 = random_vector(N, dim)
    a2 = random_vector(N, dim)
    bias = random_vector(N, 1)
    return [A11, A12, A22, a1, a2, bias]


class QuadraticGameDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return [d[index] for d in self.data]


class TwoPlayers(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.player1 = nn.Parameter(torch.zeros(dim))
        self.player2 = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.player1, std=1. / dim)
        nn.init.normal_(self.player2, std=1. / dim)


def quadratic_game(
    data,
    batch_size,
    shuffle=None,
    sampler_callback=None,
    drop_last=True,
    **loader_kwargs
):
    # force set sampler to be None
    # sampler_callback = None

    dataset = QuadraticGameDataset(data)

    sampler = sampler_callback(dataset) if sampler_callback else None
    log_dict(
        {
            "Type": "Setup",
            "Dataset": "quadratic_game",
            "batch_size": batch_size,
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
