
import torch
import torch.nn as nn
from torchvision import datasets
from ..utils import log_dict


def random_matrix(num_samples, dim, mu=0., ell=None):
    # https://github.com/hugobb/sgda/blob/main/gamesopt/games/quadratic_games.py
    matrix = torch.randn(num_samples, dim, dim)
    eigs: torch.Tensor
    eigs, V = torch.linalg.eig(matrix)
    eigs.real = abs(eigs.real)
    if ell is not None:
        R_0 = ((1 / eigs).real).min(-1, keepdim=True)[0]
        eigs = eigs * R_0 * ell
    matrix = (V @ torch.diag_embed(eigs) @ torch.linalg.inv(V)).real
    return matrix


def random_vector(N, dim):
    return torch.randn(N, dim)


def generate_quadratic_game_dataset(N, dim, mu=0., ell=None, num_players=2):
    if num_players > 2:
        A = random_matrix(N, dim * num_players, mu=mu, ell=ell)
        b = random_vector(N, dim * num_players) * 100 / dim**0.5
        return [A, b]
    else:
        A11 = random_matrix(N, dim, mu=mu, ell=ell)
        A12 = random_matrix(N, dim)
        A22 = random_matrix(N, dim, mu=mu, ell=ell)
        b1 = random_vector(N, dim) * 100 / dim**0.5
        b2 = random_vector(N, dim) * 100 / dim**0.5
        return [A11, A12, A22, b1, b2]


def quadratic_loss_2player(w1, w2, A11, A12, A22, b1, b2):
    loss = torch.einsum("bij,i,j->b", A12, w1, w2)
    loss += 0.5 * torch.einsum("bij,i,j->b", A11, w1, w1)
    loss -= 0.5 * torch.einsum("bij,i,j->b", A22, w2, w2)
    loss += torch.einsum("bi,i->b", b1, w1)
    loss -= torch.einsum("bi,i->b", b2, w2)
    return loss.mean()


def quadratic_loss_generalized(players, k, A, b):
    dim = len(players[0])
    player1_indices = slice(k * dim, (k + 1) * dim)
    rhs = b[:, player1_indices]
    for j in range(len(players)):
        player2_indices = slice(j * dim, (j + 1) * dim)
        A_kj = A[:, player1_indices, player2_indices]
        rhs += torch.einsum("bij,j->bi", A_kj, players[j])
    loss = torch.einsum("bi,i->b", rhs, players[k])
    return loss.mean()


def get_quadratic_loss(num_players):
    if num_players > 2:
        return quadratic_loss_generalized
    else:
        return quadratic_loss_2player


class QuadraticGameDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        return [d[index] for d in self.data]


class MultiPlayer(nn.Module):
    def __init__(self, num_players=2, dim=2):
        super().__init__()
        self.players = nn.ParameterList([nn.Parameter(torch.zeros(dim)) for _ in range(num_players)])
        for player in self.players:
            nn.init.normal_(player, std=1. / dim)


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
