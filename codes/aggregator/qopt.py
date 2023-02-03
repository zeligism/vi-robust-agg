import torch
from .base import _BaseAggregator


class QuadraticOptimal(_BaseAggregator):
    def __init__(self, lmbd=0.5):
        super().__init__()
        self.lmbd = lmbd  # tikhonov reg for pseudo-inverse

    def __call__(self, inputs, target_grad_closure=None):
        G = torch.stack(inputs, dim=0)  # Mxd
        GG = torch.einsum("id,jd->ij", G, G)  # MxM
        eye = torch.eye(*GG.size(), out=torch.empty_like(GG))
        GG_inv = torch.linalg.inv(GG + self.lmbd * eye)
        if target_grad is None:
            # reduces to fedavg
            target_grad = torch.mean(G, dim=0)

        weights = torch.einsum("ij,jd,d->i", GG_inv, G, target_grad)
        weights = torch.relu(weights)

        return torch.sum(weights.view(-1,1) * G, dim=0)

    def __str__(self):
        return "QuadraticOptimal"
