import torch
from .base import _BaseAggregator


class QuadraticOptimal(_BaseAggregator):
    def __init__(self, lmbd=1., betas=(0.5, 0.9), wmomentum=0.5, target_grad_closure=None):
        super().__init__()
        self.lmbd = lmbd  # tikhonov reg for pseudo-inverse
        self.betas = betas
        self.wmomentum = wmomentum
        self.target_grad_closure = None#target_grad_closure
        self.G = None
        self.GG = None
        self.weights = None

    def __call__(self, inputs):
        G = torch.stack(inputs, dim=0)  # Mxd
        GG = torch.einsum("id,jd->ij", G, G)  # MxM

        # Momentum for grads and grads-covariance across workers
        if self.G is None:
            self.G = G
        if self.GG is None:
            self.GG = GG
        self.G = self.betas[0] * self.G + (1 - self.betas[0]) * G
        self.GG = self.betas[1] * self.GG + (1 - self.betas[1]) * GG
        G = self.G
        GG = self.GG

        # Create a target "validation" grad
        if self.target_grad_closure is None:
            target_grad = torch.mean(G, dim=0)  # reduces to fedavg
        else:
            target_grad = self.target_grad_closure()

        # Inverse with regularization, decrease lmbd linearlt
        eye = torch.eye(*GG.size(), out=torch.empty_like(GG))
        GG_inv = torch.linalg.inv(GG + self.lmbd * eye)
        self.lmbd = 1 / (1 + 1 / self.lmbd)

        # weights = (G.T G + lmbd*eye)^-1 G.T target_grad
        weights = torch.einsum("ij,jd,d->i", GG_inv, G, target_grad)
        # weights = torch.relu(weights)
        if self.weights is None:
            self.weights = weights
        self.weights = self.wmomentum * self.weights + (1 - self.wmomentum) * weights
        weights = self.weights

        return torch.sum(weights.view(-1,1) * G, dim=0)

    def __str__(self):
        return "QuadraticOptimal"
