import torch
from .base import _BaseAggregator


class QuadraticOptimal(_BaseAggregator):
    def __init__(self, betas=(0.0, 0.0), lmbd=0.5, w_momentum=0.0, normalize_weights=False):
        super().__init__()
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.lmbd = lmbd  # tikhonov reg for pseudo-inverse
        self.w_momentum = w_momentum
        self.normalize_weights = normalize_weights
        self.M = None
        self.V = None
        self.weights = None

    def __call__(self, inputs):
        G = torch.stack(inputs, dim=0)  # Mxd
        GG = torch.einsum("id,jd->ij", G, G)
        eye = torch.eye(*GG.size(), out=torch.empty_like(GG))
        GG += self.lmbd * eye
        # `valid grad = average grad` this works when the majority are good workers
        gv = torch.sum(G, dim=0)

        ### Does not work with check computations as some workers will be banned ###
        # if self.M is None:
        #     self.M = G
        # if self.V is None:
        #     self.V = GG
        # self.M = self.beta1 * self.M + (1 - self.beta1) * G
        # self.V = self.beta2 * self.V + (1 - self.beta2) * GG
        self.M = G
        self.V = GG

        weights = torch.einsum("ij,jd,d->i", torch.linalg.inv(self.V), self.M, gv)
        weights = torch.relu(weights)
        if self.normalize_weights:
            weights /= weights.sum()

        ### Does not work with check computations as some workers will be banned ###
        # if self.weights is None:
        #     self.weights = weights
        # self.weights = self.w_momentum * self.weights + (1 - self.w_momentum) * weights
        self.weights = weights

        return torch.sum(self.weights.view(-1,1) * self.M, dim=0)

    def __str__(self):
        return "QuadraticOptimal"
