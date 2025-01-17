import torch
from .base import _BaseAggregator


class QuadraticOptimal(_BaseAggregator):
    def __init__(self, lmbd=1., betas=(0.5, 0.9), wmomentum=0.5, target_grad_closure=None):
        super().__init__()
        self.lmbd = lmbd  # tikhonov reg for pseudo-inverse
        self.betas = betas
        self.wmomentum = wmomentum
        self.target_grad_closure = target_grad_closure
        self.G = None
        self.GG = None
        self.weights = None

    def __call__(self, inputs):
        # G = torch.stack(inputs, dim=0)  # Mxd
        # GG = torch.einsum("id,jd->ij", G, G)  # MxM

        # # Momentum for grads and grads-covariance across workers
        # if self.G is None:
        #     self.G = G
        # if self.GG is None:
        #     self.GG = GG
        # self.G = self.betas[0] * self.G + (1 - self.betas[0]) * G
        # self.GG = self.betas[1] * self.GG + (1 - self.betas[1]) * GG
        # G = self.G
        # GG = self.GG

        # # Create a target "validation" grad
        # if self.target_grad_closure is None:
        #     target_grad = torch.mean(G, dim=0)  # reduces to fedavg
        # else:
        #     target_grad = self.target_grad_closure()

        # # Inverse with regularization, decrease lmbd linearly
        # eye = torch.eye(*GG.size(), out=torch.empty_like(GG))
        # GG_inv = torch.linalg.inv(GG + self.lmbd * eye)
        # self.lmbd /= self.lmbd + 1

        # # weights = (G.T G + lmbd*eye)^-1 G.T target_grad, where here G is actually dxM
        # weights = torch.einsum("ij,jd,d->i", GG_inv, G, target_grad)
        # # weights = torch.relu(weights)
        # if self.weights is None:
        #     self.weights = weights
        # self.weights = self.wmomentum * self.weights + (1 - self.wmomentum) * weights
        # weights = self.weights

        # aggregated_grad = torch.sum(weights.view(-1,1) * G, dim=0)

        self.betas = (0,0)
        Gval = self.target_grad_closure()
        G = torch.stack(inputs + [Gval], dim=1)  # dxM, M << d
        GG = torch.einsum("di,dj->ij", G, G)  # MxM
        # update estimates with exponential averaging
        if self.G is None:
            self.G = G
        if self.GG is None:
            self.GG = GG
        self.G = self.betas[0] * self.G + (1 - self.betas[0]) * G
        self.GG = self.betas[1] * self.GG + (1 - self.betas[1]) * GG

        eye = torch.eye(*self.GG.size(), out=torch.empty_like(self.GG))
        GG_inv = torch.linalg.inv(self.GG + self.lmbd * eye)

        # If G = USV.T, where U: dxd, S: dxM (M singluar values), and V: MxM, then
        # orthogonal proj = G (G.T G)^-1 G.T = USV.T (VS^2V.T)^-1 VSU.T = US S^-2 SU.T = UU.T
        # weights = torch.einsum("mn,ng,g->m", GG_inv, self.G.T, torch.mean(self.G, dim=1))
        # aggregated_grad = torch.einsum("dm,m->d", self.G, weights)

        # ???
        aggregated_grad = Gval.pow(2).sum()**-1 * (Gval * torch.mean(self.G, dim=1)).sum() * Gval

        return aggregated_grad

    def __str__(self):
        return "QuadraticOptimal"


class MirrorDescent(_BaseAggregator):
    def __init__(self, model, target_grad_closure=None, num_iters=5):
        super().__init__()
        self.model = model
        self.target_grad_closure = target_grad_closure
        self.num_iters = num_iters
        self.lr = 0.1
        self.params_cache = {}
        self.weights = None

    @torch.no_grad()
    def __call__(self, inputs):
        # only allow grads for calculating target grad
        self.target_grad_closure = torch.enable_grad()(self.target_grad_closure)

        G = torch.stack(inputs, dim=0)  # Mxd

        if self.weights is None:
            weights = torch.ones(len(inputs))
            weights /= weights.sum()
            self.weights = weights

        for p in self.model.parameters():
            self.params_cache[p] = p.clone().detach()

        for iters in range(self.num_iters):
            aggregated = torch.sum(self.weights.view(-1,1) * G, dim=0)
            # Update params with aggregated grad
            i = 0
            for p in self.model.parameters():
                j = i + p.numel()
                p.sub_(aggregated[i:j].reshape_as(p), alpha=self.lr)
                i = j
            # calculate target grad
            target_grad = self.target_grad_closure(self.model)
            # revert to old params
            for p in self.model.parameters():
                p.copy_(self.params_cache[p])
            # apply weights update
            weights_prop = self.weights * torch.exp(-self.lr * torch.einsum("ij,j->i", G, target_grad))
            self.weights = torch.softmax(weights_prop, dim=0)
            # print(iters, weights_prop)

        return torch.sum(self.weights.view(-1,1) * G, dim=0)

    def __str__(self):
        return "MirrorDescent"
