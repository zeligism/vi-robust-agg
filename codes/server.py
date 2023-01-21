import torch


class TorchServer(object):
    def __init__(self, optimizer: torch.optim.Optimizer, model):
        self.optimizer = optimizer
        self.model = model

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.empty_like(p)
                # for p in self.model.parameters():
                end = beg + len(p.view(-1))
                g = gradient[beg:end].reshape_as(p)
                p.grad.copy_(g.clone().detach())
                beg = end
