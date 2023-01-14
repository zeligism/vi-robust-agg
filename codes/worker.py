import torch
from collections import defaultdict
from typing import Optional, Union, Callable, Any, Tuple
from copy import deepcopy

from .gan_utils import make_grid


class TorchWorker(object):
    """A worker for distributed training.

    Compute gradients locally and store the gradient.
    """

    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
    ):
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.device = device

        # self.running has attribute:
        #   - `train_loader_iterator`: data iterator
        #   - `data`: last data
        #   - `target`: last target
        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)

    def add_metric(
        self,
        name: str,
        callback: Callable[[torch.Tensor, torch.Tensor], float],
    ):
        """
        The `callback` function takes predicted and groundtruth value
        and returns its metric.
        """
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    def __str__(self) -> str:
        return "TorchWorker"

    def train_epoch_start(self) -> None:
        self.running["train_loader_iterator"] = iter(self.data_loader)
        self.model.train()

    def compute_gradient(self) -> Tuple[float, int]:
        results = {}

        data, target = self.running["train_loader_iterator"].__next__()
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_func(output, target)
        loss.backward()
        self._save_grad()

        self.running["data"] = data
        self.running["target"] = target

        results["loss"] = loss.item()
        results["length"] = len(target)
        results["metrics"] = {}
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric(output, target)
        return results

    def get_gradient(self) -> torch.Tensor:
        return self._get_saved_grad()

    def apply_gradient(self) -> None:
        self.optimizer.step()

    def set_gradient(self, gradient: torch.Tensor) -> None:
        beg = 0
        for p in self.model.parameters():
            end = beg + len(p.grad.view(-1))
            x = gradient[beg:end].reshape_as(p.grad.data)
            p.grad.data = x.clone().detach()
            beg = end

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                param_state["saved_grad"] = torch.clone(p.grad).detach()

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["saved_grad"].data.view(-1))
        return torch.cat(layer_gradients)


class MomentumWorker(TorchWorker):
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                else:
                    # param_state["momentum_buffer"].mul_(self.momentum).add_(p.grad)
                    param_state["momentum_buffer"].mul_(self.momentum).add_((1 - self.momentum) * p.grad)  # XXX

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                layer_gradients.append(param_state["momentum_buffer"].data.view(-1))
        return torch.cat(layer_gradients)


class AdamWorker(TorchWorker):
    def __init__(self, betas=(0.5, 0.9), eps=1e-8, correct_bias=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.betas = betas
        self.eps = eps
        self.correct_bias = correct_bias

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if p.grad is None:
                    # param_state["momentum_buffer"] = torch.empty_like(p)
                    # param_state["momentumsq_buffer"] = torch.empty_like(p)
                    continue
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = torch.clone(p.grad).detach()
                    param_state["momentumsq_buffer"] = torch.clone(p.grad**2).detach()
                    # param_state["momentumsq_buffer"] = torch.clone(param_state["momentum_buffer"]**2).detach()
                    param_state['steps'] = 1
                else:
                    param_state["momentum_buffer"].mul_(self.betas[0]).add_((1 - self.betas[0]) * p.grad)
                    param_state["momentumsq_buffer"].mul_(self.betas[1]).add_((1 - self.betas[1]) * p.grad**2)
                    # param_state["momentumsq_buffer"].mul_(self.betas[1]).add_((1 - self.betas[1]) * param_state["momentum_buffer"]**2)
                    param_state['steps'] += 1

    def _get_saved_grad(self) -> torch.Tensor:
        layer_gradients = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                m = param_state["momentum_buffer"].data.view(-1)
                v = param_state["momentumsq_buffer"].data.view(-1)
                if self.correct_bias:
                    m = m / (1 - self.betas[0] ** param_state['steps'])
                    v = v / (1 - self.betas[1] ** param_state['steps'])
                layer_gradients.append(m / (v**0.5 + self.eps))
        return torch.cat(layer_gradients)


###############################################################################
class GANWorker(TorchWorker):
    """
    A worker for GAN training.
    """
    METRICS = ['D->D(x)', 'D->D(G(z))', 'G->D(G(z))']

    def __init__(self,
                 worker_id,
                 D_iters: int = 3,
                 conditional: bool = False,
                 *args, **kwargs):
        self.lr_mult = 1.0
        self.batch_size = None
        self.worker_id = worker_id
        super().__init__(*args, **kwargs)
        self.D_iters = D_iters
        self.conditional = conditional
        self.D_optimizer = self.optimizer["D"]
        self.G_optimizer = self.optimizer["G"]
        self.optimizer = self.optimizer["all"]  # dummy optimizer for passing grads
        self.init_fixed_sample()
        self.progress_frames = []
        self.progress_frames_freq = 16  # per epoch, better if = multiple of 2
        self.progress_frames_maxlen = 200

    def __str__(self) -> str:
        return f"GANWorker [{self.worker_id}]"

    @staticmethod
    def add_noise(tensor, std=0.02):
        return tensor + std * torch.randn_like(tensor)

    def get_state(self):
        return {
            "data": [],
            "rng": torch.get_rng_state(),
            "model": deepcopy(self.model.state_dict()),
            # "D_optimizer": deepcopy(self.D_optimizer.state_dict()),
            # "G_optimizer": deepcopy(self.G_optimizer.state_dict()),
        }

    def set_state(self, state):
        self.model.load_state_dict(state['model'])
        # self.D_optimizer.load_state_dict(state['D_optimizer'])
        # self.G_optimizer.load_state_dict(state['G_optimizer'])
        torch.set_rng_state(state['rng'])

    def compute_gradient(self, recompute_state={}) -> Tuple[float, int]:
        results = {}

        # cache batch size
        if self.batch_size is None:
            data, target = self.running["train_loader_iterator"].__next__()
            self.batch_size = data.shape[0]

        # reset prev state if not given fixed state
        self.prev_state = self.get_state()
        if recompute_state:
            self.set_state(recompute_state)

        def sample_data():
            if recompute_state:
                data, target = recompute_state['data'].pop(0)
            else:
                data, target = self.running["train_loader_iterator"].__next__()
                self.prev_state['data'].append((data, target))
            data, target = data.to(self.device), target.to(self.device)
            return data, target

        # save old params
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["old"] = p.detach().clone()

        ### Train discriminator ###
        for _ in range(10):
            for _ in range(self.D_iters):
                data, target = sample_data()
                D_metrics = self.compute_D_grad(data, target if self.conditional else None)
                self.D_optimizer.step()
            ### Train generator every `D_iters` ###
            G_metrics = self.compute_G_grad()
            self.G_optimizer.step()

        # save neg param diff as grad and restore old param
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    if p.grad is None:
                        p.grad = torch.empty_like(p)
                    delta = (param_state["old"] - p) / group['lr']
                    p.copy_(param_state["old"])
                    p.grad.copy_(delta)
                    param_state['grad'] = p.grad.detach().clone()

        self.running["data"] = data
        self.running["target"] = target
        results["length"] = len(target)
        results["loss"] = 0.0
        results["metrics"] = dict(**D_metrics, **G_metrics)

        self._save_grad()

        # if given state, reset optimizer to original state XXXXXXXXXXXXXX
        if recompute_state:
            self.set_state(self.prev_state)

        return results

    ########################################################################
    # The following are GAN-specific methods.
    @torch.no_grad()
    def init_fixed_sample(self):
        # Sample a global data point and latent to examine generator's progress
        self.fixed_x, self.fixed_y = next(iter(self.data_loader))
        self.fixed_x = self.fixed_x[:16].to(self.device)
        self.fixed_y = self.fixed_y[:16].to(self.device)
        self.fixed_latent = torch.randn(16*3, self.model.num_latents).to(self.device)

    @torch.no_grad()
    def update_G_progress(self):
        label = self.fixed_y if self.conditional else None
        fake_x = self.model.G(self.fixed_latent, cond=label)
        im_grid = torch.cat([self.fixed_x, fake_x], dim=0)
        im_grid = 0.5 * im_grid + 0.5  # inv_normalize to [0,1]
        grid = make_grid(im_grid, nrow=8, padding=2).cpu()
        self.progress_frames.append(grid)
        # downsample if progress_frames became too big
        if len(self.progress_frames) > self.progress_frames_maxlen and self.progress_frames_freq > 1:
            self.progress_frames = self.progress_frames[::2]
            self.progress_frames_freq = max(self.progress_frames_freq // 2, 1)
        return grid

    def generate_fake(self, batch_size):
        latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        fake_label = None
        if self.conditional:
            if hasattr(self.data_loader.dataset, "labels"):
                local_labels = self.data_loader.dataset.labels
                rand_label_idx = torch.randint(0, len(local_labels), (batch_size,))
                fake_label = local_labels[rand_label_idx].to(self.device)
            else:
                raise Exception("Can't really sample from local labels without knowing them.")
                # fake_label = torch.randint(0, self.model.num_classes, (batch_size,)).to(self.device)
        fake = self.model.G(latent, label=fake_label)
        return fake, fake_label

    def compute_D_grad(self, real, label=None):
        real = self.add_noise(real)
        # sample fake data
        with torch.no_grad():
            fake, fake_label = self.generate_fake(self.batch_size)
            fake = self.add_noise(fake)
        # Classify real and fake data
        D_real, h_real = self.model.D(real, label=label, return_h=True)
        D_fake, h_fake = self.model.D(fake, label=fake_label, return_h=True)
        # Adversarial loss
        D_loss = self.loss_func('D', D_real, D_fake)
        # Optimize
        self.D_optimizer.zero_grad()
        D_loss.backward()
        # return metrics
        return {'D->D(x)': D_real.mean().item(), 'D->D(G(z))': D_fake.mean().item()}

    def compute_G_grad(self):
        fake, fake_label = self.generate_fake(self.batch_size)
        fake = self.add_noise(fake)
        # Classify fake data
        D_fake = self.model.D(fake, label=fake_label)
        # Adversarial loss
        G_loss = self.loss_func('G', ..., D_fake)
        # Optimize
        self.G_optimizer.zero_grad()
        G_loss.backward()
        # return metrics
        return {'G->D(G(z))': D_fake.mean().item()}


class GANMomentumWorker(GANWorker, MomentumWorker):
    def __str__(self) -> str:
        return f"GANMomentumWorker [{self.worker_id}]"


class GANAdamWorker(GANWorker, AdamWorker):
    def __str__(self) -> str:
        return f"GANAdamWorker [{self.worker_id}]"


###############################################################################
class QuadraticGameWorker(TorchWorker):
    """
    A worker for quadratic games.
    """

    def __init__(self,
                 worker_id,
                 *args, **kwargs):
        self.worker_id = worker_id
        super().__init__(*args, **kwargs)
        self.optimizer1 = self.optimizer["player1"]
        self.optimizer2 = self.optimizer["player2"]
        self.optimizer = self.optimizer["all"]  # dummy optimizer for passing grads

    def __str__(self) -> str:
        return f"QuadraticGameWorker [{self.worker_id}]"

    def compute_gradient(self, given_state={}) -> Tuple[float, int]:
        results = {}

        # reset prev state if not given fixed state
        self.prev_state = {
            "data": [],
            "rng": [],
            "model": deepcopy(self.model.state_dict()),
            "optimizer1": deepcopy(self.optimizer1.state_dict()),
            "optimizer2": deepcopy(self.optimizer2.state_dict()),
        }

        if given_state:
            self.model.load_state_dict(given_state['model'])
            self.optimizer1.load_state_dict(given_state['optimizer1'])
            self.optimizer2.load_state_dict(given_state['optimizer2'])

        def sample_data():
            if given_state:
                data = given_state['data'].pop(0)
            else:
                data = self.running["train_loader_iterator"].__next__()
                self.prev_state['data'].append(data)
            data = [d.to(self.device) for d in data]
            return data

        def update_rng_state():
            if given_state:
                torch.set_rng_state(given_state['rng'].pop(0))
            else:
                self.prev_state['rng'].append(torch.get_rng_state())

        # save old params
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["old"] = p.detach().clone()

        ### Train player1 ###
        data = sample_data()
        update_rng_state()
        loss1 = self.loss_func(self.model.player1, self.model.player2, *data)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        ### Train player2 ###
        data = sample_data()
        update_rng_state()
        loss2 = self.loss_func(self.model.player1, self.model.player2, *data)
        self.optimizer2.zero_grad()
        (-loss2).backward()
        self.optimizer2.step()

        # save neg param diff as grad and restore old param
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    if p.grad is None:
                        p.grad = torch.empty_like(p)
                    delta = param_state["old"] - p
                    p.copy_(param_state["old"])
                    p.grad.copy_(delta)
                    param_state['grad'] = p.grad.detach().clone()

        self.running["data"] = data
        results["length"] = data[0].size(0)
        results["loss"] = (loss1.item() + loss2.item()) / 2
        results["metrics"] = {"loss1": loss1.item(), "loss2": loss2.item()}

        # recomputed grad should be accessed directly from p.grad
        if not given_state:
            self._save_grad()

        # if given state, reset optimizer to original state
        if given_state:
            self.model.load_state_dict(self.prev_state['model'])
            self.optimizer1.load_state_dict(self.prev_state['optimizer1'])
            self.optimizer2.load_state_dict(self.prev_state['optimizer2'])

        return results


class QuadraticGameMomentumWorker(QuadraticGameWorker, MomentumWorker):
    def __str__(self) -> str:
        return f"QuadraticGameMomentumWorker [{self.worker_id}]"

###############################################################################


class ByzantineWorker(TorchWorker):
    def configure(self, simulator):
        # call configure after defining DistribtuedSimulator
        self.simulator = simulator
        simulator.register_omniscient_callback(self.omniscient_callback)

    def compute_gradient(self) -> Tuple[float, int]:
        # Use self.simulator to get all other workers
        # Note that the byzantine worker does not modify the states directly.
        return super().compute_gradient()

    def get_gradient(self) -> torch.Tensor:
        # Use self.simulator to get all other workers
        return super().get_gradient()

    def omniscient_callback(self):
        raise NotImplementedError

    def __str__(self) -> str:
        return "ByzantineWorker"


class ByzantineQuadraticGameWorker(QuadraticGameWorker, ByzantineWorker):
    def __str__(self) -> str:
        return "ByzantineQuadraticGameWorker"


class ByzantineGANWorker(GANWorker, ByzantineWorker):
    def __str__(self) -> str:
        return "ByzantineGANWorker"
