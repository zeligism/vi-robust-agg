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
        worker_id: int,
        data_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        worker_steps: int = 1,
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

        self.worker_id = worker_id
        self.worker_steps = min(worker_steps, len(self.data_loader) - 1)
        self.batch_size = None
        self.num_iters = 0

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
        return f"TorchWorker [{self.worker_id}]"

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
                if p.grad is None:
                    continue
                layer_gradients.append(param_state["saved_grad"].data.view(-1))
        return torch.cat(layer_gradients)

    @torch.no_grad()
    def _cache_old_params(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["old"] = p.detach().clone()

    @torch.no_grad()
    def _set_delta_as_grad(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if p.grad is None:
                    p.grad = torch.empty_like(p)
                delta = (param_state["old"] - p) / group['lr']
                # p.copy_(param_state["old"])  # XXX
                p.grad.copy_(delta)
                param_state['grad'] = p.grad.detach().clone()

    def _get_model_state(self):
        return {
            "model": deepcopy(self.model.state_dict()),
            "num_iters": self.num_iters,
            "rng": torch.get_rng_state(),
        }

    def _set_model_state(self, state):
        self.model.load_state_dict(state['model'])
        self.num_iters = state['num_iters']
        torch.set_rng_state(state['rng'])

    def _sample_data(self, recompute_state={}):
        if recompute_state:
            data = recompute_state['data'].pop(0)
        else:
            data = self.running["train_loader_iterator"].__next__()
            self.prev_state['data'].append([d.detach() for d in data])
        data = [d.to(self.device) for d in data]
        return data


class MomentumWorker(TorchWorker):
    def __init__(self, momentum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

    def __str__(self) -> str:
        return f"MomentumWorker [{self.worker_id}]"

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

    def __str__(self) -> str:
        return f"AdamWorker [{self.worker_id}]"

    def _save_grad(self) -> None:
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                if p.grad is None:
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
                if "momentum_buffer" not in param_state:
                    continue
                m = param_state["momentum_buffer"].data.view(-1)
                v = param_state["momentumsq_buffer"].data.view(-1)
                if self.correct_bias:
                    m = m / (1 - self.betas[0] ** param_state['steps'])
                    v = v / (1 - self.betas[1] ** param_state['steps'])
                layer_gradients.append(m / (v**0.5 + self.eps))
        return torch.cat(layer_gradients)


# class ExtraGradientWorker(TorchWorker):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def __str__(self) -> str:
#         return f"ExtraGradientWorker [{self.worker_id}]"

#     def _save_grad(self) -> None:
#         for group in self.optimizer.param_groups:
#             for p in group["params"]:
#                 param_state = self.state[p]
#                 if p.grad is None:
#                     continue
#                 if "mid" not in param_state:
#                     param_state["prev"] = torch.clone(p).detach()
#                     param_state["mid"] = torch.clone(p).detach()
#                     param_state['steps'] = 1
#                 else:
#                     param_state["prev"] = torch.clone(param_state["mid"]).detach()
#                     param_state["mid"] = torch.clone(p).detach()
#                     param_state['steps'] += 1

#     def _get_saved_grad(self) -> torch.Tensor:
#         layer_gradients = []
#         for group in self.optimizer.param_groups:
#             for p in group["params"]:
#                 param_state = self.state[p]
#                 if "mid" not in param_state:
#                     continue
#                 g = param_state["mid"] - param_state["prev"] + p.grad
#                 layer_gradients.append(g.data.view(-1))
#         return torch.cat(layer_gradients)


###############################################################################
class GANWorker(TorchWorker):
    """
    A worker for GAN training.
    """
    METRICS = ['D->D(x)', 'D->D(G(z))', 'G->D(G(z))']

    def __init__(self,
                 D_iters: int = 3,
                 conditional: bool = False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, torch.nn.Module):
            self.model = self.model()
        self.D_iters = D_iters
        self.conditional = conditional
        if callable(self.optimizer):
            self.optimizer_dict = self.optimizer(self.model)
        else:
            self.optimizer_dict = self.optimizer
        self.D_optimizer = self.optimizer_dict["D"]
        self.G_optimizer = self.optimizer_dict["G"]
        self.optimizer = self.optimizer_dict["all"]  # dummy optimizer for passing grads
        self.init_fixed_sample()
        self.progress_frames = []
        self.progress_frames_freq = 4  # per epoch, better if = multiple of 2
        self.progress_frames_maxlen = 200
        self.raise_stopiter_later = False

        for p in self.model.parameters():
            self.state[p]["resync"] = True

    def __str__(self) -> str:
        return f"GANWorker [{self.worker_id}]"

    @staticmethod
    def add_noise(tensor, std=0.02):
        return tensor + std * torch.randn_like(tensor)

    def _compute_gradient(self, recompute_state={}) -> Tuple[float, int]:
        D_metrics = {}
        G_metrics = {}
        # # XXXXXXXX
        # self.worker_steps = self.D_iters if (self.num_iters + 1) % (self.D_iters + 1) > 0 else 1
        # # XXXXXXXX
        for _ in range(self.worker_steps):
            if (self.num_iters + 1) % (self.D_iters + 1) > 0:
                ### Train discriminator ###
                data, target = self._sample_data(recompute_state)
                length = len(target)
                D_metrics = self.compute_D_grad(data, target if self.conditional else None)
                self.D_optimizer.step()
            else:
                ### Train generator every `D_iters` ###
                data, target = None, None
                length = self.batch_size
                G_metrics = self.compute_G_grad()
                self.G_optimizer.step()
            # Update stats
            self.num_iters += 1
            self.running["data"] = data
            self.running["target"] = target
            self.results["length"] = length
            self.results["loss"] = 0.0
            self.results["metrics"] = dict(**D_metrics, **G_metrics)
            for metric in self.METRICS:
                if metric not in self.results["metrics"]:
                    self.results["metrics"][metric] = 0.0

    def train_epoch_start(self) -> None:
        super().train_epoch_start()
        self.raise_stopiter_later = False

    def compute_gradient(self, recompute_state={}) -> Tuple[float, int]:
        """
        Note 1: Recompute_state works only for SGD.
        Note 2: Some StopIteration acrobatics was done in order to achieve
                multi-local steps without changing the original simulator.
        """
        if self.raise_stopiter_later:
            self.raise_stopiter_later = False
            raise StopIteration

        # cache batch size
        if self.batch_size is None:
            data = self.running["train_loader_iterator"].__next__()
            self.batch_size = data[0].shape[0]

        # reset prev state if not given fixed state
        self.prev_state = self._get_model_state()
        self.prev_state["data"] = []
        if recompute_state:
            self._set_model_state(recompute_state)

        # Update worker's gradient
        self._cache_old_params()
        try:
            self.results = {}
            self._compute_gradient(recompute_state)
        except StopIteration:
            # always raise stopiters if results are empty,
            # otherwise raise later in order to compute this step.
            # note that this will never happen with `recompute_state`.
            if not self.results:
                raise StopIteration
            else:
                self.raise_stopiter_later = True
        self._set_delta_as_grad()
        self._save_grad()
        # reset to previous state when done recomputing
        if recompute_state:
            self._set_model_state(self.prev_state)

        return self.results

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
        self.model.G.eval()
        fake_x = self.model.G(self.fixed_latent, cond=label)
        self.model.G.train()
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
            elif hasattr(self.data_loader.dataset, "targets"):
                targets = self.data_loader.dataset.targets
                local_labels = torch.Tensor(list(set(t.item() for t in targets))).int()
                self.data_loader.dataset.labels = local_labels
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


###############################################################################
class QuadraticGameWorker(TorchWorker):
    """
    A worker for quadratic games.
    """

    def __init__(self,
                 sparsity=0.,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsity = sparsity
        self.optimizers = self.optimizer["players"]
        self.optimizer = self.optimizer["all"]  # dummy optimizer for passing grads
        self.num_iters = 0

    def __str__(self) -> str:
        return f"QuadraticGameWorker [{self.worker_id}]"

    def _compute_gradient(self, recompute_state={}) -> Tuple[float, int]:
        losses = []
        for _ in range(self.worker_steps):
            data = self._sample_data(recompute_state)
            turn = self.num_iters % len(self.model.players)
            if len(self.model.players) > 2:
                loss = self.loss_func(self.model.players, turn, *data)
                sign = 1
            else:
                loss = self.loss_func(self.model.players[0], self.model.players[1], *data)
                sign = (-1)**turn  # player2 maximizes objective
            reg = self.sparsity * torch.linalg.vector_norm(self.model.players[turn], ord=1)
            self.optimizers[turn].zero_grad()
            (sign * loss + reg).backward()
            self.optimizers[turn].step()
            losses.append(loss.detach())
            self.num_iters += 1
            self.running["data"] = data
            self.results["length"] = data[0].size(0)
            self.results["loss"] = torch.stack(losses).mean().item()
            self.results["metrics"] = {}

    def train_epoch_start(self) -> None:
        super().train_epoch_start()
        self.raise_stopiter_later = False

    def compute_gradient(self, recompute_state={}) -> Tuple[float, int]:
        """
        Note 1: Recompute_state works only for SGD.
        Note 2: Some StopIteration acrobatics was done in order to achieve
                multi-local steps without changing the original simulator.
        """
        if self.raise_stopiter_later:
            self.raise_stopiter_later = False
            raise StopIteration

        # cache batch size
        if self.batch_size is None:
            data = self.running["train_loader_iterator"].__next__()
            self.batch_size = data[0].shape[0]

        # reset prev state if not given fixed state
        self.prev_state = self._get_model_state()
        self.prev_state["data"] = []
        if recompute_state:
            self._set_model_state(recompute_state)

        # Update worker's gradient
        self._cache_old_params()
        try:
            self.results = {}
            self._compute_gradient(recompute_state)
        except StopIteration:
            # always raise stopiters if results are empty,
            # otherwise raise later in order to compute this step.
            # note that this will never happen with `recompute_state`.
            if not self.results:
                raise StopIteration
            else:
                self.raise_stopiter_later = True
        self._set_delta_as_grad()
        self._save_grad()
        # reset to previous state when done recomputing
        if recompute_state:
            self._set_model_state(self.prev_state)

        return self.results


###############################################################################
class TorchWorkerWithAdversary(TorchWorker):
    """
    A worker with an adversary.
    """

    def __init__(self, reg: float = 0., adv_reg: float = 1e10, scaled_reg: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model, torch.nn.Module):
            self.model = self.model()
        if callable(self.optimizer):
            self.optimizers = self.optimizer(self.model)
        else:
            self.optimizers = self.optimizer
        self.optimizer = self.optimizers["all"]  # dummy optimizer for passing grads
        self.reg = reg
        self.adv_reg = adv_reg
        if scaled_reg:
            self.reg /= sum(p.numel() for p in self.model.model.parameters())**0.5
            self.adv_reg /= sum(p.numel() for p in self.model.adversary.parameters())**0.5

    def __str__(self) -> str:
        return f"TorchWorkerWithAdversary [{self.worker_id}]"

    def _compute_gradient(self, recompute_state={}) -> Tuple[float, int]:
        losses = []
        for _ in range(self.worker_steps):
            data, target = self._sample_data(recompute_state)
            output = self.model(data)
            turn = "model" if self.num_iters % 2 == 0 else "adversary"
            sign = -1. if turn == "adversary" else 1.
            loss = self.loss_func(output, target)
            # regularization
            model_flat_params = torch.cat([p.view(-1) for p in self.model.model.parameters()])
            model_l2_norm = torch.linalg.vector_norm(model_flat_params, ord=2)
            adv_flat_params = torch.cat([p.view(-1) for p in self.model.adversary.parameters()])
            adv_l2_norm = torch.linalg.vector_norm(adv_flat_params, ord=2)
            if turn == "model":
                l2_reg = 0.5 * self.reg * model_l2_norm**2
            else:
                l2_reg = 0.5 * self.adv_reg * adv_l2_norm**2
            # optimize
            self.optimizers[turn].zero_grad()
            (sign * loss + l2_reg).backward()
            self.optimizers[turn].step()
            losses.append(loss.detach())
            self.num_iters += 1
            self.running["data"] = data
            self.results["length"] = data[0].size(0)
            self.results["loss"] = torch.stack(losses).mean().item()
            self.results["metrics"] = {}
            for name, metric in self.metrics.items():
                self.results["metrics"][name] = metric(output, target)
            self.results["metrics"]["model_l2_norm"] = model_l2_norm.item()
            self.results["metrics"]["adversary_l2_norm"] = adv_l2_norm.item()

    def train_epoch_start(self) -> None:
        super().train_epoch_start()
        self.raise_stopiter_later = False

    def compute_gradient(self, recompute_state={}) -> Tuple[float, int]:
        """
        Note 1: Recompute_state works only for SGD.
        Note 2: Some StopIteration acrobatics was done in order to achieve
                multi-local steps without changing the original simulator.
        """
        if self.raise_stopiter_later:
            self.raise_stopiter_later = False
            raise StopIteration

        # cache batch size
        if self.batch_size is None:
            data = self.running["train_loader_iterator"].__next__()
            self.batch_size = data[0].shape[0]

        # reset prev state if not given fixed state
        self.prev_state = self._get_model_state()
        self.prev_state["data"] = []
        if recompute_state:
            self._set_model_state(recompute_state)

        # Update worker's gradient
        self._cache_old_params()
        try:
            self.results = {}
            self._compute_gradient(recompute_state)
        except StopIteration:
            # always raise stopiters if results are empty,
            # otherwise raise later in order to compute this step.
            # note that this will never happen with `recompute_state`.
            if not self.results:
                raise StopIteration
            else:
                self.raise_stopiter_later = True
        self._set_delta_as_grad()
        self._save_grad()
        # reset to previous state when done recomputing
        if recompute_state:
            self._set_model_state(self.prev_state)

        return self.results


###############################################################################
### Extend Classes ###

class GANMomentumWorker(GANWorker, MomentumWorker):
    def __str__(self) -> str:
        return f"GANMomentumWorker [{self.worker_id}]"


class GANAdamWorker(GANWorker, AdamWorker):
    def __str__(self) -> str:
        return f"GANAdamWorker [{self.worker_id}]"


class QuadraticGameMomentumWorker(QuadraticGameWorker, MomentumWorker):
    def __str__(self) -> str:
        return f"QuadraticGameMomentumWorker [{self.worker_id}]"


class QuadraticGameAdamWorker(QuadraticGameWorker, AdamWorker):
    def __str__(self) -> str:
        return f"QuadraticGameAdamWorker [{self.worker_id}]"


class MomentumWorkerWithAdversary(TorchWorkerWithAdversary, MomentumWorker):
    def __str__(self) -> str:
        return f"MomentumWorkerWithAdversary [{self.worker_id}]"


class AdamWorkerWithAdversary(TorchWorkerWithAdversary, AdamWorker):
    def __str__(self) -> str:
        return f"AdamWorkerWithAdversary [{self.worker_id}]"


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


class ByzantineWorkerWithAdversary(TorchWorkerWithAdversary, ByzantineWorker):
    def __str__(self) -> str:
        return "ByzantineWorkerWithAdversary"
