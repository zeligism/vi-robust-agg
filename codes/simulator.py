import logging
import numpy as np
import torch
from typing import Union, Callable, Any
from copy import deepcopy

from .worker import TorchWorker
from .server import TorchServer


class DistributedSimulatorBase(object):
    """Simulate distributed programs with low memory usage.

    Functionality:
    1. randomness control: numpy, torch, torch-cuda
    2. add workers

    This base class is used by both trainer and evaluator.
    """

    def __init__(self, metrics: dict, use_cuda: bool, debug: bool):
        """
        Args:
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.debug = debug
        self.workers = []

        self.json_logger = logging.getLogger("stats")
        self.debug_logger = logging.getLogger("debug")


class ParallelTrainer(DistributedSimulatorBase):
    """Synchronous and parallel training with specified aggregator."""

    def __init__(
        self,
        server: TorchServer,
        aggregator: Callable[[list], torch.Tensor],
        pre_batch_hooks: list,
        post_batch_hooks: list,
        max_batches_per_epoch: int,
        log_interval: int,
        metrics: dict,
        use_cuda: bool,
        debug: bool,
        extragradient: bool = False,
        optimistic: bool = False,
    ):
        """
        Args:
            aggregator (callable): A callable which takes a list of tensors and returns
                an aggregated tensor.
            max_batches_per_epoch (int): Set the maximum number of batches in an epoch.
                Usually used for debugging.
            log_interval (int): Control the frequency of logging training batches
            metrics (dict): dict of metric names and their functions
            use_cuda (bool): Use cuda or not
            debug (bool):
        """
        self.aggregator = aggregator
        self.server = server
        self.pre_batch_hooks = pre_batch_hooks or []
        self.post_batch_hooks = post_batch_hooks or []
        self.log_interval = log_interval
        self.max_batches_per_epoch = max_batches_per_epoch
        self.omniscient_callbacks = []
        self.random_states = {}
        self.extragradient = extragradient
        self.optimistic = optimistic
        super().__init__(metrics, use_cuda, debug)

    def aggregation_and_update(self, aggregated=None):
        # If there are Byzantine workers, ask them to craft attacks based on the updated models.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()

        if aggregated is None:
            gradients = self.parallel_get(lambda w: w.get_gradient())
            # XXX: remove a random gradient!
            gradients.pop(torch.randint(len(gradients), (1,)).item())
            aggregated = self.aggregator(gradients)

        # Assume that the model and optimizers are shared among workers.
        self.server.set_gradient(aggregated)
        self.server.apply_gradient()
        # self.update_moments()

        return aggregated

    def update_moments(self):

        def get_states(w, state):
            flat_states = []
            for group in w.optimizer.param_groups:
                for p in group["params"]:
                    param_state = w.state[p]
                    if state not in param_state:
                        continue  # TODO: byzantine?
                    flat_states.append(param_state[state].data.view(-1))
            return None if len(flat_states) == 0 else torch.cat(flat_states)

        momenta = self.parallel_get(lambda w: get_states(w, "momentum_buffer"))
        momenta = [momentum for momentum in momenta if momentum is not None]
        momenta_sq = self.parallel_get(lambda w: get_states(w, "momentumsq_buffer"))
        momenta_sq = [momentum_sq for momentum_sq in momenta_sq if momentum_sq is not None]
        if len(momenta) > 0:
            aggregated_momenta = self.aggregator(momenta)
        if len(momenta_sq) > 0:
            aggregated_momenta_sq = self.aggregator(momenta_sq)

        @torch.no_grad()
        def set_aggregate_momentum(w):
            i = 0
            for param in zip(w.model.parameters()):
                param_state = w.state[param]
                if "momentum_buffer" not in param_state:
                    continue
                j = i + param.numel()
                if len(momenta) > 0:
                    param_state["momentum_buffer"].copy_(aggregated_momenta[i::j])
                if len(momenta_sq) > 0:
                    param_state["momentumsq_buffer"].copy_(aggregated_momenta_sq[i::j])
                i = j

        self.parallel_call(set_aggregate_momentum)

    def train(self, epoch):
        self.debug_logger.info(f"Train epoch {epoch}")
        self.parallel_call(lambda worker: worker.train_epoch_start())

        @torch.no_grad()
        def resync_params(w):
            for param, global_param in zip(
                    w.model.parameters(), self.server.model.parameters()):
                if "resync" not in w.state[param] or w.state[param]["resync"]:
                    param.copy_(global_param.clone().detach())

        progress = 0
        prev_mid_aggregated = None  # TODO: make attr and put at init?
        self.parallel_call(resync_params)
        for batch_idx in range(self.max_batches_per_epoch):
            try:
                self._run_pre_batch_hooks(epoch, batch_idx)
                if self.extragradient:
                    self.server.save_param()  # orig params

                if self.optimistic and prev_mid_aggregated is not None:
                    self.aggregation_and_update(prev_mid_aggregated)  # prev mid grad
                else:
                    results = self.parallel_get(lambda w: w.compute_gradient())  # orig grad
                    self.aggregation_and_update()
                self.parallel_call(resync_params)

                if self.extragradient:
                    results = self.parallel_get(lambda w: w.compute_gradient())  # mid grad
                    self.server.load_param()  # orig params
                    prev_mid_aggregated = self.aggregation_and_update()
                    self.parallel_call(resync_params)

                progress += sum(res["length"] for res in results)
                if batch_idx % self.log_interval == 0:
                    self.log_train(progress, batch_idx, epoch, results)
                self._run_post_batch_hooks(epoch, batch_idx)
            except StopIteration:
                break

    # ---------------------------------------------------------------------------- #
    #                                    Utility                                   #
    # ---------------------------------------------------------------------------- #

    def add_worker(self, worker: TorchWorker):
        worker.add_metrics(self.metrics)
        self.workers.append(worker)
        self.debug_logger.info(f"=> Add worker {worker}")

    def register_omniscient_callback(self, callback):
        self.omniscient_callbacks.append(callback)

    def cache_random_state(self) -> None:
        if self.use_cuda:
            self.random_states["torch_cuda"] = torch.cuda.get_rng_state()
        self.random_states["torch"] = torch.get_rng_state()
        self.random_states["numpy"] = np.random.get_state()

    def restore_random_state(self) -> None:
        if self.use_cuda:
            torch.cuda.set_rng_state(self.random_states["torch_cuda"])
        torch.set_rng_state(self.random_states["torch"])
        np.random.set_state(self.random_states["numpy"])

    def parallel_call(self, f: Callable[[TorchWorker], None]) -> None:
        for w in self.workers:
            self.cache_random_state()
            f(w)
            self.restore_random_state()

    def parallel_get(self, f: Callable[[TorchWorker], Any]) -> list:
        results = []
        for w in self.workers:
            self.cache_random_state()
            results.append(f(w))
            self.restore_random_state()
        return results

    def _run_pre_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.pre_batch_hooks]

    def _run_post_batch_hooks(self, epoch, batch_idx):
        [f(self, epoch, batch_idx) for f in self.post_batch_hooks]

    # ---------------------------------------------------------------------------- #
    #                                Log information                               #
    # ---------------------------------------------------------------------------- #

    def __str__(self):
        return (
            "ParallelTrainer("
            f"aggregator={self.aggregator}, "
            f"max_batches_per_epoch={self.max_batches_per_epoch}, "
            f"log_interval={self.log_interval}, "
            f"metrics={list(self.metrics.keys())}"
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def log_train(self, progress, batch_idx, epoch, results):
        length = sum(res["length"] for res in results)

        r = {
            "_meta": {"type": "train"},
            "E": epoch,
            "B": batch_idx,
            "Length": length,
            "Loss": sum(res["loss"] * res["length"] for res in results) / length,
        }

        for metric_name in self.metrics:
            r[metric_name] = (
                sum(res["metrics"][metric_name] * res["length"] for res in results)
                / length
            )

        # Output to console
        total = len(self.workers[0].data_loader.dataset)
        pct = 100 * progress / total
        self.debug_logger.info(
            f"[E{r['E']:2}B{r['B']:<3}| {progress:6}/{total} ({pct:3.0f}%) ] Loss: {r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
        )

        # Output to file
        self.json_logger.info(r)


class DistributedEvaluator(DistributedSimulatorBase):
    def __init__(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        loss_func: torch.nn.modules.loss._Loss,
        device: Union[torch.device, str],
        metrics: dict,
        use_cuda: bool,
        debug: bool,
        log_identifier_type="validation",
    ):
        super().__init__(metrics, use_cuda, debug)
        self.model = model
        self.data_loader = data_loader
        self.loss_func = loss_func
        self.device = device
        self.log_identifier_type = log_identifier_type

    def __str__(self):
        return (
            "DistributedEvaluator("
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def evaluate(self, epoch):
        self.model.eval()
        r = {
            "_meta": {"type": self.log_identifier_type},
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0

        with torch.no_grad():
            for _, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                r["Loss"] += self.loss_func(output, target).item() * len(target)
                r["Length"] += len(target)

                for name, metric in self.metrics.items():
                    r[name] += metric(output, target) * len(target)

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )


class QuadraticGameEvaluator(DistributedEvaluator):
    def __str__(self):
        return (
            "QuadraticGameEvaluator("
            f"use_cuda={self.use_cuda}, "
            f"debug={self.debug}, "
            ")"
        )

    def evaluate(self, epoch):
        self.model.eval()
        r = {
            "_meta": {"type": self.log_identifier_type},
            "E": epoch,
            "Length": 0,
            "Loss": 0,
        }
        for name in self.metrics:
            r[name] = 0

        with torch.no_grad():
            validation_loss = 0.
            for _, data in enumerate(self.data_loader):
                data = [d.to(self.device) for d in data]
                length = data[0].size(0)
                loss = 0.
                for turn in range(len(self.model.players)):
                    if len(self.model.players) > 2:
                        loss += self.loss_func(self.model.players, turn, *data)
                    else:
                        loss += self.loss_func(self.model.players[0], self.model.players[1], *data)
                validation_loss += loss.item() / len(self.data_loader)
        r["Loss"] += validation_loss * length
        r["Length"] += length

        for name, metric in self.metrics.items():
            r[name] += 0.0

        for name in self.metrics:
            r[name] /= r["Length"]
        r["Loss"] /= r["Length"]

        # Output to file
        self.json_logger.info(r)
        self.debug_logger.info(
            f"\n=> Eval Loss={r['Loss']:.4f} "
            + " ".join(name + "=" + "{:>8.4f}".format(r[name]) for name in self.metrics)
            + "\n"
        )


class ParallelTrainerCC(ParallelTrainer):
    """
    A parallel trainer with checks of computations of gradients.
    NOTE: Works only with SGD (`w.get_gradient()` may not return exact computation due to momentum, etc.)
    """

    def __init__(self, num_peers: int = 0, real_check: bool = False, *args, **kwargs):
        """
        Args:
            num_checks (int): num of peers to validate other workers grad
        """
        super().__init__(*args, **kwargs)
        self.num_peers = num_peers
        self.real_check = real_check
        self.tolerance = 0.5
        self.banned = set()
        self.previous_validators = []
        print("--- Check of Computation ---")

    def aggregation_and_update(self, aggregated=None):
        # If there are Byzantine workers, ask them to craft attacks based on the updated models.
        for omniscient_attacker_callback in self.omniscient_callbacks:
            omniscient_attacker_callback()

        #-------- Check Computation --------#
        # sample validators and targets
        while True:
            shuffled = torch.randperm(len(self.workers)).tolist()
            shuffled_filtered = [
                w for w in shuffled
                if w not in self.banned and w not in self.previous_validators
            ]
            sampled = shuffled_filtered[:2 * self.num_peers]
            if len(shuffled_filtered) == len(sampled):
                # num of sampled peers should be less than total number of workers,
                # otherwise we might ban them all!
                self.num_peers = self.num_peers // 2
                validators, targets = [], []
                if self.num_peers == 0:
                    break  # no more peers :(
            else:
                validators, targets = sampled[0::2], sampled[1::2]
                break

        self.previous_validators = validators

        if self.debug:
            print(f"validators = {validators}, targets = {targets}")

        def sends_grad(w):
            return w.worker_id not in self.banned and w.worker_id not in validators

        if aggregated is None:
            gradients = self.parallel_get(lambda w: w.get_gradient() if sends_grad(w) else None)
            aggregated = self.aggregator([g for g in gradients if g is not None])

        if self.real_check:
            orig_rng_state = torch.get_rng_state()
            for validator, target in zip(validators, targets):
                # Get target's state
                if "data" not in self.workers[target].prev_state \
                        or len(self.workers[target].prev_state["data"]) == 0:
                    if self.debug:
                        print("Skipping validation as target's previous iterate is data-independent.")
                    continue
                prev_target_state = deepcopy(self.workers[target].prev_state)
                # The validator recomputes the grad at target's state and checks for significant mismatch
                target_grad = gradients[target]
                self.workers[validator].compute_gradient(recompute_state=prev_target_state)
                target_grad_by_validator = self.workers[validator].get_gradient()
                mismatch = torch.linalg.vector_norm(target_grad - target_grad_by_validator, ord=2).item()
                norm = torch.linalg.vector_norm(target_grad, ord=2).item()
                rel_error = float('inf') if norm == 0 else mismatch / norm
                if rel_error > self.tolerance:
                    self.banned.add(validator)
                    self.banned.add(target)
                    if self.debug:
                        print(f"Banning worker {validator} and {target}"
                              f" (reason: {validator} accused {target}, rel_error={rel_error})")
            torch.set_rng_state(orig_rng_state)
        else:
            GoodWorker = type(self.workers[0])  # worker 0 is a ref for good worker
            for validator, target in zip(validators, targets):
                validator_is_good = isinstance(self.workers[validator], GoodWorker)
                target_is_good = isinstance(self.workers[target], GoodWorker)
                if validator_is_good and target_is_good:
                    # good workers don't accuse each other
                    pass
                elif not validator_is_good and not target_is_good:
                    # bad workers don't accuse each other
                    pass
                else:
                    if self.debug:
                        print(f"Banning worker {validator} and {target}.")
                    # good workers accuse bad workers, and bad workers accuse good workers.
                    self.banned.add(validator)
                    self.banned.add(target)

        #--------------------------------#

        # Assume that the model and optimizers are shared among workers.
        self.server.set_gradient(aggregated)
        self.server.apply_gradient()
