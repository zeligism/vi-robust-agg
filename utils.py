import argparse
import numpy as np
import os
import torch
from collections import Counter
from PIL import Image
from torchvision import datasets
import torch.nn.functional as F

# Utility functions
from codes.tasks.mnist import mnist, Net
from codes.utils import top1_accuracy, initialize_logger, quadratic_loss
from codes.gan_utils import get_GAN_loss_func

# Attacks
from codes.attacks.labelflipping import LableFlippingWorker
from codes.attacks.bitflipping import BitFlippingWorker
from codes.attacks.mimic import MimicAttacker, MimicVariantAttacker
from codes.attacks.xie import IPMAttack
from codes.attacks.alittle import ALittleIsEnoughAttack

# Main Modules
from codes.worker import (
    TorchWorker,
    MomentumWorker,
    QuadraticGameWorker,
    QuadraticGameMomentumWorker,
    ByzantineQuadraticGameWorker,
    GANWorker,
    GANMomentumWorker,
    GANAdamWorker,
    ByzantineGANWorker,
)
from codes.server import TorchServer
from codes.simulator import (
    ParallelTrainer,
    ParallelTrainerCC,
    DistributedEvaluator,
    QuadraticGameEvaluator,
)

# IID vs Non-IID
from codes.sampler import (
    DistributedSampler,
    DecentralizedNonIIDSampler,
    NONIIDLTSampler,
)

# Aggregators
from codes.aggregator.base import Mean
from codes.aggregator.qopt import QuadraticOptimal
from codes.aggregator.coordinatewise_median import CM
from codes.aggregator.clipping import Clipping
from codes.aggregator.rfa import RFA
from codes.aggregator.trimmed_mean import TM
from codes.aggregator.krum import Krum

# Quadratic games and GANs
from codes.tasks.quadratic import quadratic_game, TwoPlayers, generate_quadratic_game_dataset
from codes.tasks.gan import mnist32, ResNetGAN, ConditionalResNetGAN
from codes.gan_utils import tensor_to_np, make_animation

QUADRATIC_GAME_DATA = None


def get_args():
    parser = argparse.ArgumentParser(description="")

    # Utility
    parser.add_argument("--use-cuda", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--identifier", type=str, default="debug", help="")
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="If plot is enabled, then ignore all other options.",
    )

    # Experiment configuration
    parser.add_argument("-n", type=int, help="Number of workers")
    parser.add_argument("-f", type=int, help="Number of Byzantine workers.")
    parser.add_argument("--attack", type=str, default="NA", help="Type of attacks.")
    parser.add_argument("--agg", type=str, default="avg", help="")
    parser.add_argument(
        "--noniid",
        action="store_true",
        default=False,
        help="[HP] noniidness.",
    )
    parser.add_argument("--LT", action="store_true", default=False, help="Long tail")

    # Key hyperparameter
    parser.add_argument("--bucketing", type=int, default=0, help="[HP] s")
    parser.add_argument("--momentum", type=float, default=0.0, help="[HP] momentum")

    parser.add_argument("--clip-tau", type=float, default=10.0, help="[HP] momentum")
    parser.add_argument("--clip-scaling", type=str, default=None, help="[HP] momentum")

    parser.add_argument(
        "--mimic-warmup", type=int, default=1, help="the warmup phase in iterations."
    )

    parser.add_argument(
        "--op",
        type=int,
        default=1,
        help="[HP] controlling the degree of overparameterization. "
        "Only used in exp8.",
    )

    # Quadratic Game
    parser.add_argument("--quadratic", action="store_true", default=False,
                        help="Setup for quadratic games (ignores other setups except args.gan)")
    parser.add_argument("--quadratic-N", type=int, default=10000,
                        help="[HP] size of quadratic game dataset")
    parser.add_argument("--quadratic-dim", type=int, default=10,
                        help="[HP] dimension of quadratic game")
    # GAN
    parser.add_argument("--gan", action="store_true", default=False, help="Setup for GAN training (ignores other setups)")
    parser.add_argument("--conditional", action="store_true", default=False, help="Conditional GAN")
    parser.add_argument("--D-iters", type=int, default=1, help="[HP] D iters per G iter")
    parser.add_argument("--betas", type=float, default=(0.0, 0.0), nargs=2, help="[HP] betas for AdamWorker")
    # Check Computation
    parser.add_argument("--num-peers", type=int, default=0, help="[HP] num of peers for checking grad validity")

    args = parser.parse_args()

    if args.n <= 0 or args.f < 0 or args.f >= args.n:
        raise RuntimeError(f"n={args.n} f={args.f}")

    assert args.bucketing >= 0, args.bucketing
    assert args.momentum >= 0, args.momentum
    assert len(args.identifier) > 0
    return args


ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
DATA_DIR = ROOT_DIR + "datasets/"
EXP_DIR = ROOT_DIR + f"outputs/"

LR = 0.01
# Fixed HPs
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128


def _get_aggregator(args):
    if args.agg == "avg":
        return Mean()

    if args.agg == "qopt":
        return QuadraticOptimal()

    if args.agg == "cm":
        return CM()

    if args.agg == "cp":
        if args.clip_scaling is None:
            tau = args.clip_tau
        elif args.clip_scaling == "linear":
            tau = args.clip_tau / (1 - args.momentum)
        elif args.clip_scaling == "sqrt":
            tau = args.clip_tau / np.sqrt(1 - args.momentum)
        else:
            raise NotImplementedError(args.clip_scaling)
        return Clipping(tau=tau, n_iter=3)

    if args.agg == "rfa":
        return RFA(T=8)

    if args.agg == "tm":
        return TM(b=args.f)

    if args.agg == "krum":
        T = int(np.ceil(args.n / args.bucketing)) if args.bucketing > 0 else args.n
        return Krum(n=T, f=args.f, m=1)

    raise NotImplementedError(args.agg)


def bucketing_wrapper(args, aggregator, s):
    """
    Key functionality.
    """
    print("Using bucketing wrapper.")

    def aggr(inputs):
        indices = list(range(len(inputs)))
        np.random.shuffle(indices)

        T = int(np.ceil(args.n / s))

        reshuffled_inputs = []
        for t in range(T):
            indices_slice = indices[t * s : (t + 1) * s]
            g_bar = sum(inputs[i] for i in indices_slice) / len(indices_slice)
            reshuffled_inputs.append(g_bar)
        return aggregator(reshuffled_inputs)

    return aggr


def get_aggregator(args):
    aggr = _get_aggregator(args)
    if args.bucketing == 0:
        return aggr

    return bucketing_wrapper(args, aggr, args.bucketing)


def get_sampler_callback(args, rank):
    """
    Get sampler based on the rank of a worker.
    The first `n-f` workers are good, and the rest are Byzantine
    """
    n_good = args.n - args.f
    sampler_opts = dict(num_replicas=n_good, shuffle=True)

    def get_sampler(dataset):
        if rank >= n_good or args.quadratic:
            # Byzantine workers, or quadratic games in general
            return DistributedSampler(dataset=dataset,
                                      rank=rank % (n_good),
                                      **sampler_opts)
        else:
            return NONIIDLTSampler(dataset=dataset,
                                   rank=rank,
                                   alpha=not args.noniid,
                                   beta=0.5 if args.LT else 1.0,
                                   **sampler_opts)

    return get_sampler


def get_test_sampler_callback(args):
    # This alpha argument is not important as there is only 1 replica
    sampler_opts = dict(num_replicas=1, rank=0, shuffle=False)

    def get_test_sampler(dataset):
        if args.quadratic:
            return DistributedSampler(dataset=dataset, **sampler_opts)
        else:
            return NONIIDLTSampler(dataset=dataset,
                                   alpha=True,
                                   beta=0.5 if args.LT else 1.0,
                                   **sampler_opts)

    return get_test_sampler


def initialize_worker(
    args,
    trainer,
    worker_rank,
    model,
    optimizer,
    loss_func,
    device,
    kwargs,
):
    if args.gan or not args.quadratic:
        dataset = mnist32 if args.gan else mnist
        train_loader = dataset(
            data_dir=DATA_DIR,
            train=True,
            download=True,
            batch_size=BATCH_SIZE,
            sampler_callback=get_sampler_callback(args, worker_rank),
            dataset_cls=datasets.MNIST,
            drop_last=True,  # Exclude the influence of non-full batch.
            **kwargs,
        )
    else:
        train_loader = quadratic_game(
            data=QUADRATIC_GAME_DATA,
            batch_size=BATCH_SIZE,
            sampler_callback=get_sampler_callback(args, worker_rank),
            drop_last=True,
            **kwargs,
        )

    # Define worker opts
    default_worker_opts = dict(data_loader=train_loader,
                               model=model,
                               loss_func=loss_func,
                               device=device,
                               optimizer=optimizer,
                               **kwargs)
    if args.gan:
        worker_opts = dict(worker_id=worker_rank,
                           D_iters=args.D_iters,
                           conditional=args.conditional,
                           **default_worker_opts)
    elif args.quadratic:
        worker_opts = dict(worker_id=worker_rank, **default_worker_opts)
    else:
        worker_opts = dict(**default_worker_opts)

    # The first n - f workers are benign workers
    if worker_rank < args.n - args.f:
        if args.gan:
            return GANAdamWorker(betas=args.betas, **worker_opts)
        elif args.quadratic:
            return QuadraticGameMomentumWorker(momentum=args.momentum, **worker_opts)
        else:
            return MomentumWorker(momentum=args.momentum, **worker_opts)

    else:
        if args.attack == "BF":
            Attacker = BitFlippingWorker
            attacker_opts = {}
            should_configure = False

        elif args.attack == "LF":
            Attacker = LableFlippingWorker
            attacker_opts = dict(revertible_label_transformer=lambda target: 9 - target,)
            should_configure = False

        elif args.attack == "mimic":
            Attacker = MimicVariantAttacker
            attacker_opts = dict(warmup=args.mimic_warmup,)
            should_configure = True

        elif args.attack == "IPM":
            Attacker = IPMAttack
            attacker_opts = dict(epsilon=0.1)
            should_configure = True

        elif args.attack == "ALIE":
            Attacker = ALittleIsEnoughAttack
            attacker_opts = dict(n=args.n, m=args.f,)# z=1.5,)
            should_configure = True

        else:
            raise NotImplementedError(f"No such attack {args.attack}")

        # This is a nice trick for redefining byzantine attacks for other workers
        if args.gan:
            class GANAttacker(Attacker, ByzantineGANWorker):
                pass
            attacker = GANAttacker(**attacker_opts, **worker_opts)
        elif args.quadratic:
            class QuadraticGameAttacker(Attacker, ByzantineQuadraticGameWorker):
                pass
            attacker = QuadraticGameAttacker(**attacker_opts, **worker_opts)
        else:
            attacker = Attacker(**attacker_opts, **worker_opts)

        if should_configure:
            attacker.configure(trainer)

        return attacker


def main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH):
    initialize_logger(LOG_DIR)

    if args.use_cuda and not torch.cuda.is_available():
        print("=> There is no cuda device!!!!")
        device = "cpu"
    else:
        device = torch.device("cuda" if args.use_cuda else "cpu")
    # kwargs = {"num_workers": 1, "pin_memory": True} if args.use_cuda else {}
    kwargs = {"pin_memory": True} if args.use_cuda else {}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ### GAN Setup ###
    if args.gan:
        LR = 1e-3
        client_lr = LR
        server_lr = LR
        if args.conditional:
            model = ConditionalResNetGAN().to(device)
        else:
            model = ResNetGAN().to(device)
        print("D num of params:", sum(p.numel() for p in model.D.parameters()))
        print("G num of params:", sum(p.numel() for p in model.G.parameters()))
        # optimizers = [{
        #     "D": torch.optim.SGD(model.D.parameters(), lr=client_lr * 2),
        #     "G": torch.optim.SGD(model.G.parameters(), lr=client_lr),
        #     "all": torch.optim.SGD(model.parameters(), lr=client_lr),
        # } for _ in range(args.n)]
        # server_opt = torch.optim.SGD(model.parameters(), lr=server_lr)
        betas = (0.5, 0.9)
        optimizers = [{
            "D": torch.optim.Adam(model.D.parameters(), lr=client_lr * 2, betas=betas),
            "G": torch.optim.Adam(model.G.parameters(), lr=client_lr, betas=betas),
            "all": torch.optim.Adam(model.parameters(), lr=client_lr),
        } for _ in range(args.n)]
        server_opt = torch.optim.Adam(model.parameters(), lr=server_lr, betas=betas)
        loss_func = get_GAN_loss_func()
        # Save GAN snapshots to track progress
        out_dir = os.path.join(LOG_DIR, "gan_output")
        os.makedirs(out_dir, exist_ok=True)

        def save_snapshot_hook(trainer, epoch, batch_idx):

            def save_snapshot(w):
                if batch_idx % (len(w.data_loader) // w.progress_frames_freq) == 0:
                    frame = w.update_G_progress()
                    fp = os.path.join(out_dir, f'w{w.worker_id:02d}_epoch{epoch:03d}_batch{batch_idx:04d}.png')
                    im = Image.fromarray(tensor_to_np(frame))
                    im.save(fp)
                # Also store progress video for last iter
                if epoch == EPOCHS - 1 and batch_idx == len(w.data_loader) - 1:
                    fp = os.path.join(out_dir, f'w{w.worker_id:02d}_progress.mp4')
                    make_animation(w.progress_frames, fp)

            trainer.parallel_call(save_snapshot)

        post_batch_hooks = [save_snapshot_hook]
        # metrics are calculated during the run, just provide any dummy function
        metrics = {"D->D(x)": lambda x: x, "D->D(G(z))": lambda x: x, "G->D(G(z))": lambda x: x}

    ### Quadratic Game Setup ###
    elif args.quadratic:
        global QUADRATIC_GAME_DATA
        if QUADRATIC_GAME_DATA is None:
            print("Generating dataset for quadratic game...")
            QUADRATIC_GAME_DATA = generate_quadratic_game_dataset(N=args.quadratic_N,
                                                                  dim=args.quadratic_dim)
        client_lr = LR
        server_lr = LR
        model = TwoPlayers(dim=args.quadratic_dim)
        optimizers = [{
            "player1": torch.optim.SGD([model.player1], lr=client_lr),
            "player2": torch.optim.SGD([model.player2], lr=client_lr),
            "all": torch.optim.SGD(model.parameters(), lr=client_lr),
        } for _ in range(args.n)]
        server_opt = torch.optim.SGD(model.parameters(), lr=server_lr)
        loss_func = quadratic_loss
        post_batch_hooks = []
        metrics = {"loss1": lambda x: x, "loss2": lambda x: x,}

    ### Normal Setup ###
    else:
        model = Net().to(device)
        optimizers = [torch.optim.SGD(model.parameters(), lr=LR) for _ in range(args.n)]
        post_batch_hooks = []
        metrics = {"top1": top1_accuracy}
        server_opt = torch.optim.SGD(model.parameters(), lr=LR)
        loss_func = F.nll_loss

    ### Server ###
    server = TorchServer(optimizer=server_opt)
    if args.num_peers == 0:
        Trainer = ParallelTrainer
        trainer_kwargs = {}
    else:
        Trainer = ParallelTrainerCC
        trainer_kwargs = {'num_peers': args.num_peers}

    ### Simulator ###
    trainer = Trainer(
        server=server,
        aggregator=get_aggregator(args),
        pre_batch_hooks=[],
        post_batch_hooks=post_batch_hooks,
        max_batches_per_epoch=MAX_BATCHES_PER_EPOCH,
        log_interval=args.log_interval,
        metrics=metrics,
        use_cuda=args.use_cuda,
        debug=False,
        **trainer_kwargs,
    )

    ### Test set ###
    if not args.gan:
        if args.quadratic:
            test_loader = quadratic_game(
                data=QUADRATIC_GAME_DATA,
                batch_size=TEST_BATCH_SIZE,
                shuffle=False,
                sampler_callback=get_test_sampler_callback(args),
                **kwargs,
            )
            Evaluator = QuadraticGameEvaluator
        else:
            test_loader = mnist(
                data_dir=DATA_DIR,
                train=False,
                download=True,
                batch_size=TEST_BATCH_SIZE,
                shuffle=False,
                sampler_callback=get_test_sampler_callback(args),
                **kwargs,
            )
            Evaluator = DistributedEvaluator

        evaluator = Evaluator(
            model=model,
            data_loader=test_loader,
            loss_func=loss_func,
            device=device,
            metrics=metrics,
            use_cuda=args.use_cuda,
            debug=False,
        )

    ### Init Workers ###
    for worker_rank in range(args.n):
        worker = initialize_worker(
            args,
            trainer,
            worker_rank,
            model=model,
            optimizer=optimizers[worker_rank],
            loss_func=loss_func,
            device=device,
            kwargs={},
        )
        trainer.add_worker(worker)

    ### Run ###
    if not args.dry_run:
        for epoch in range(1, EPOCHS + 1):
            trainer.train(epoch)
            if not args.gan:
                evaluator.evaluate(epoch)
            trainer.parallel_call(lambda w: w.data_loader.sampler.set_epoch(epoch))
