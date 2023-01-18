
# Run as `python run_experiments.py --gan --use-cuda`, for example.
# Only specify meta-args not hardcoded in this file

import os
import sys
import time
from argparse import Namespace
from itertools import product
from random import random, shuffle

from utils import get_args
from utils import main
from utils import EXP_DIR


# Default hyperparameters for both experiments
GAN_DEFAULT_HP = {
    "epochs": 20,
    "batch_size": 128,
    "lr": 2e-4,
    "n": 20,
    "f": 4,
    "D_iters": 5,
}

QUADRATIC_DEFAULT_HP = {
    "epochs": 50,
    "batch_size": 16,
    "lr": 2e-4,
    "n": 20,
    "f": 4,
    "quadratic_players": 2,
    "quadratic_N": 1000,
    "quadratic_dim": 100,
    "quadratic_mu": 1.,
    "quadratic_ell": 1000.,
}

# Hyperparameters search space
HP_SPACE = {
    "seed": range(3),
    "attack": ["NA", "BF", "LF", "IPM", "ALIE"],
    "worker_steps": [1, 100],
}

# Load experiment name automatically, argparser will handle the rest
EXP = "gan" if "--gan" in sys.argv else "quadratic"
log_dir = EXP_DIR + f"{EXP}/"
default_hp = GAN_DEFAULT_HP if EXP == "gan" else QUADRATIC_DEFAULT_HP

# Give other jobs a chance to avoid conflicts in file creation
time.sleep(3 * random())

for hyperparams in product(*HP_SPACE.values()):
    hyperparams_dict = dict(zip(HP_SPACE.keys(), hyperparams))
    namespace = Namespace(**default_hp, **hyperparams_dict)

    ### Experiment setup 1 ###
    # sgd + robust aggregator
    args = get_args(namespace=namespace)
    if args.attack == "NA":
        args.f = 0
    args.agg = "rfa"
    args.bucketing = 2
    current_log_dir = log_dir + f"sgd_robust/"
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_lr{args.lr}_seed{args.seed}_wsteps{args.worker_steps}"
    # Skip if another job already started on this
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        print(f"Experiment {current_log_dir} already exists.")

    ### Experiment setup 2 ###
    # momentum/adam worker + robust aggregator
    args = get_args(namespace=namespace)
    if args.attack == "NA":
        args.f = 0
    if EXP == "gan":
        args.betas = (0.5, 0.9)
    else:
        args.momentum = 0.9
    args.agg = "rfa"
    args.bucketing = 2
    current_log_dir = log_dir + f"adam_robust/"
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_lr{args.lr}_seed{args.seed}_wsteps{args.worker_steps}"
    # Skip if another job already started on this
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        print(f"Experiment {current_log_dir} already exists.")

    ### Experiment setup 3 ###
    # sgd + avg aggregator + check of computation
    args = get_args(namespace=namespace)
    if args.attack == "NA":
        args.f = 0
    args.num_peers = 1
    current_log_dir = log_dir + f"sgd_cc/"
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_lr{args.lr}_seed{args.seed}_wsteps{args.worker_steps}"
    # Skip if another job already started on this
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        print(f"Experiment {current_log_dir} already exists.")

