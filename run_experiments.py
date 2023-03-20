
# Run as `python run_experiments.py --gan --use-cuda`, for example.
# Only specify meta-args not hardcoded in this file

import os
import glob
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
    "epochs": 50,
    "batch_size": 64,
    "lr": 1e-3,
    "n": 10,
    "f": 2,
    "D_iters": 3,
}

QUADRATIC_DEFAULT_HP = {
    "epochs": 50,
    "batch_size": 16,
    "lr": 1e-4,
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
    "seed": range(1),
    "attack": ["NA", "BF", "LF", "IPM", "ALIE"],
    "worker_steps": [4],  # 'D_iters + 1' or 'quadratic_players' for full minimax step per agg
}

# Load experiment name automatically, argparser will handle the rest
EXP = "gan" if "--gan" in sys.argv else "quadratic"
log_dir = EXP_DIR + f"{EXP}/"
default_hp = GAN_DEFAULT_HP if EXP == "gan" else QUADRATIC_DEFAULT_HP

# Give other jobs a chance to avoid conflicts in file creation
time.sleep(3 * random())

def run_exp(current_log_dir):
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_lr{args.lr}_seed{args.seed}_wsteps{args.worker_steps}"
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        gan_output_dir = os.path.join(current_log_dir, "gan_output")
        model_paths = list(glob.glob(os.path.join(gan_output_dir, "*.pl")))
        if len(model_paths) > 0:
            print(f"Continuing training of experiment {current_log_dir}.")
            get_epoch = lambda pth: int(pth.split("model_epoch")[1].split(".")[0])
            model_paths = sorted(model_paths, key=get_epoch)
            args.load_model = model_paths[-1]  # load latest model
            current_log_dir += f"epoch{get_epoch(args.load_model)}"  # start from a new dir to avoid removing prev dir
            main(args, current_log_dir, args.epochs, 10**10)
        else:
            print(f"Experiment {current_log_dir} already exists.")

for hp_combination in product(*HP_SPACE.values()):
    args_dict = dict(**default_hp, **dict(zip(HP_SPACE.keys(), hp_combination)))
    if args_dict["attack"] == "NA":
        args_dict["f"] = 0
    if args_dict["attack"] == "LF" and EXP == "quadratic":
        continue

    ### Experiment setup 1 ###
    args = get_args(namespace=Namespace(**args_dict))
    args.agg = "rfa"
    args.bucketing = 2
    current_log_dir = log_dir + f"SGDA_RA/"
    run_exp(current_log_dir)

    ### Experiment setup 2 ###
    args = get_args(namespace=Namespace(**args_dict))
    if EXP == "gan":
        args.betas = (0.5, 0.9)
    else:
        args.momentum = 0.9
    args.agg = "rfa"
    args.bucketing = 2
    current_log_dir = log_dir + f"M_SGDA_RA/"
    run_exp(current_log_dir)

    ### Experiment setup 3 ###
    args = get_args(namespace=Namespace(**args_dict))
    args.num_peers = 1
    current_log_dir = log_dir + f"SGDA_CC/"
    run_exp(current_log_dir)

    ### Experiment setup 4 ###
    args = get_args(namespace=Namespace(**args_dict))
    args.extragradient = True
    args.agg = "tm"
    current_log_dir = log_dir + f"SEG_TM/"
    run_exp(current_log_dir)


