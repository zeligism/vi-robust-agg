
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


# Default hyperparameters
ADV_DEFAULT_HP = {
    "epochs": 50,
    "n": 20,
    "f": 4,
    "lr": 1e-2,
    "batch_size": 32,
}

EXPERIMENT = 4

if EXPERIMENT == 1:
    HP_SPACE = {
        "seed": range(3),
        "attack": ["NA", "LF", "ALIE"],
        "reg": [0, 1],
        "adv_reg": [0, 0.01, 0.1, 1, 10, 100],
        "adv_strength": [1, 10],
        "worker_steps": [1, 2],
    }
elif EXPERIMENT == 2:
    HP_SPACE = {
        "seed": range(5),
        "attack": ["NA", "LF", "IPM", "ALIE"],
        "reg": [0, 1],
        "adv_reg": [0, 1],
        "worker_steps": [1, 2],
        "momentum": [0.0, 0.9],
    }
elif EXPERIMENT == 3:
    HP_SPACE = {
        "seed": range(3),
        "attack": ["NA", "LF", "BF", "IPM", "ALIE"],
        "reg": [0, 0.5],
        "adv_reg": [1e-2, 1e0, 1e2],
    }

elif EXPERIMENT == 4:
    HP_SPACE = {
        "seed": range(3),
        "attack": ["NA", "LF", "BF", "IPM", "ALIE"],
        "reg": [0, 0.5],
        "adv_reg": [0, 1e2],
        "worker_steps": [1, 2],
    }

# Load experiment name automatically, argparser will handle the rest
log_dir = EXP_DIR + f"adv{EXPERIMENT}/"
default_hp = ADV_DEFAULT_HP

# Give other jobs a chance to avoid conflicts in file creation
time.sleep(3 * random())

for hp_combination in product(*HP_SPACE.values()):
    args_dict = dict(**default_hp, **dict(zip(HP_SPACE.keys(), hp_combination)))
    args_dict["adversarial"] = True
    if args_dict["attack"] == "NA":
        args_dict["f"] = 0

    ### Experiment setup 1 ###
    # sgd + robust aggregator
    args = get_args(namespace=Namespace(**args_dict))
    args.agg = "rfa"
    args.bucketing = 2
    current_log_dir = log_dir + f"SGDA_RA/"
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_seed{args.seed}"
    current_log_dir += f"_reg{args.reg}_advreg{args.adv_reg}_m{args.momentum}_wsteps{args.worker_steps}"
    # Skip if another job already started on this
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        print(f"Experiment {current_log_dir} already exists.")

    ### Experiment setup 2 ###
    # momentum/adam worker + robust aggregator
    args = get_args(namespace=Namespace(**args_dict))
    args.momentum = 0.9
    args.agg = "rfa"
    args.bucketing = 2
    current_log_dir = log_dir + f"M_SGDA_RA/"
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_seed{args.seed}"
    current_log_dir += f"_reg{args.reg}_advreg{args.adv_reg}_m{args.momentum}_wsteps{args.worker_steps}"
    # Skip if another job already started on this
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        print(f"Experiment {current_log_dir} already exists.")

    ### Experiment setup 3 ###
    # sgd + extragradient + trimmed mean
    args = get_args(namespace=Namespace(**args_dict))
    args.agg = "tm"
    args.extragradient = True
    current_log_dir = log_dir + f"SEG_TM/"
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_seed{args.seed}"
    current_log_dir += f"_reg{args.reg}_advreg{args.adv_reg}_m{args.momentum}_wsteps{args.worker_steps}"
    # Skip if another job already started on this
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        print(f"Experiment {current_log_dir} already exists.")

    ### Experiment setup 4 ###
    # sgd + avg aggregator + check of computation
    args = get_args(namespace=Namespace(**args_dict))
    args.num_peers = 1
    args.agg = "avg"
    current_log_dir = log_dir + f"SGDA_CC/"
    current_log_dir += f"n{args.n}_f{args.f}_{args.agg}_{args.attack}_seed{args.seed}"
    current_log_dir += f"_reg{args.reg}_advreg{args.adv_reg}_m{args.momentum}_wsteps{args.worker_steps}"
    # Skip if another job already started on this
    if not os.path.exists(current_log_dir):
        main(args, current_log_dir, args.epochs, 10**10)
    else:
        print(f"Experiment {current_log_dir} already exists.")

