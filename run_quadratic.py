
# Example 1:
# python run_quadratic.py -n 10 -f 0 --agg avg --lr 2e-4 --epochs 50 \
#     --quadratic-players 2 --quadratic-dim 100 --quadratic-ell 1000 --quadratic-mu 1. --quadratic-sparsity 0.001

# Example 2:
# python run_quadratic.py -n 10 -f 0 --agg avg --lr 1e-3 --epochs 50 \
#     --quadratic-players 5 --quadratic-dim 20 --quadratic-ell 1000 --quadratic-mu 1. --quadratic-sparsity 0.001

# Example 3:
# python run_quadratic.py -n 10 -f 2 --attack ALIE --agg avg --lr 1e-3 --epochs 50 \
#     --quadratic-players 5 --quadratic-dim 20 --quadratic-N 1000 --quadratic-ell 1000 \
#     --quadratic-mu 1. --quadratic-sparsity 0.001 --worker-steps 100 --num-peers 1

from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
args.quadratic = True

LOG_DIR = EXP_DIR + "run_quadratic/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.agg}_{args.attack}_lr{args.lr}_m{args.momentum}_seed{args.seed}"

main(args, LOG_DIR, args.epochs, args.max_batches_per_epoch)
