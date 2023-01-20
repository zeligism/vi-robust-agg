
# Example:
# python run_gan.py -n 20 -f 0 --agg avg --epochs 20 --batch-size 64 --lr 2e-4 --betas 0.5 0.9 --D-iters 3 --save-model-every 10 --worker-steps 12

from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
args.gan = True

LOG_DIR = EXP_DIR + "exp_gan/"

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
