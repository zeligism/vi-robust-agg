
# Example:
# python run_normal.py -n 10 -f 3 --attack LF --agg qopt --noniid --momentum 0.9

from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()

LOG_DIR = EXP_DIR + "run_normal/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.agg}_{args.attack}_lr{args.lr}_m{args.momentum}_seed{args.seed}"

MAX_BATCHES_PER_EPOCH = 10**10
EPOCHS = args.epochs
# args.lr = 0.01
# args.batch_size = 32

main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
