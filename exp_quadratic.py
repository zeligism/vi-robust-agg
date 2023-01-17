# python exp_quadratic.py -n 10 -f 0 --agg qopt --quadratic-ell 50 --quadratic-mu 1.

from utils import get_args
from utils import main
from utils import EXP_DIR

args = get_args()
args.quadratic = True

LOG_DIR = EXP_DIR + "exp_quadratic/"

if args.identifier:
    LOG_DIR += f"{args.identifier}/"
elif args.debug:
    LOG_DIR += "debug/"
else:
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.agg}_{args.attack}_{args.momentum}_s{args.bucketing}_seed{args.seed}"

MAX_BATCHES_PER_EPOCH = 30
EPOCHS = args.epochs
# args.lr = 2e-4
# args.batch_size = 10

main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
