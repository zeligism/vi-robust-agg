
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
    LOG_DIR += f"n{args.n}_f{args.f}_{args.noniid}/"

INP_DIR = LOG_DIR
OUT_DIR = LOG_DIR + "output/"
LOG_DIR += f"{args.agg}_{args.attack}_{args.momentum}_s{args.bucketing}_seed{args.seed}"

MAX_BATCHES_PER_EPOCH = 10**10
EPOCHS = args.epochs
# args.batch_size = 128
# args.lr = 2e-4

main(args, LOG_DIR, EPOCHS, MAX_BATCHES_PER_EPOCH)
