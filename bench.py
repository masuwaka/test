from pathlib import Path
from time import time

import pandas as pd
import torch

import functions
from optimizers import get_optim_dict
from utils import get_best_y, make_result_dataframe, parse_args, set_random

args = parse_args()
optim_dict = get_optim_dict()
assert (
    args.acq in optim_dict.keys()
), f'Optimizer "{args.acq}" is missing in optimizers.py.\nAvailable functions are below.\n{optim_dict.keys()}'

func_dict = functions.get_func_dict()
assert (
    args.func in func_dict.keys()
), f'Function "{args.func}" is missing in functions.py.\nAvailable functions are below.\n{func_dict.keys()}'
set_random(args.seed)
p = Path(args.log) / args.func / args.acq / f"seed_{args.seed:04d}.csv"
p.parent.mkdir(exist_ok=True, parents=True)
result = []

optimizer = optim_dict[args.acq](
    func_dict[args.func],
    negate=True,
    bounds=None,
    noise_std=args.noise_std,
    device=args.device,
    dtype=torch.double,
)

best_y, best_y_noise = float("-inf"), float("-inf")

batch = 0
sample = 1

time_st = time()
X = optimizer.get_warmup_samples(args.n_warmup)
t = time() - time_st
y, feas, c, y_noise, feas_noise, c_noise = optimizer.observe(X)
best_y, best_y_noise, by, byn = get_best_y(y, y_noise, feas, feas_noise, best_y, best_y_noise)
result.append(make_result_dataframe(X, y, by, feas, c, y_noise, byn, feas_noise, c_noise, sample, batch, t, rand=True))
pd.concat(result).set_index("Sample").to_csv(p)
batch += 1
sample += args.n_warmup

n_batch = (args.n_samples - args.n_warmup - 1) // args.batch_size + 1
for batch in range(1, n_batch + 1):
    batch_size = (
        args.batch_size
        if sample + args.batch_size <= args.n_samples + 1
        else (args.n_samples - args.n_warmup) % args.batch_size
    )
    time_st = time()
    new_X = optimizer.optim_get_candidates(X, y_noise, c_noise, batch_size)
    t = time() - time_st
    new_y, new_feas, new_c, new_y_noise, new_feas_noise, new_c_noise = optimizer.observe(new_X)
    best_y, best_y_noise, by, byn = get_best_y(new_y, new_y_noise, new_feas, new_feas_noise, best_y, best_y_noise)
    result.append(
        make_result_dataframe(
            new_X,
            new_y,
            by,
            new_feas,
            new_c,
            new_y_noise,
            byn,
            new_feas_noise,
            new_c_noise,
            sample,
            batch,
            t,
        )
    )
    pd.concat(result).set_index("Sample").to_csv(p)
    sample += batch_size

    X = torch.cat([X, new_X], dim=0)
    y_noise = torch.concat([y_noise, new_y_noise], dim=0)
    c_noise = torch.concat([c_noise, new_c_noise], dim=0)
