import random
from argparse import ArgumentParser, Namespace

import numpy as np
import pandas as pd
import torch

AVAIL_ACQ = ["Sobol", "CEI"]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-sd", "--seed", type=int, required=True, help="Random seed.")
    parser.add_argument("-fc", "--func", type=str, default="Toy2D", required=False, help="Function name.")
    parser.add_argument("-nstd", "--noise_std", type=float, default=0.0, required=False, help="Std of noise.")
    parser.add_argument("-dev", "--device", type=str, default="cpu", required=False, help="Device.")
    parser.add_argument("-l", "--log", type=str, required=True, help="Parent of log directory.")
    parser.add_argument("-acq", "--acq", type=str, required=True, help="Aquisition function.")
    parser.add_argument("-nwarm", "--n_warmup", type=int, default=10, required=False, help="Warm-up samples.")
    parser.add_argument("-ns", "--n_samples", type=int, default=100, required=False, help="# trials.")
    parser.add_argument("-bs", "--batch_size", type=int, default=1, required=False, help="Batch size")

    args = parser.parse_args()
    assert args.acq in AVAIL_ACQ, f'Acquisition function must be in {AVAIL_ACQ}. f"{args.acq}" is not supported.'
    assert args.noise_std >= 0, f"Noise(std) must be >=0."
    assert args.device in ["cuda", "cpu"], f'Device must be in ["cuda", "cpu"]. "{args.device}" is not supported.'
    if args.device == "cuda":
        assert torch.cuda.is_available(), f"CUDA is not available in this machine. Use cpu."
    assert args.n_warmup >= 0, f"Warm-up samples must be >=0."
    assert args.n_samples >= args.n_warmup, f"# trials must be >= warm-up samples."
    assert args.batch_size >= 1, f"Batch size must be > 0."

    return args


def set_random(seed: int):
    np.random.seed(seed)
    random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.cuda.manual_seed(seed + 3)


def make_result_dataframe(X, y, by, feas, c, y_noise, byn, feas_noise, c_noise, sample, batch, t, rand=False):
    res = []
    res.append(pd.DataFrame(X.detach().cpu().numpy(), columns=[f"X{dim+1}" for dim in range(X.shape[-1])]))
    res.append(pd.DataFrame(-y.detach().cpu().numpy(), columns=["y"]))
    res.append(pd.DataFrame(-np.array(by), columns=["y_best"]))
    res.append(pd.DataFrame(feas.detach().cpu().numpy(), columns=["feas"]))
    res.append(pd.DataFrame(c.detach().cpu().numpy(), columns=[f"c{cn+1}" for cn in range(c.shape[-1])]))
    res.append(pd.DataFrame(-y_noise.detach().cpu().numpy(), columns=["y_n"]))
    res.append(pd.DataFrame(-np.array(byn), columns=["y_n_best"]))
    res.append(pd.DataFrame(feas_noise.detach().cpu().numpy(), columns=["feas_n"]))
    res.append(pd.DataFrame(c_noise.detach().cpu().numpy(), columns=[f"c_n{cn+1}" for cn in range(c_noise.shape[-1])]))
    res = pd.concat(res, axis=1)
    res.insert(0, "Sample", range(sample, sample + X.shape[0]))
    res.insert(1, "Batch", batch)
    res.insert(2, "InBatchSample", range(1, X.shape[0] + 1))
    res.insert(3, "Random", rand)
    res["TimeSec"] = t

    return res


def get_best_y(y, y_noise, feas, feas_noise, best_y, best_y_noise):
    by, byn = [], []
    for i in range(y.shape[0]):
        yd = y.detach().cpu().numpy()
        best_y = yd[i, 0] if (best_y < yd[i, 0] and feas[i, 0]) else best_y
        yd_noise = y_noise.detach().cpu().numpy()
        best_y_noise = yd_noise[i, 0] if (best_y_noise < yd_noise[i, 0] and feas_noise[i, 0]) else best_y_noise
        by.append(best_y)
        byn.append(best_y_noise)

    return best_y, best_y_noise, by, byn
