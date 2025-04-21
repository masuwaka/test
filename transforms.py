from typing import Any

import numpy as np
import torch
from botorch.models.transforms.outcome import OutcomeTransform
from scipy.stats import norm
from torch import Tensor


def gaussian_copula_transform(y: torch.Tensor):
    y_trans = []
    for d in range(y.shape[1]):
        col = y[:, d]

        r = torch.argsort(torch.argsort(col))
        q = (r.double() + 0.5) / y.shape[0]

        y_trans.append(np.sqrt(2) * torch.erfinv(2 * q - 1).unsqueeze(-1))

    y_trans = torch.cat(y_trans, dim=1)

    return y_trans


def bilog_transform(c: torch.Tensor):
    c_trans = []
    for d in range(c.shape[1]):
        col = c[:, d]

        c_trans.append((torch.sign(col) * torch.log(1 + torch.abs(col))).unsqueeze(-1))

    c_trans = torch.cat(c_trans, dim=1)

    return c_trans
