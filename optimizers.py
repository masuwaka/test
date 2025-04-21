from abc import abstractmethod

import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from transforms import bilog_transform, gaussian_copula_transform


def get_optim_dict():
    return {
        name: obj
        for name, obj in globals().items()
        if isinstance(obj, type) and issubclass(obj, Optimizer) and obj is not Optimizer and obj.__module__ == __name__
    }


class Optimizer:
    def __init__(
        self,
        func,
        negate=True,
        bounds=None,
        noise_std=0,
        device="cuda",
        dtype=torch.double,
    ):
        self.func = func(noise_std=noise_std, negate=negate, bounds=bounds)
        self.device = device
        self.dtype = dtype
        self.noise_std = noise_std
        self.input_bounds = torch.tensor(
            [[b[0] for b in func._bounds], [b[1] for b in func._bounds]], dtype=dtype, device=device
        )
        self.input_bounds_norm = torch.tensor([[0] * func.dim, [1] * func.dim], dtype=dtype, device=device)
        self.sobol = SobolEngine(self.func.dim, scramble=True)

    def get_warmup_samples(self, n_warmup):
        candidates = unnormalize(self.sobol.draw(n_warmup, dtype=self.dtype).to(self.device), bounds=self.input_bounds)

        return candidates

    @abstractmethod
    def optim_get_candidates(self, X, y, c, batch_size):
        pass

    def observe(self, X):
        y, y_noise = self.func.evaluate_true(X)
        _cons = self.func.gs(X)
        c, c_noise = [], []
        for _c in _cons:
            c.append(_c[0])
            c_noise.append(_c[1])
        c = torch.cat(c, dim=-1)
        feas = torch.le(c, 0).all(dim=-1).unsqueeze(-1)
        c_noise = torch.concat(c_noise, dim=-1)
        feas_noise = torch.le(c_noise, 0).all(dim=-1).unsqueeze(-1)

        return y, feas, c, y_noise, feas_noise, c_noise


class Sobol(Optimizer):
    def __init__(
        self,
        func,
        negate=True,
        bounds=None,
        noise_std=0,
        device="cuda",
        dtype=torch.double,
    ):
        super().__init__(
            func,
            negate=negate,
            bounds=bounds,
            noise_std=noise_std,
            device=device,
            dtype=dtype,
        )

    def optim_get_candidates(self, X, y, c, batch_size):
        candidates = unnormalize(
            self.sobol.draw(batch_size, dtype=self.dtype).to(self.device), bounds=self.input_bounds
        )

        return candidates


def initialize_model(X, y, c, yvar, state_dict=None):
    y_trans = gaussian_copula_transform(y)

    model_y = SingleTaskGP(X, y_trans, yvar.expand_as(y_trans), input_transform=Normalize(d=X.shape[-1])).to(X)

    model_c = []
    for cn in range(c.shape[-1]):
        c_trans = bilog_transform(c[:, cn].unsqueeze(-1))
        model_c.append(
            SingleTaskGP(X, c_trans, yvar.expand_as(c_trans), input_transform=Normalize(d=X.shape[-1])).to(X)
        )

    model = ModelListGP(model_y, *model_c)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model


class CEI(Optimizer):
    def __init__(
        self,
        func,
        negate=True,
        bounds=None,
        noise_std=0,
        device="cuda",
        dtype=torch.double,
    ):
        super().__init__(
            func,
            negate=negate,
            bounds=bounds,
            noise_std=noise_std,
            device=device,
            dtype=dtype,
        )
        self.state_dict = None
        self.yvar = torch.tensor(self.noise_std**2, device=device, dtype=dtype)

    def optim_get_candidates(self, X, y, c, batch_size):
        mll, model = initialize_model(X, y, c, self.yvar, self.state_dict)

        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))

        objective = GenericMCObjective(objective=lambda Z, X: Z[..., 0])

        constraints = [lambda Z: Z[..., i + 1] for i in range(c.shape[-1])]

        if self.yvar <= 1e-8:
            y_trans = gaussian_copula_transform(y)
            ei = qLogExpectedImprovement(
                model=model,
                best_f=(y_trans * torch.le(c, 0).to(y_trans)).max(),
                sampler=sampler,
                objective=objective,
                constraints=constraints,
            )
        else:
            ei = qLogNoisyExpectedImprovement(
                model=model, X_baseline=X, sampler=sampler, objective=objective, constraints=constraints
            )

        candidates, _ = optimize_acqf(
            acq_function=ei,
            bounds=self.input_bounds,
            q=batch_size,
            num_restarts=20,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )

        self.state_dict = model.state_dict()

        return candidates
