# 3‑phase BP with Gamma‑weight messages
# -----------------------------------------------------------------------------
#  Features
#    • Variable  : Gaussian belief (μ,σ)
#    • Weight    : Gamma belief (α,β) – precision‑scale influence
#    • StructurePrior Ω : Gamma(α0,β0) factor
#    • ObjectiveEI / ConstraintPhi send messages to BOTH x (Gaussian) & w (Gamma)
#    • Bethe Free‑Energy  =  E[‑log factor] + λ_x H[x] + λ_ω H[w] + KL(w‖prior)
# -----------------------------------------------------------------------------

import math
import re
from typing import Dict, List, Tuple, Union

import torch
from botorch.models import SingleTaskGP
from torch import Tensor, tensor
from torch.distributions import Normal


# Utility functions -----------------------------------------------------------
def gaussian_entropy(s2: Tensor) -> Tensor:
    """Entropy of the Gaussian variable with variance of s2.

    Args:
        s2 (Tensor): Variance of a Gaussian variable.

    Returns:
        Tensor: Entropy.
    """

    return 0.5 * torch.log(2 * math.pi * math.e * s2)


def gamma_entropy(a: Tensor, b: Tensor) -> Tensor:
    """Entropy of the Gamma variable with variance of s2.

    Args:
        a (Tensor): Shape paramter of a Gamma variable.
        b (Tensor): Rate paramter of a Gamma variable.

    Returns:
        Tensor: Entropy.
    """

    return a - torch.log(b) + torch.lgamma(a) + (1 - a) * torch.digamma(a)


def mus2_to_e1e2(mu: Tensor, s2: Tensor) -> Tuple[Tensor, Tensor]:
    """Transform Gaussian parameters (mu, s2) to natural parameters (eta1, eta2).

    Args:
        mu (Tensor): Mean of a Gaussian variable.
        s2 (Tensor): Variance of a Gaussian variable.

    Returns:
        Tuple[Tensor, Tensor]: Natural parameters (eta1, eta2).
    """

    eta1 = mu / s2
    eta2 = -1 / (2 * s2)

    return (eta1, eta2)


def e1e2_to_mus2(eta1: Tensor, eta2: Tensor) -> Tuple[Tensor, Tensor]:
    """Transform natural parameters (eta1, eta2) to Gaussian parameters (mu, s2).

    Args:
        eta1 (Tensor): Natural parameter 1 (coefficient of x).
        eta2 (Tensor): Natural parameter 2 (coefficient of x^2).

    Returns:
        Tuple[Tensor, Tensor]: Gaussian parameters (mu, s2)
    """
    mu = -eta1 / (2 * eta2)
    s2 = -1 / (2 * eta2)

    return mu, s2


def ab_to_e1e2(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """Transform Gamma parameters (a, b) to natural parameters (eta1, eta2).

    Args:
        a (Tensor): Shape parameter of a Gamma variable.
        b (Tensor): Rate parameter of a Gamma variable.

    Returns:
        Tuple[Tensor, Tensor]: Natural parameters (eta1, eta2)
    """

    eta1 = a - 1
    eta2 = -b

    return (eta1, eta2)


def e1e2_to_ab(eta1: Tensor, eta2: Tensor) -> Tuple[Tensor, Tensor]:
    """Transform natural parameters (eta1, eta2) to Gamma parameters (a, b).

    Args:
        eta1 (Tensor): Natural parameter 1 (coefficient of log(x)).
        eta2 (Tensor): Natural parameter 2 (coefficient of x).

    Returns:
        Tuple[Tensor, Tensor]: Gamma parameters (a, b).
    """

    a = eta1 + 1
    b = -eta2

    return (a, b)


# Variable (Gaussian) ---------------------------------------------------------
class Variable:
    def __init__(self, name: str, domain: Tuple[float, float], prior_mu: float = 0.5, prior_s2: float = 0.5):
        self.name = name
        self.domain = domain
        self.prior_mu, self.prior_s2 = tensor(prior_mu), tensor(prior_s2)
        self.mu, self.s2 = tensor(prior_mu), tensor(prior_s2)
        self.received_messages = {}

    # External message (excluding self message)
    def get_message(self, to_name: str) -> Tuple[Tensor, Tensor]:
        eta1, eta2 = tensor(0.0), tensor(0.0)
        for name, (mu, s2) in self.received_messages.items():
            if name == to_name:
                continue
            e1, e2 = mus2_to_e1e2(mu, s2)
            eta1 += e1
            eta2 += e2

        if eta2 == 0.0:
            return (self.prior_mu, self.prior_s2)

        mu, s2 = e1e2_to_mus2(eta1, eta2)
        return (mu, s2)

    # Posterior belief
    def update_belief(self, lambda_x: float):
        eta1, eta2 = tensor(0.0), tensor(-0.5 * lambda_x)  # lamda is prior precision (1/s2)
        for name, (mu, s2) in self.received_messages.items():
            e1, e2 = mus2_to_e1e2(mu, s2)
            eta1 += e1
            eta2 += e2

        if eta2 == 0.0:
            return

        self.mu, self.s2 = e1e2_to_mus2(eta1, eta2)

    # Entropy of the variable
    def entropy(self) -> Tensor:
        return gaussian_entropy(self.s2)


# Weight variable (Gamma) -----------------------------------------------------
class WeightVariable:
    def __init__(self, name: str, prior_a: float = 2.0, prior_b: float = 2.0):
        self.name = name
        self.prior_a, self.prior_b = tensor(prior_a), tensor(prior_b)
        self.a, self.b = tensor(prior_a), tensor(prior_b)
        self.received_messages = {}

    # Mean function
    def mu(self) -> Tensor:
        return self.a / self.b

    # External message (excluding self message)
    def get_message(self, to_name: str) -> Tuple[Tensor, Tensor]:
        eta1, eta2 = tensor(0.0), tensor(0.0)
        for name, (a, b) in self.received_messages.items():
            if name == to_name:
                continue
            e1, e2 = ab_to_e1e2(a, b)
            eta1 += e1
            eta2 += e2

        if eta2 == 0.0:
            return (self.prior_a, self.prior_b)

        a, b = e1e2_to_ab(eta1, eta2)
        return (a, b)

    # Posterior belief
    def update_belief(self, lambda_w: float):
        # start from prior natural params
        eta1, eta2 = ab_to_e1e2(self.prior_a, self.prior_b)
        for _, (a, b) in self.received_messages.items():
            d1, d2 = ab_to_e1e2(a, b)
            eta1 += d1
            eta2 += d2
        # λ_w acts on rate ⇒ η₂ -= λ_w
        eta2 -= lambda_w
        if eta2 == 0.0:
            return
        self.a, self.b = e1e2_to_ab(eta1, eta2)

    # Entripy of the variable
    def entropy(self) -> Tensor:
        return gamma_entropy(self.a, self.b)

    # KLD from prior
    def KLD_prior(self) -> Tensor:
        prior_a, prior_b = self.prior_a, self.prior_b
        a, b = self.a, self.b
        return (
            (a - prior_a) * torch.digamma(a)
            - torch.lgamma(a)
            + torch.lgamma(prior_a)
            + prior_a * torch.log(b / prior_b)
            + (prior_b - b) * a / b
        )


# Weight (prior) factor (Gamma) -----------------------------------------------
class WeightFactor:
    def __init__(self, name: str, to_name: str, prior_a: float = 2.0, prior_b: float = 2.0):
        self.name = name
        self.to_name = to_name
        self.prior_a, self.prior_b = tensor(prior_a), tensor(prior_b)

    def send_message(self, variables: Dict[str, Variable], weight_variables: Dict[str, WeightVariable]):
        weight_variables[self.to_name].received_messages[self.name] = (self.prior_a, self.prior_b)

    def expected_log_score(self, vars) -> Tensor:
        return tensor(0.0)


# Objective EI factor ---------------------------------------------------------
class ObjectiveFactor:
    def __init__(
        self, name: str, gp_model: SingleTaskGP, f_best: Tensor, variable_names: List[str], kappa_w: float = 1e-3
    ):
        self.name = name
        self.gp_model = gp_model
        self.f_best = f_best
        self.variable_names = variable_names
        self.kappa_w = tensor(kappa_w)

    def _log_ei(self, x: Tensor):
        posterior = self.gp_model.posterior(x)
        mu = posterior.mean.squeeze(-1)
        s2 = posterior.variance.squeeze(-1)
        z = (self.f_best - mu) / s2.sqrt()
        ei = s2.sqrt() * (z * Normal(0, 1).cdf(z) + torch.exp(Normal(0, 1).log_prob(z)))
        return torch.log(ei + 1e-9)

    def send_message(self, variables: Dict[str, Variable], weight_variables: Dict[str, WeightVariable]):
        for vname in self.variable_names:
            mu, s2 = variables[vname].get_message(self.name)
            omega = torch.tensor([mu], requires_grad=True)
            opt = torch.optim.LBFGS([omega], max_iter=10)
            opt.step(lambda: (-self._log_ei(omega.view(1, 1))).backward() or self._log_ei(omega))
            with torch.enable_grad():
                l = self._log_ei(omega.view(1, 1))
                g1 = torch.autograd.grad(l, omega, create_graph=True)[0]
                g2 = torch.autograd.grad(g1, omega)[0]
            s2_hat = 1 / (-g2 + self.kappa_w)
            variables[vname].received_messages[self.name] = (omega.detach()[0], s2_hat.detach()[0])

            # weight message: use curvature to set (α̂,β̂)
            a_hat = tensor(2.0)
            b_hat = tensor(2.0) + max(0.0, -g2.item())

            # broadcast to all connected weight variables (here assume single vname="x_{d}" and wname="w_{d}_{k}")
            vnum = vname.split("_")[1]
            for wname, weight_variable in weight_variables.items():
                wnum = wname.split("_")[1]
                if vnum == wnum:
                    weight_variable.received_messages[self.name] = (a_hat, b_hat)

    # expected log score for Bethe F (single-point approx)
    def expected_log_score(self, variables: Dict[str, Variable]) -> Tensor:
        x = tensor([[variables[self.variable_names[0]].mu]])
        return -self._log_ei(x).detach()


# Constraint factor -----------------------------------------------------------
class ConstraintFactor:
    def __init__(self, name: str, gp_model: SingleTaskGP, variable_names: List[str], kappa_w: float = 1e-3):
        self.name = name
        self.gp_model = gp_model
        self.variable_names = variable_names
        self.kappa_w = tensor(kappa_w)

    def _log_phi(self, x: Tensor):
        posterior = self.gp_model.posterior(x)
        mu = posterior.mean.squeeze(-1)
        s2 = posterior.variance.squeeze(-1)
        z = -mu / s2.sqrt()
        return torch.log(Normal(0, 1).cdf(z) + 1e-9)

    def send_message(self, variables: Dict[str, Variable], weight_variables: Dict[str, WeightVariable]):
        for vname in self.variable_names:
            mu, s2 = variables[vname].get_message(self.name)
            omega = torch.tensor([mu], requires_grad=True)
            opt = torch.optim.LBFGS([omega], max_iter=10)
            opt.step(lambda: (-self._log_phi(omega.view(1, 1))).backward() or self._log_phi(omega))
            with torch.enable_grad():
                l = self._log_phi(omega.view(1, 1))
                g1 = torch.autograd.grad(l, omega, create_graph=True)[0]
                g2 = torch.autograd.grad(g1, omega)[0]
            s2_hat = 1 / (-g2 + self.kappa_w)
            variables[vname].received_messages[self.name] = (omega.detach()[0], s2_hat.detach()[0])

            a_hat = tensor(2.0)
            b_hat = tensor(2.0) + max(0.0, -g2.item())

            vnum = vname.split("_")[1]
            for wname, weight_variable in weight_variables.items():
                wnum = wname.split("_")[1]
                if vnum == wnum:
                    weight_variable.received_messages[self.name] = (a_hat, b_hat)

    def expected_log_score(self, variables: Dict[str, Variable]) -> Tensor:
        x = tensor([[variables[self.variable_names[0]].mu]])
        return -self._log_phi(x).detach()


# Factor Graph ----------------------------------------------------------------
class FactorGraph:
    def __init__(self):
        self.variables: Dict[str, Variable] = {}
        self.weight_variables: Dict[str, WeightVariable] = {}
        self.factors: List[Union[WeightFactor, ObjectiveFactor, ConstraintFactor]] = []

    def run(self, max_iter: int = 20, delta: float = 0.05):
        lamda_base = math.sqrt(math.log(2 / delta) / (2 * max(1, len(self.variables))))
        F = []
        for i in range(max_iter):
            # 1) variable update
            for variable in self.variables.values():
                # use only weights connected to this variable (same index)
                vid = re.findall(r"\d+", variable.name)[0]
                degree = tensor(
                    sum([self.weight_variables[w].mu() for w in self.weight_variables if re.search(rf"_{vid}$", w)])
                )
                variable.update_belief(lambda_x=lamda_base * max(0.0, 1.0 - degree.item()))

            # 2) weight update
            for weight_variable in self.weight_variables.values():
                weight_variable.update_belief(lambda_w=lamda_base)

            # 3) factor messages
            for factor in self.factors:
                factor.send_message(self.variables, self.weight_variables)

            # free‑energy monitor
            F.append(self.free_energy(lamda_base))
            if i > 4 and abs(F[-2] - F[-1]) < 1e-6:
                break
        return F

    def free_energy(self, lamda_base: float):
        Hx = sum(v.entropy() for v in self.variables.values())
        Hw = sum(w.entropy() for w in self.weight_variables.values())
        kl = sum(w.KLD_prior() for w in self.weight_variables.values())
        energy = sum(f.expected_log_score(self.variables) for f in self.factors)
        return energy + lamda_base * Hx + lamda_base * Hw + kl
