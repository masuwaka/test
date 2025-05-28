import math
from typing import Dict, List, Tuple

import torch
from botorch.models import SingleTaskGP
from torch import Tensor
from torch.distributions import Normal


def gaussian_entropy(s2: Tensor) -> Tensor:
    """Entropy of the Gaussian variable with variance of s2.

    Args:
        s2 (Tensor): Variance of a Gaussian variable.

    Returns:
        Tensor: Entropy.
    """

    return 0.5 * torch.log(2 * math.pi * math.e * s2)


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


class Variable:
    def __init__(self, name: str, domain: Tuple[float, float], prior_mu: float = 0.0, prior_s2: float = 1.0):
        self.name = name
        self.domain = domain
        self.prior_mu = torch.tensor(prior_mu)
        self.prior_s2 = torch.tensor(prior_s2)
        self.belief_mu = torch.tensor(prior_mu)
        self.belief_s2 = torch.tensor(prior_s2)
        self.received_messages = {}

    def update_belief(self, incoming_messages: List[Tuple[str, Tuple[Tensor, Tensor]]], lambda_x: float = 0.0):
        self.received_messages = {name: message for name, message in incoming_messages}
        if incoming_messages == []:
            self.belief_mu = self.prior_mu.clone()
            self.belief_s2 = self.prior_s2.clone()
            return

        eta1, eta2 = torch.tensor(0.0), torch.tensor(0.0)
        for name, (mu, s2) in incoming_messages:
            e1, e2 = mus2_to_e1e2(mu, s2)
            eta1 += e1
            eta2 += e2
        eta2 -= lambda_x
        self.belief_mu, self.belief_s2 = e1e2_to_mus2(eta1, eta2)

    def get_message(self, to_name: str, incoming_messages: List[Tuple[str, Tuple[Tensor, Tensor]]]):
        filtered_messages = [(mu, s2) for name, (mu, s2) in incoming_messages if name != to_name]
        if filtered_messages == []:
            return (self.prior_mu, self.prior_s2)

        eta1, eta2 = torch.tensor(0.0), torch.tensor(0.0)
        for mu, s2 in filtered_messages:
            e1, e2 = mus2_to_e1e2(mu, s2)
            eta1 += e1
            eta2 += e2
        mu, s2 = e1e2_to_mus2(eta1, eta2)
        return (mu, s2)

    def entropy(self):
        return gaussian_entropy(self.belief_s2)

    def effective_degree(self, structure_variables):
        return sum([sv.belief_mu for sv in structure_variables])


class StructureVariable:
    def __init__(self, name: str, prior_mu: float = 0.0, prior_s2: float = 1.0):
        self.name = name
        self.prior_mu = torch.tensor(prior_mu)
        self.prior_s2 = torch.tensor(prior_s2)
        self.belief_mu = torch.tensor(prior_mu)
        self.belief_s2 = torch.tensor(prior_s2)
        self.received_messages = {}

    def update_belief(self, incoming_messages: List[tuple[str, Tuple[Tensor, Tensor]]], lambda_omega: float = 0.0):
        self.received_messages = {name: message for name, message in incoming_messages}
        if incoming_messages == []:
            self.belief_mu = self.prior_mu.clone()
            self.belief_s2 = self.prior_s2.clone()
            return

        eta1, eta2 = torch.tensor(0.0), torch.tensor(0.0)
        for name, (mu, s2) in incoming_messages:
            e1, e2 = mus2_to_e1e2(mu, s2)
            eta1 += e1
            eta2 += e2
        eta2 -= lambda_omega
        self.belief_mu, self.belief_s2 = e1e2_to_mus2(eta1, eta2)

    def get_message(self, to_name: str, incoming_messages: List[tuple[str, Tuple[Tensor, Tensor]]]):
        filtered_messages = [(mu, s2) for name, (mu, s2) in incoming_messages if name != to_name]
        if filtered_messages == []:
            return (self.prior_mu, self.prior_s2)

        eta1, eta2 = torch.tensor(0), torch.tensor(0)
        for mu, s2 in filtered_messages:
            e1, e2 = mus2_to_e1e2(mu, s2)
            eta1 += e1
            eta2 += e2
        mu, s2 = e1e2_to_mus2(eta1, eta2)

    def entropy(self):
        return gaussian_entropy(self.belief_s2)


class StructureFactor:
    def __init__(self, name: str, to_name: str, mu0: float = 0.0, s20: float = 1.0):
        self.name = name
        self.to_name = to_name
        self.mu0 = torch.tensor(mu0)
        self.s20 = torch.tensor(s20)

    def update_messages(self, variables: Dict[str, Variable], structure_variables: Dict[str, StructureVariable]):
        structure_variables[self.to_name].received_messages[self.name] = (self.mu0, self.s20)


class ObjectiveFactor:
    def __init__(self, name: str, connected_variable_names: List[str], gp_f: SingleTaskGP, f_best: Tensor):
        self.name = name
        self.connected_variable_names = connected_variable_names
        self.gp_f = gp_f
        self.f_best = f_best

    def update_messages(self, variables: Dict[str, Variable], structure_variables: Dict[str, StructureVariable]):
        for vn in self.connected_variable_names:
            incoming_messages = [
                (name, message) for name, message in variables[vn].received_messages.items() if name != self.name
            ]
            mu_ext, s2_ext = (
                variables[vn].get_message(self.name, incoming_messages)
                if incoming_messages != []
                else (variables[vn].prior_mu, variables[vn].prior_s2)
            )
            omega = torch.tensor([mu_ext], requires_grad=True)

            def log_ei():
                post = self.gp_f.posterior(omega.view(1, 1))
                mu, s2 = post.mean.squeeze(-1), post.variance.squeeze(-1)
                z = (self.f_best - mu) / s2.sqrt()
                ei = s2.sqrt() * (z * Normal(0, 1).cdf(z) + torch.exp(Normal(0, 1).log_prob(z)))
                return torch.log(ei + 1e-9)

            opt = torch.optim.LBFGS([omega], max_iter=10)
            opt.step(lambda: (-log_ei()).backward() or log_ei())
            with torch.enable_grad():
                l = log_ei()
                g1 = torch.autograd.grad(l, omega, create_graph=True)[0]
                g2 = torch.autograd.grad(g1, omega)[0]
            s2 = 1 / (-g2 + 1e-6)
            variables[vn].received_messages[self.name] = (omega.detach()[0], s2.detach()[0])


class ConstraintFactor:
    def __init__(self, name: str, connected_variable_names: List[str], gp_c: SingleTaskGP):
        self.name = name
        self.connected_variable_names = connected_variable_names
        self.gp_c = gp_c

    def update_messages(self, variables, structure_variables):
        for vn in self.connected_variable_names:
            incoming_messages = [
                (name, message) for name, message in variables[vn].received_messages.items() if name != self.name
            ]
            mu_ext, s2_ext = (
                variables[vn].get_message(self.name, incoming_messages)
                if incoming_messages != []
                else (variables[vn].prior_mu, variables[vn].prior_s2)
            )
            omega = torch.tensor([mu_ext], requires_grad=True)

            def log_phi():
                post = self.gp_c.posterior(omega.view(1, 1))
                mu, s2 = post.mean.squeeze(-1), post.variance.squeeze(-1)
                z = -mu / s2.sqrt()
                phi = Normal(0, 1).cdf(z)
                return torch.log(phi + 1e-9)

            opt = torch.optim.LBFGS([omega], max_iter=10)
            opt.step(lambda: (-log_phi()).backward() or log_phi())
            with torch.enable_grad():
                l = log_phi()
                g1 = torch.autograd.grad(l, omega, create_graph=True)[0]
                g2 = torch.autograd.grad(g1, omega)[0]
            s2 = 1 / (-g2 + 1e-6)
            variables[vn].received_messages[self.name] = (omega.detach()[0], s2.detach()[0])


class FactorGraph:
    def __init__(self):
        self.variables = {}
        self.structure_variables = {}
        self.structure_map = {}
        self.structure_factors = []
        self.constraint_factors = []
        self.objective_factors = []

    def __call__(self, max_iters: int = 10, delta: float = 0.05):
        # λ from PAC‑Bayes bound (McAllester ‘99): $\sqrt{\frac{\ln(2/\delta)}{2N} }$
        N = max(1, len(self.variables))
        lambda_base = math.sqrt(math.log(2 / delta) / (2 * N))
        lambda_x_base = lambda_base
        lambda_w = lambda_base

        for _ in range(max_iters):
            # Update Variables
            for vname, variable in self.variables.items():
                w_refs = [self.structure_variables[o] for o in self.structure_map.get(vname, [])]
                degree = variable.effective_degree(w_refs)
                lambda_x = lambda_x_base * max(0.0, 1.0 - degree)
                incoming_messages = [(name, message) for name, message in variable.received_messages.items()]
                variable.update_belief(incoming_messages, lambda_x)

            # Update StructureVariables
            for structure_variable in self.structure_variables.values():
                incoming_messages = [(name, message) for name, message in structure_variable.received_messages.items()]
                structure_variable.update_belief(incoming_messages, lambda_w)

            # Send messages
            for factor in (*self.structure_factors, *self.constraint_factors, *self.objective_factors):
                factor.update_messages(self.variables, self.structure_variables)
            for variable in self.variables.values():
                for to in variable.received_messages:
                    variable.received_messages[to] = variable.get_message(to, list(variable.received_messages.items()))
            for structure_variable in self.structure_variables.values():
                for to in structure_variable.received_messages:
                    structure_variable.received_messages[to] = structure_variable.get_message(
                        to, list(structure_variable.received_messages.items())
                    )

    def compute_bethe_free_energy(self, delta: float = 0.05):
        N = max(1, len(self.variables))
        lambda_base = math.sqrt(math.log(2 / delta) / (2 * N))
        F = torch.tensor(0.0)
        for vname, variable in self.variables.items():
            w_refs = [self.structure_variables[o] for o in self.structure_map.get(vname, [])]
            degree = variable.effective_degree(w_refs)
            lambda_x = lambda_base * max(0.0, 1.0 - degree)
            F += lambda_x * variable.entropy()
        for structure_variable in self.structure_variables.values():
            F += lambda_base * structure_variable.entropy()
        for structure_factor in self.structure_factors:
            structure_variable = self.structure_variables[structure_factor.to_name]
            mu, s2 = structure_variable.belief_mu, structure_variable.belief_s2
            mu0, s20 = structure_factor.mu0, structure_factor.s20
            KLD = ((mu - mu0) ** 2 + s2) / (2 * s20) - 0.5 * torch.log(s2 / s20)
            F += KLD

        return F.item()
