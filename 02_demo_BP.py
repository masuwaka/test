import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions.normal import Normal

import factor_graph as fg

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------
# 1.  toy データ : 目的 f(x)=sin(6x)、制約 c(x)=0.5-(x-0.7)^2 (<0 可)
# --------------------------------------------------------------------------
XX = torch.linspace(0, 1, 100).unsqueeze(-1)
ff = torch.sin(6 * XX)
cc = -0.05 + (XX - 0.3) ** 2
plt.plot(XX, ff, c="blue")
plt.plot(XX[np.where(cc < 0)], ff[np.where(cc < 0)], c="red", lw=4)
plt.twinx()
plt.plot(XX, cc, c="black", ls=":")
plt.savefig("02_fc.png", dpi=300)

X = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
f = torch.sin(6 * X) + 0.05 * torch.randn_like(X)
c = -0.05 + (X - 0.3) ** 2 + 0.05 * torch.randn_like(X)

gp_f = SingleTaskGP(X, f)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_f.likelihood, gp_f))
gp_c = SingleTaskGP(X, c)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_c.likelihood, gp_c))
f_best = f.min()

# variable & weight
g = fg.FactorGraph()
g.variables["x_1"] = fg.Variable("x_1", (0, 1))
g.weight_variables["w_1_1"] = fg.WeightVariable("w_1_1")
g.factors.append(fg.WeightFactor("Omega_1_1", "w_1_1"))

# toy GP / EI & Φ 因子を追加（前デモと同じ手順）
g.factors.append(fg.ObjectiveFactor("EI", gp_f, f_best, ["x_1"]))
g.factors.append(fg.ConstraintFactor("Phi", gp_c, ["x_1"]))

F_curve = g.run(max_iter=30)

plt.plot(F_curve)
plt.savefig("02_F_curve.png")
