import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.distributions.normal import Normal

import factor_graph as fg
from myvis import set_colormap

set_colormap()
torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------
# 1.  toy データ : 目的 f(x)=sin(6x)、制約 c(x)=0.5-(x-0.7)^2 (<0 可)
# --------------------------------------------------------------------------
XX = torch.linspace(0, 1, 100).unsqueeze(-1)
ff = torch.sin(6 * XX)
cc = -0.05 + (XX - 0.3) ** 2

plt.figure()
plt.plot(XX, ff, c="blue")
plt.plot(XX[np.where(cc < 0)], ff[np.where(cc < 0)], c="red", lw=4)
plt.ylabel("objective value", fontsize=16)
plt.xlabel("X", fontsize=16)
plt.twinx()
plt.plot(XX, cc, c="black", ls=":")
plt.ylabel("constraint value", fontsize=16)
plt.savefig("02_fc_0.png", dpi=300)

X = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
f = torch.sin(6 * X)
c = -0.05 + (X - 0.3) ** 2

plt.figure()
plt.plot(XX, ff, c="blue")
plt.plot(XX[np.where(cc < 0)], ff[np.where(cc < 0)], c="red", lw=4)
plt.plot(X, f, ls="", marker="o", ms=12, mec="white")
plt.ylabel("objective value", fontsize=16)
plt.xlabel("X", fontsize=16)
plt.twinx()
plt.plot(XX, cc, c="black", ls=":")
plt.ylabel("constraint value", fontsize=16)
plt.savefig("02_fc_1.png", dpi=300)

from botorch.utils.transforms import normalize, unnormalize

from optimizers import get_GP_model, initialize_model
from transforms import gaussian_copula_transform

input_bounds = torch.tensor([[0], [1]])
input_bounds_norm = torch.tensor([[0], [1]])
X_raw = X
X_norm = normalize(X_raw, input_bounds)
y_raw = f
yvar = torch.tensor(0.0)

X_norm = normalize(X_raw, input_bounds)
mll, model = initialize_model(X_norm, y_raw, c, yvar, None)
fit_gpytorch_mll(mll)


is_feasible = (c <= 0).all(dim=-1)
# y_trans = gaussian_copula_transform(y_raw)
y_trans = y_raw
best_f = y_trans[is_feasible].min() if is_feasible.any() else y_trans.min()

g = fg.FactorGraph(model_list=model, f_best=best_f)
F_curve = g.run()
print(F_curve)
# g.variables["x_1"] = fg.Variable("x_1", (0, 1))
# g.weight_variables["w_1_1"] = fg.WeightVariable("w_1_1")
# g.factors.append(fg.WeightFactor("Omega_1_1", "w_1_1"))

# # toy GP / EI & Φ 因子を追加（前デモと同じ手順）
# g.factors.append(fg.ObjectiveFactor("EI", gp_f, f_best, ["x_1"]))
# g.factors.append(fg.ConstraintFactor("Phi", gp_c, ["x_1"]))

# F_curve = g.run(max_iter=30)

# plt.plot(F_curve)
# plt.savefig("02_F_curve.png")
