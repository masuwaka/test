import matplotlib.pyplot as plt
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
X = torch.linspace(0.05, 0.95, 10).unsqueeze(-1)
f = torch.sin(6 * X) + 0.05 * torch.randn_like(X)
c = 0.5 - (X - 0.7) ** 2 + 0.05 * torch.randn_like(X)

gp_f = SingleTaskGP(X, f)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_f.likelihood, gp_f))
gp_c = SingleTaskGP(X, c)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_c.likelihood, gp_c))
f_best = f.min()


def feas_gp(x):
    post = gp_c.posterior(x)
    mu, sig = post.mean.squeeze(-1), post.variance.sqrt().squeeze(-1)
    return mu, sig


# --------------------------------------------------------------------------
# 2.  Factor Graph 構築
# --------------------------------------------------------------------------
g = fg.FactorGraph()

g.variables["x1"] = fg.Variable("x1", domain=(0.0, 1.0))
g.structure_variables["w11"] = fg.StructureVariable("w11")  # Gamma belief
g.structure_map["x1"] = ["w11"]

# Gamma prior factor  α0=β0=2  (平均=1, 疎性 moderate)
g.structure_factors.append(fg.StructureFactor("Omega11", "w11", mu0=0.0, s20=1.0))

g.objective_factors.append(fg.ObjectiveFactor("EI", ["x1"], gp_f=gp_f, f_best=f_best))
g.constraint_factors.append(fg.ConstraintFactor("Phi", ["x1"], gp_c=gp_c))

# --------------------------------------------------------------------------
# 3.  BP 反復しながら Bethe F を記録
# --------------------------------------------------------------------------
max_iters, delta = 30, 0.05
F_vals = []

for it in range(max_iters):
    # ---- BP 1 ステップ (グラフは __call__ 内で 1 周回す設計に変更も可) ----
    g(max_iters=1, delta=delta)  # 1 反復だけ回す
    F_now = g.compute_bethe_free_energy(delta=delta)
    F_vals.append(F_now)

    # 収束判定
    if it > 1 and abs(F_vals[-2] - F_vals[-1]) < 1e-4:
        print(f"Converged at iter {it}")
        break

# --------------------------------------------------------------------------
# 4.  結果表示
# --------------------------------------------------------------------------
print(f"x1 belief  : {g.variables['x1'].belief_mu.item():.4f} " f"+/- {g.variables['x1'].belief_s2.sqrt().item():.4f}")
print(
    f"w11 belief : {g.structure_variables['w11'].belief_mu.item():.4f} "
    f"+/- {g.structure_variables['w11'].belief_s2.sqrt().item():.4f}"
)

plt.figure(figsize=(5, 3))
plt.plot(F_vals, marker="o")
plt.xlabel("BP iteration")
plt.ylabel("Bethe free-energy  F")
plt.title("Bethe F convergence")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("01_bethe.png")
