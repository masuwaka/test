import math

import torch

import factor_graph as fg  # canvas のコードを import

torch.set_default_dtype(torch.float64)

# 1. ちいさな toy GP を作る --------------------------------------------------
X = torch.tensor([[0.1], [0.4], [0.8]])
f = torch.sin(6 * X) + 0.05 * torch.randn_like(X)  # objective
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood

gp_f = SingleTaskGP(X, f)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_f.likelihood, gp_f))
f_best = f.min()

# 2. 制約 GP (Φ＝P[c(x)<0]) を作る -------------------------------------------
c_true = lambda x: 0.5 - (x - 0.7) ** 2  # 円制約もどき
c_obs = c_true(X) + 0.05 * torch.randn_like(X)
gp_c = SingleTaskGP(X, c_obs)
fit_gpytorch_mll(ExactMarginalLogLikelihood(gp_c.likelihood, gp_c))


# 3. FactorGraph を組む -------------------------------------------------------
g = fg.FactorGraph()

# variable x1 ∈ [0,1]
g.variables["x1"] = fg.Variable("x1", domain=(0.0, 1.0))

# structure weight w11  (Gamma prior→ StructureFactor)
g.structure_variables["w11"] = fg.StructureVariable("w11")
g.structure_map["x1"] = ["w11"]
g.structure_factors.append(fg.StructureFactor("Omega11", "w11", mu0=0.0, s20=1.0))

# EI 目的因子
g.objective_factors.append(fg.ObjectiveFactor("EI", ["x1"], gp_f=gp_f, f_best=f_best))

# 可行確率 Φ 因子
g.constraint_factors.append(fg.ConstraintFactor("Phi", ["x1"], gp_c=gp_c))

# 4. BP を 10 反復 ------------------------------------------------------------
g(max_iters=10)  # λ は内部で自動 (PAC-Bayes)

# 5. 結果表示 ---------------------------------------------------------------
print(f"x1 belief  : {g.variables['x1'].belief_mu.item():.3f} " f"+/- {g.variables['x1'].belief_s2.sqrt().item():.3f}")
print(
    f"w11 belief : {g.structure_variables['w11'].belief_mu.item():.3f} "
    f"+/- {g.structure_variables['w11'].belief_s2.sqrt().item():.3f}"
)
print("Bethe F    :", g.compute_bethe_free_energy())
