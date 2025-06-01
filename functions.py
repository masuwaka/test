import math

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor


def get_func_dict():
    return {name: obj for name, obj in globals().items() if isinstance(obj, type) and obj.__module__ == __name__}


class TCS3D:
    """3D Tension-Compression String"""

    dim = 3
    num_cons = 4
    _bounds = [(0.05, 2), (0.25, 1.3), (2, 15)]
    _optimal_value = 0.012665285
    _optimizers = [(0.05174250340926, 0.35800478345599, 11.21390736278739)]

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        y = (X[:, 0] ** 2) * X[:, 1] * (X[:, 2] + 2)
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = 1 - ((X[:, 1] * 3) * X[:, 2]) / (71785 * (X[:, 0] ** 4))
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        g = (
            (4 * (X[:, 1] ** 2) - X[:, 0] * X[:, 1]) / (12566 * (X[:, 0] ** 3) * (X[:, 1] - X[:, 0]))
            + 1 / (5108 * (X[:, 0] ** 2))
            - 1
        )
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g3(self, X: Tensor):
        g = 1 - (140.45 * X[:, 0]) / (X[:, 2] * (X[:, 1] ** 2))
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g4(self, X: Tensor):
        g = (X[:, 0] + X[:, 1]) / 1.5 - 1
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return self.g1(X), self.g2(X), self.g3(X), self.g4(X)


class PVD4D:
    """4D Pressure Vessel Design
    X0 and X1 should be integer multiples of 0.0625"""

    dim = 4
    num_cons = 4
    _bounds = [(0, 10), (0, 10), (10, 50), (150, 200)]
    _optimal_value = 6059.946341
    _optimizers = [(0.8125, 0.4375, 42.097398, 176.654047)]
    _d = 0.0625

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        y = (
            0.06224 * X[:, 0] * X[:, 2] * X[:, 3]
            + 1.7781 * X[:, 1] * (X[:, 2] ** 2)
            + 3.1661 * (X[:, 0] ** 2) * X[:, 3]
            + 19.84 * (X[:, 0] ** 2) * X[:, 2]
        )
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = -torch.round(X[:, 0] / self._d) * self._d + 0.0193 * X[:, 2]
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        g = -torch.round(X[:, 1] / self._d) * self._d + 0.00954 * X[:, 2]
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g3(self, X: Tensor):
        g = -torch.pi * (X[:, 2] ** 2) * X[:, 3] - 4 / 3 * torch.pi * (X[:, 2] ** 3) + 1296000
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g4(self, X: Tensor):
        g = X[:, 3] - 240
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return self.g1(X), self.g2(X), self.g3(X), self.g4(X)


class WBD4D:
    """4D Welded Beam Design"""

    dim = 4
    num_cons = 5
    _bounds = [(0.125, 10), (0.1, 10), (0.1, 10), (0.1, 10)]
    _optimal_value = 1.7250022
    _optimizers = [(0.20564426101885, 3.47257874213172, 9.03662391018928, 0.20572963979791)]

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        y = 1.10471 * ((X[:, 0] ** 2) * X[:, 1]) + 0.04811 * X[:, 2] * X[:, 3] * (14 + X[:, 1])
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        t1 = 6000 / (torch.sqrt(torch.tensor(2)) * X[:, 0] * X[:, 1])
        t2 = (
            6000
            * (14 + 0.5 * X[:, 1])
            * torch.sqrt(0.25 * ((X[:, 1] ** 2) + ((X[:, 0] + X[:, 2]) ** 2)))
            / (2 * (0.707 * X[:, 0] * X[:, 1] * ((X[:, 1] ** 2) / 12 + 0.25 * ((X[:, 0] + X[:, 2]) ** 2))))
        )
        t = torch.sqrt(
            (t1**2) + (t2**2) + X[:, 1] * t1 * t2 / torch.sqrt(0.25 * ((X[:, 1] ** 2) + ((X[:, 0] + X[:, 1]) ** 2)))
        )
        g = t - 13600
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        s = 504000 / ((X[:, 2] ** 2) * X[:, 3])
        g = s - 30000
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g3(self, X: Tensor):
        g = X[:, 0] - X[:, 3]
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g4(self, X: Tensor):
        P = 64746.022 * (1 - 0.0282346 * X[:, 2]) * X[:, 2] * (X[:, 3] ** 3)
        g = 6000 - P
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g5(self, X: Tensor):
        delta = 2.1952 / ((X[:, 2] ** 3) * X[:, 3])
        g = delta - 0.25
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return self.g1(X), self.g2(X), self.g3(X), self.g4(X), self.g5(X)


class SR7D:
    """7D Speed Reducer"""

    dim = 7
    num_cons = 11
    _bounds = [(2.6, 3.6), (0.7, 0.8), (17.0, 28.0), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]
    _optimal_value = 2996.3482
    _optimizers = [(3.5, 0.7, 17, 7.3, 7.8, 3.350215, 5.286683)]

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        y = (
            0.7854 * X[:, 0] * (X[:, 1] ** 2) * (3.3333 * (X[:, 2] ** 2) + 14.9334 * X[:, 2] - 43.0934)
            - 1.508 * X[:, 0] * ((X[:, 5] ** 2) + (X[:, 6] ** 2))
            + 7.477 * ((X[:, 5] ** 3) + (X[:, 6] ** 3))
            + 0.7854 * (X[:, 3] * (X[:, 5] ** 2) + X[:, 4] * (X[:, 6] ** 2))
        )
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = 27 / (X[:, 0] * (X[:, 1] ** 2) * X[:, 2]) - 1
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        g = 397.5 / (X[:, 0] * (X[:, 1] ** 2) * (X[:, 2] ** 2)) - 1
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g3(self, X: Tensor):
        g = 1.93 * (X[:, 3] ** 3) / (X[:, 1] * X[:, 2] * (X[:, 5] ** 4)) - 1
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g4(self, X: Tensor):
        g = 1.93 * (X[:, 4] ** 3) / (X[:, 1] * X[:, 2] * (X[:, 6] ** 4)) - 1
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g5(self, X: Tensor):
        g = (
            1 / (0.1 * (X[:, 5] ** 3)) * torch.sqrt(((((745 * X[:, 3]) / (X[:, 1] * X[:, 2])) ** 2) + 16.9 * (10**6)))
            - 1100
        )
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g6(self, X: Tensor):
        g = (
            1 / (0.1 * (X[:, 6] ** 3)) * torch.sqrt(((((745 * X[:, 4]) / (X[:, 1] * X[:, 2])) ** 2) + 157.5 * (10**6)))
            - 850
        )
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g7(self, X: Tensor):
        g = X[:, 1] * X[:, 2] - 40
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g8(self, X: Tensor):
        g = 5 - X[:, 0] / X[:, 1]
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g9(self, X: Tensor):
        g = X[:, 0] / X[:, 1] - 12
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g10(self, X: Tensor):
        g = (1.5 * X[:, 5] + 1.9) / X[:, 3] - 1
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g11(self, X: Tensor):
        g = (1.1 * X[:, 6] + 1.9) / X[:, 4] - 1
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return (
            self.g1(X),
            self.g2(X),
            self.g3(X),
            self.g4(X),
            self.g5(X),
            self.g6(X),
            self.g7(X),
            self.g8(X),
            self.g9(X),
            self.g10(X),
            self.g11(X),
        )


class Ackley10D:
    """10D Ackley"""

    dim = 10
    num_cons = 2
    _bounds = [(-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10), (-5, 10)]
    _optimal_value = 0
    _optimizers = [(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        a = 20
        b = 0.2
        c = 2 * torch.pi
        sx = torch.stack([X[:, i] ** 2 for i in range(self.dim)], dim=-1).sum(-1)
        cx = torch.stack([torch.cos(c * X[:, i]) for i in range(self.dim)], dim=-1).sum(-1)

        y = -a * torch.exp(-b * torch.sqrt(sx / self.dim)) - torch.exp(cx / self.dim) + a + torch.exp(torch.tensor(1.0))
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = torch.stack([X[:, i] for i in range(self.dim)], dim=-1).sum(-1)
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        g = torch.sqrt(torch.stack([X[:, i] ** 2 for i in range(self.dim)], dim=-1).sum(-1)) - 5
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return self.g1(X), self.g2(X)


class Ackley6D:
    """6D Ackley"""

    dim = 6
    num_cons = 1
    _bounds = [(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
    _optimal_value = None
    _optimizers = None

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        a = 20
        b = 0.2
        c = 2 * torch.pi
        sx = torch.stack([X[:, i] ** 2 for i in range(self.dim)], dim=-1).sum(-1)
        cx = torch.stack([torch.cos(c * X[:, i]) for i in range(self.dim)], dim=-1).sum(-1)

        y = -a * torch.exp(-b * torch.sqrt(sx / self.dim)) - torch.exp(cx / self.dim) + a + torch.exp(torch.tensor(1.0))
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = torch.stack([X[:, i] for i in range(self.dim)], dim=-1).sum(-1) - 3
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return (self.g1(X),)


class Keane30D:
    """30D Keane Bump"""

    dim = 30
    num_cons = 2
    _bounds = [
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
    ]
    _optimal_value = -0.818056222
    _optimizers = [
        (
            3.16822553,
            3.14621165,
            3.12453109,
            3.10297971,
            3.08182362,
            3.0605448,
            3.03919559,
            3.01767984,
            2.99563685,
            2.97354375,
            2.95066648,
            2.92756246,
            2.90308871,
            0.440434895,
            0.437505191,
            0.43510334,
            0.432460693,
            0.429815709,
            0.427295405,
            0.424698735,
            0.422641158,
            0.420028735,
            0.417678117,
            0.415577752,
            0.413108742,
            0.410869231,
            0.408999549,
            0.406826514,
            0.405042008,
            0.402869708,
        )
    ]

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        sc4x = torch.stack([torch.cos(X[:, i]) ** 4 for i in range(self.dim)], dim=-1).sum(-1)
        pc2x = torch.stack([torch.cos(X[:, i]) ** 2 for i in range(self.dim)], dim=-1).prod(-1)
        ix2 = torch.stack([(X[:, i] ** 2) * (i + 1) for i in range(self.dim)], dim=-1).sum(-1)

        y = torch.abs((sc4x - 2 * pc2x) / torch.sqrt(ix2))
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = 0.75 - torch.stack([X[:, i] for i in range(self.dim)], dim=-1).sum(-1)
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        g = torch.stack([X[:, i] for i in range(self.dim)], dim=-1).sum(-1) - 7.5 * self.dim
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return self.g1(X), self.g2(X)


class Toy2D:
    """2D Toy"""

    dim = 2
    num_cons = 2
    _bounds = [(0, 1), (0, 1)]
    _optimal_value = 0.6358
    _optimizers = [(0.1954, 0.4404)]

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        y = X[:, 0] + X[:, 1]
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = 0.5 * torch.sin(2 * torch.pi * ((X[:, 0] ** 2) - 2 * X[:, 1])) + X[:, 0] + 2 * X[:, 1] - 1.5
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        g = -(X[:, 0] ** 2) - (X[:, 1] ** 2) + 1.5
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return self.g1(X), self.g2(X)


class Rosenbrock5D:
    """5D Rosenbrock"""

    dim = 5
    num_cons = 2
    _bounds = [(-3, 5), (-3, 5), (-3, 5), (-3, 5), (-3, 5)]
    _optimal_value = None
    _optimizers = None

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        y = torch.stack(
            [(X[:, i + 1] - (X[:, i] ** 2)) ** 2 + ((X[:, i] - 1) ** 2) for i in range(self.dim - 1)], dim=1
        ).sum(dim=-1)
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = (
            ((X[:, 0] - 1) ** 2)
            + torch.cat([(i + 1) * ((2 * (X[:, i] ** 2) - X[:, i - 1]) ** 2) for i in range(1, self.dim)], dim=-1).sum(
                dim=-1
            )
            - 10
        )
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def g2(self, X: Tensor):
        w = [1 + (X[:, i] - 1) / 4 for i in range(self.dim)]
        g = (
            torch.sin(torch.pi * w[0]) ** 2
            + torch.cat(
                [((w[i] - 1) ** 2) * (1 + 10 * (torch.sin(torch.pi * w[i] + 1) ** 2)) for i in range(self.dim - 1)],
                dim=-1,
            ).sum(dim=-1)
            + ((w[self.dim - 1] - 1) ** 2) * (1 + (torch.sin(2 * torch.pi * w[self.dim - 1]) ** 2))
        ) - 10
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return self.g1(X), self.g2(X)


class Branin2D:
    """2D Branin"""

    dim = 2
    num_cons = 1
    _bounds = [(-5, 10), (0, 15)]
    _optimal_value = 0.3979
    _optimizers = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]

    def __init__(self, noise_std=0, negate=False, bounds=None):
        super().__init__()
        if bounds is not None:
            self._bounds = bounds
        self.noise_std = noise_std
        self.negate = negate

    def evaluate_true(self, X: Tensor):
        a = 1.0
        b = 5.1 / (4 * (torch.pi) ** 2)
        c = 5 / torch.pi
        r = 6
        s = 10
        t = 1 / (8 * torch.pi)

        y = (a * (X[:, 1] - b * (X[:, 0] ** 2) + c * X[:, 0] - r) ** 2) + s * (1 - t) * torch.cos(X[:, 0]) + s
        y = -y if self.negate else y
        y = y.unsqueeze(-1)

        return y, y + self.noise_std * torch.randn_like(y)

    def g1(self, X: Tensor):
        g = 50 - (((X[:, 0] - 2.5) ** 2) + ((X[:, 1] - 7.5) ** 2))
        g = g.unsqueeze(-1)

        return g, g + self.noise_std * torch.randn_like(g)

    def gs(self, X: Tensor):
        return (self.g1(X),)
