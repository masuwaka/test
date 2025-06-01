import math
from typing import Dict, Optional, Tuple, Union

import torch
from botorch.models import ModelListGP, SingleTaskGP
from torch import Tensor, tensor
from torch.distributions import Normal
from torch.optim import LBFGS


def gaussian_entropy(s2: Tensor) -> Tensor:
    """分散が s2 のガウス分布のエントロピー。

    Args:
        s2 (Tensor): 分散。

    Returns:
        Tensor: エントロピー。
    """

    return 0.5 * torch.log(2 * math.pi * math.e * s2)


def gamma_entropy(a: Tensor, b: Tensor) -> Tensor:
    """形状パラとレートパラが (a,b) のガンマ分布のエントロピー。

    Args:
        a (Tensor): 形状パラメータ。
        b (Tensor): レートパラメータ。

    Returns:
        Tensor: エントロピー。
    """

    return a - torch.log(b) + torch.lgamma(a) + (1 - a) * torch.digamma(a)


def mus2_to_e1e2(mu: Tensor, s2: Tensor) -> Tuple[Tensor, Tensor]:
    """パラメータ (mu, s2) のガウス分布の自然パラメータ (eta1, eta2) を返す。

    Args:
        mu (Tensor): ガウス分布の平均。
        s2 (Tensor): ガウス分布の分散。

    Returns:
        Tuple[Tensor, Tensor]: 自然パラメータ (eta1, eta2)。
    """

    eta1 = mu / s2
    eta2 = -1 / (2 * s2)

    return (eta1, eta2)


def e1e2_to_mus2(eta1: Tensor, eta2: Tensor) -> Tuple[Tensor, Tensor]:
    """自然パラメータ (eta1, eta2) のガウス分布のパラメータ (mu1, mu2) を返す。

    Args:
        eta1 (Tensor): 自然パラメータ1 (x の係数).
        eta2 (Tensor): 自然パラメータ2 (x^2 の係数).

    Returns:
        Tuple[Tensor, Tensor]: ガウス分布のパラメータ (mu, s2)。
    """

    mu = -eta1 / (2 * eta2)
    s2 = -1 / (2 * eta2)

    return mu, s2


def ab_to_e1e2(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    """パラメータ (a, b) のガンマ分布の自然パラメータ (eta1, eta2) を返す。

    Args:
        a (Tensor): ガンマ分布の形状パラメータ。
        b (Tensor): ガンマ分布のレートパラメータ。

    Returns:
        Tuple[Tensor, Tensor]: 自然パラメータ (eta1, eta2)。
    """

    eta1 = a - 1
    eta2 = -b

    return (eta1, eta2)


def e1e2_to_ab(eta1: Tensor, eta2: Tensor) -> Tuple[Tensor, Tensor]:
    """自然パラメータ (eta1, eta2) の ガンマ分布のパラメータ (a, b) を返す.

    Args:
        eta1 (Tensor): 自然パラメータ1 (log(x) の係数)。
        eta2 (Tensor): 自然パラメータ2 (x の係数)。

    Returns:
        Tuple[Tensor, Tensor]: ガンマ分布のパラメータ (a, b)。
    """

    a = eta1 + 1
    b = -eta2

    return (a, b)


class Variable:
    """変数ノードクラス。"""

    def __init__(
        self,
        name: str,
        prior_mu: float = 0.5,
        prior_s2: float = 0.5,
    ):
        """初期化関数。

        Args:
            name (str):
                ノード名 (e.g., "x_1", "x_2", ..., "x_{D}")。
                インスタンスは D 個存在。

            prior_mu (float, optional): 事前分布の平均. Defaults to 0.5。
            prior_s2 (float, optional): 事前分布の分散. Defaults to 0.5。
        """

        self.name = name

        # 事前信念
        self.prior_mu, self.prior_s2 = tensor(prior_mu), tensor(prior_s2)

        # 事後信念 (１ループ目は事前信念そのまま)
        self.post_mu, self.post_s2 = tensor(prior_mu), tensor(prior_s2)

        # １ループ前に因子ノードに送ったメッセージ (e.g., to_factor_name: (sent_mu, sent_sigma))
        self.sent_messages = {}

        # 因子ノードから受け取ったメッセージ (e.g., from_factor_name: (recv_mu, recv_sigma))
        self.recv_messages = {}

    def send_message(self, to_factor):
        """因子ノードへのメッセージ送信関数。

        Args:
            to_factor (Factor): メッセージ送信先の Factor インスタンス。
        """

        # 送信先因子ノードからの受信メッセージがない場合 (１ループ目)、事前信念が送られる。
        if to_factor.name not in self.recv_messages.keys():
            self.sent_messages[to_factor.name] = (self.prior_mu, self.prior_s2)
            to_factor.recv_messages[self.name] = (self.prior_mu, self.prior_s2)

        # 送信先因子ノードからの受信メッセージがある場合（２ループ目以降）
        # 事後信念から受信メッセージを差し引いて送る（外部信念）
        else:
            post_eta1, post_eta2 = mus2_to_e1e2(self.post_mu, self.post_s2)
            recv_eta1, recv_eta2 = mus2_to_e1e2(*self.recv_messages[to_factor.name])
            sent_eta1, sent_eta2 = post_eta1 - recv_eta1, post_eta2 - recv_eta2
            sent_mu, sent_s2 = e1e2_to_mus2(sent_eta1, sent_eta2)

            self.sent_messages[to_factor.name] = (sent_mu, sent_s2)
            to_factor.recv_messages[self.name] = (sent_mu, sent_s2)

    def update_belief(self, lam_x):
        """事後信念の更新関数。"""

        post_eta1, post_eta2 = mus2_to_e1e2(self.prior_mu, self.prior_s2)
        for from_factor_name in self.recv_messages.keys():
            recv_eta1, recv_eta2 = mus2_to_e1e2(*self.recv_messages[from_factor_name])
            post_eta1, post_eta2 = post_eta1 + recv_eta1, post_eta2 + recv_eta2
        post_eta2 = post_eta2 - lam_x
        self.post_mu, self.post_s2 = e1e2_to_mus2(post_eta1, post_eta2)

    def entropy(self) -> Tensor:
        """事後信念のエントロピー取得関数。

        Returns:
            Tensor: エントロピー。
        """

        return gaussian_entropy(self.post_s2)


class StructureVariable:
    """構造変数ノードクラス。"""

    def __init__(self, name: str, prior_a: float = 2.0, prior_b: float = 2.0):
        """初期化関数。

        Args:
            name (str):
                ノード名 (e.g., "o_1_0", ..., "o_{D}_{K}")。
                "o_d_k" は "x_d" -> "f_k" への辺に対応。
                インスタンスは D x (K+1) 個存在。

            prior_a (float, optional): 事前分布の形状パラメータ。 Defaults to 2.0。
            prior_b (float, optional): 事前分布のレートパラメータ. Defaults to 2.0。
        """

        self.name = name

        # 事前信念
        self.prior_a, self.prior_b = tensor(prior_a), tensor(prior_b)

        # 事後信念 (１ループ目は事前信念そのまま)
        self.post_a, self.post_b = tensor(prior_a), tensor(prior_b)

        # １ループ前に因子ノードに送ったメッセージ (e.g., to_factor_name: (sent_a, sent_b))
        self.sent_messages = {}

        # 因子ノードから受け取ったメッセージ (e.g., from_factor_name: (recv_a, recv_b))
        self.recv_messages = {}

    def mu(self, a: Tensor, b: Tensor) -> Tensor:
        """平均値取得関数。

        Args:
            a (Tensor): 形状パラメータ。
            b (Tensor): レートパラメータ。

        Returns:
            Tensor: 平均値。
        """

        return a / b

    def send_message(self, to_factor):
        """因子ノードへのメッセージ送信関数。

        Args:
            to_factor (Factor): メッセージ送信先の Factor インスタンス。
        """

        # 送信先因子ノードからの受信メッセージがない場合 (１ループ目)、事前信念が送られる。
        if to_factor.name not in self.recv_messages.keys():
            self.sent_messages[to_factor.name] = (self.prior_a, self.prior_b)
            to_factor.recv_messages[self.name] = (self.prior_a, self.prior_b)

        # 送信先因子ノードからの受信メッセージがある場合（２ループ目以降）
        # 事後信念から受信メッセージを差し引いて送る（外部信念）
        else:
            post_eta1, post_eta2 = ab_to_e1e2(self.post_a, self.post_b)
            recv_eta1, recv_eta2 = ab_to_e1e2(*self.recv_messages[to_factor.name])
            sent_eta1, sent_eta2 = post_eta1 - recv_eta1, post_eta2 - recv_eta2
            sent_a, sent_b = e1e2_to_ab(sent_eta1, sent_eta2)

            self.sent_messages[to_factor.name] = (sent_a, sent_b)
            to_factor.recv_messages[self.name] = (sent_a, sent_b)

    def update_belief(self, lam_w: float):
        """事後信念の更新関数。"""
        post_eta1, post_eta2 = ab_to_e1e2(self.prior_a, self.prior_b)
        for from_factor_name in self.recv_messages.keys():
            recv_eta1, recv_eta2 = ab_to_e1e2(*self.recv_messages[from_factor_name])
            post_eta1, post_eta2 = post_eta1 + recv_eta1, post_eta2 + recv_eta2
        post_eta2 = post_eta2 - lam_w
        self.post_a, self.post_b = e1e2_to_ab(post_eta1, post_eta2)

    def entropy(self) -> Tensor:
        """事後信念のエントロピー取得関数。

        Returns:
            Tensor: エントロピー。
        """
        return gamma_entropy(self.post_a, self.post_b)

    def KLD_prior(self) -> Tensor:
        """事前信念と事後信念の間の KLD 取得関数。

        Returns:
            Tensor: KLD。
        """

        prior_a, prior_b = self.prior_a, self.prior_b
        post_a, post_b = self.post_a, self.post_b
        return (
            (post_a - prior_a) * torch.digamma(post_a)
            - torch.lgamma(post_a)
            + torch.lgamma(prior_a)
            + prior_a * torch.log(post_b / prior_b)
            + (prior_b - post_b) * post_a / post_b
        )


class Factor:
    """因子ノードクラス (目的/制約因子両対応)。"""

    def __init__(
        self,
        name: str,
        gp_model: SingleTaskGP,
        variables: Dict[str, Union[Variable, StructureVariable]],
        f_best: Optional[float] = None,
    ):
        """初期化関数。

        Args:
            name (str):
                ノード名 (e.g., "f_0", "f_1", ..., f_K)。
                "f_0" は目的因子、"f_1", ..., "f_{K}" は制約因子。

            gp_model (SingleTaskGP): 学習済みGPモデル。
            variables (Dict[str, Union[Variable, StructureVariable]]): 接続されている変数ノードの辞書。
            f_best (Optional[float], optional): 目的変数のベスト値。 Defaults to None。
        """

        self.name = name
        self.gp_model = gp_model
        self.f_best = f_best
        self.post_mu, self.post_s2 = tensor(0.0), tensor(0.0)
        self.post_a, self.post_b = tensor(0.0), tensor(0.0)

        # "f_0" は目的因子
        self.is_obj = True if self.name == "f_0" else False

        # １ループ前に変数ノードに送ったメッセージ
        # (e.g., to_variable_name: (sent_mu, sent_s2) or (sent_a, sent_b))
        self.sent_messages = {}

        # 変数ノードから受け取ったメッセージ
        # (e.g., from_variable_name: (recv_mu, sent_s2) or (recv_a, recv_b))
        self.recv_messages = {}

        # 接続されている変数ノードの辞書
        self.variables = variables

    def _log_ei(self, mu: Tensor, s2: Tensor) -> Tensor:
        """期待改善量 (対数)

        Args:
            mu (Tensor): 平均。
            s2 (Tensor): 分散。

        Returns:
            Tensor: 期待改善量 (対数)
        """

        z = (self.f_best - mu) / s2.sqrt()
        ei = s2.sqrt() * (z * Normal(0.0, 1.0).cdf(z) + Normal(0.0, 1.0).log_prob(z).exp())

        return torch.log(ei)

    def _log_phi(self, mu: Tensor, s2: Tensor) -> Tensor:
        """可行確率 (対数)

        Args:
            mu (Tensor): 平均。
            s2 (Tensor): 分散。

        Returns:
            Tensor: 可行確率 (対数)
        """

        z = -mu / s2.sqrt()

        return torch.log(Normal(0.0, 1.0).cdf(z))

    def send_message(self, to_variable: Union[Variable, StructureVariable]):
        """変数ノードへのメッセージ送信関数。

        Args:
            to_variable (Union[Variables, StructureVariable]): メッセージ送信先の Variable/StructureVariable インスタンス。
        """

        if to_variable.name.startswith("x"):
            # 送信先が変数ノードの場合
            post_eta1, post_eta2 = mus2_to_e1e2(self.post_mu, self.post_s2)
            recv_eta1, recv_eta2 = mus2_to_e1e2(*self.recv_messages[to_variable.name])
            sent_eta1, sent_eta2 = post_eta1 - recv_eta1, post_eta2 - recv_eta2
            sent_mu, sent_s2 = e1e2_to_mus2(sent_eta1, sent_eta2)

            self.sent_messages[to_variable.name] = (sent_mu, sent_s2)
            to_variable.recv_messages[self.name] = (sent_mu, sent_s2)
        else:
            # 送信先が構造変数ノードの場合
            post_eta1, post_eta2 = ab_to_e1e2(self.post_a, self.post_b)
            recv_eta1, recv_eta2 = ab_to_e1e2(*self.recv_messages[to_variable.name])
            sent_eta1, sent_eta2 = post_eta1 - recv_eta1, post_eta2 - recv_eta2
            sent_a, sent_b = e1e2_to_ab(sent_eta1, sent_eta2)

            self.sent_messages[to_variable.name] = (sent_a, sent_b)
            to_variable.recv_messages[self.name] = (sent_a, sent_b)

    def update_belief(self):
        """事後信念の更新関数。"""

        # 変数ノードから受信したメッセージ
        mu = torch.stack([self.recv_messages[vn][0] for vn in self.variables.keys() if vn.startswith("x")])
        s2 = torch.stack([self.recv_messages[vn][1] for vn in self.variables.keys() if vn.startswith("x")])

        # 構造変数ノードから受信したメッセージから計算した精度パラメータ (a/b)
        p = torch.stack(
            [
                self.recv_messages[vn][0] / self.recv_messages[vn][1]
                for vn in self.variables.keys()
                if vn.startswith("o")
            ]
        )

        print()
        print(f"factor: {self.name} is_obj={self.is_obj}")
        print(f"variables: ", end="")
        [print(f"{vn} ", end="") for vn in self.variables.keys() if vn.startswith("x")]
        print()
        print(f"structures: ", end="")
        [print(f"{vn} ", end="") for vn in self.variables.keys() if vn.startswith("o")]
        print()
        print()

        # モード探索 by LBFGS
        x_dash = mu.clone().detach().requires_grad_(True)

        def closure():
            self.gp_model.zero_grad()
            if x_dash.grad is not None:
                x_dash.grad.zero_()
            posterior = self.gp_model.posterior(x_dash.unsqueeze(0))
            mu_gp = posterior.mean.squeeze()
            s2_dash = (s2 / p).sum() + posterior.variance.squeeze()
            logf = self._log_ei(mu_gp, s2_dash) if self.is_obj else self._log_phi(mu_gp, s2_dash)
            loss = -logf
            loss.backward()
            return loss

        print(f"x before LBFGS: {x_dash.detach()}")
        LBFGS([x_dash], max_iter=10, lr=1e-2, tolerance_grad=1e-6).step(closure)
        print(f"x after LBFGS: {x_dash.detach()}")
        print()

        def compute_hessian_diagonal_numerical(x_dash, eps=1e-4):
            g2_star, s2_star = [], []

            # 各次元について数値微分
            for d in range(x_dash.shape[0]):
                # 前方差分と後方差分用の点を作成
                x_plus = x_dash.clone()
                x_minus = x_dash.clone()
                x_plus[d] += eps
                x_minus[d] -= eps

                # それぞれの点で1階微分を計算
                # x_plusでの勾配
                posterior_plus = self.gp_model.posterior(x_plus.unsqueeze(0))
                mu_gp_plus = posterior_plus.mean.squeeze()
                s2_dash_plus = (s2 / p).sum() + posterior_plus.variance.squeeze()
                logf_plus = (
                    self._log_ei(mu_gp_plus, s2_dash_plus) if self.is_obj else self._log_phi(mu_gp_plus, s2_dash_plus)
                )
                g1_plus = torch.autograd.grad(logf_plus, x_plus, retain_graph=True)[0]

                # x_minusでの勾配
                posterior_minus = self.gp_model.posterior(x_minus.unsqueeze(0))
                mu_gp_minus = posterior_minus.mean.squeeze()
                s2_dash_minus = (s2 / p).sum() + posterior_minus.variance.squeeze()
                logf_minus = (
                    self._log_ei(mu_gp_minus, s2_dash_minus)
                    if self.is_obj
                    else self._log_phi(mu_gp_minus, s2_dash_minus)
                )
                g1_minus = torch.autograd.grad(logf_minus, x_minus, retain_graph=True)[0]

                # 中央差分で2階微分を近似
                g2_d = (g1_plus[d] - g1_minus[d]) / (2 * eps)
                g2_star.append(g2_d.detach())
                s2_star.append(-1.0 / g2_d.detach())

            return g2_star, s2_star

        print(f"x before hess: {x_dash.detach()}")
        g2_star, s2_star = compute_hessian_diagonal_numerical(x_dash)
        print(f"x after hess: {x_dash.detach()}")
        print()

        g2_star = torch.stack(g2_star)
        print(f"hess: {g2_star.detach()}")
        s2_star = torch.stack(s2_star)
        mu_star = x_dash.detach()
        print(f"xs: {mu_star}")
        print()

        # 事後信念更新
        post_eta1x, post_eta2x = tensor(0.0), tensor(0.0)
        post_eta1o, post_eta2o = tensor(0.0), tensor(0.0)
        for from_variable_name in self.recv_messages.keys():
            if from_variable_name.startswith("x"):
                recv_eta1, recv_eta2 = mus2_to_e1e2(*self.recv_messages[from_variable_name])
                post_eta1x, post_eta2x = post_eta1x + recv_eta1, post_eta2x + recv_eta2
            else:
                recv_eta1, recv_eta2 = ab_to_e1e2(*self.recv_messages[from_variable_name])
                post_eta1o, post_eta2o = post_eta1o + recv_eta1, post_eta2o + recv_eta2
        for d in range(len(g2_star)):
            star_eta1, star_eta2 = mus2_to_e1e2(mu_star[d], s2_star[d])
            post_eta1x, post_eta2x = post_eta1x + star_eta1, post_eta2x + star_eta2
            star_eta1, star_eta2 = -g2_star[d], tensor(0.0)
            post_eta1o, post_eta2o = post_eta1o + star_eta1, post_eta2o + star_eta2
        self.post_mu, self.post_s2 = e1e2_to_mus2(post_eta1x, post_eta2x)
        self.post_a, self.post_b = e1e2_to_ab(post_eta1o, post_eta2o)
        # print(f"{self.name} post_mu: {self.post_mu}")
        # print(f"{self.name} post_std: {math.sqrt(self.post_s2)}")
        # print(f"{self.name} post_Ew: {self.post_a/ self.post_b}")
        # exit(0)

    def expected_log_score(self) -> Tensor:
        """関数スコア計算関数。

        Returns:
            Tensor: 関数スコア。
        """
        posterior = self.gp_model.posterior(self.post_mu.unsqueeze(0))
        mu = posterior.mean.squeeze()
        s2 = posterior.variance.squeeze()
        if self.is_obj:
            return -self._log_ei(mu, s2)
        else:
            return -self._log_phi(mu, s2)


class FactorGraph:
    def __init__(self, model_list: ModelListGP, f_best):
        """初期化関数。

        Args:
            model_list (ModelListGP): 学習済みモデルリスト（目的, 制約1, 制約2, ..., 制約K の順）
            f_best (_type_): _description_
        """

        self.D = model_list.models[0].train_inputs[0].shape[-1]
        self.K = len(model_list.models) - 1

        self.variables: Dict[str, Union[Variable, StructureVariable]] = {}
        self.factors: Dict[str, Factor] = {}

        for d in range(1, self.D + 1):
            name = f"x_{d}"
            self.variables[name] = Variable(name, 0.4)

        for k in range(0, self.K + 1):
            variables = {}
            name = f"x_{d}"
            variables[name] = self.variables[name]
            for d in range(1, self.D + 1):
                name = f"o_{d}_{k}"
                self.variables[name] = StructureVariable(name)
                variables[name] = self.variables[name]
            name = f"f_{k}"
            self.factors[name] = Factor(name, model_list.models[k], variables, f_best=f_best if k == 0 else None)

    def run(self, max_iter: int = 20, delta: float = 0.05):
        lam = math.sqrt(math.log(2 / delta) / (2 * max(1, self.D)))
        F = []
        for i in range(max_iter):
            print(f"[Loop-{i}]")

            # 変数ノードから因子ノードにメッセージ送信
            for variable in self.variables.values():
                if variable.name.startswith("x"):
                    print(f"{variable.name} (mu, std)=({variable.post_mu}, {variable.post_s2})")
                else:
                    print(f"{variable.name} prec={variable.post_a/variable.post_b}")

                for to_factor in self.factors.values():
                    variable.send_message(to_factor)

            # 因子ノードの事後信念を計算
            for factor in self.factors.values():
                factor.update_belief()

            # 因子ノードから変数ノードへメッセージ送信
            for factor in self.factors.values():
                for to_variable in self.variables.values():
                    factor.send_message(to_variable)

            # 変数ノードの事後信念を計算
            for variable in self.variables.values():
                variable.update_belief(lam)

            F.append(self.free_energy(lam, lam))
            if i > 4 and abs(F[-2] - F[-1]) < 1e-6:
                break
        return F

    def free_energy(self, lam_x: float, lam_w: float):
        energy = sum(factor.expected_log_score() for factor in self.factors.values())
        KLD = sum(
            self.variables[variable_name].KLD_prior()
            for variable_name in self.variables.keys()
            if variable_name.startswith("o")
        )
        Hx = sum(
            self.variables[variable_name].entropy()
            for variable_name in self.variables.keys()
            if variable_name.startswith("x")
        )
        Hw = sum(
            self.variables[variable_name].entropy()
            for variable_name in self.variables.keys()
            if variable_name.startswith("o")
        )
        return energy + KLD + lam_x * Hx + lam_w * Hw
