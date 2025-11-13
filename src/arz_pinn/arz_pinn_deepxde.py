import deepxde as dde
import torch
import numpy as np
from ve_modules import MonotoneVePiecewise, ve_smooth_regularizer

class ARZPINNWrapper:
    """
    包装 DeepXDE 的模型：定义 ARZ PDE、输出 (rho, v)、可学习 α, τ 与 Ve(ρ)。
    """

    def __init__(self, cfg):
        self.cfg = cfg
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        self.lanes = data_cfg["lanes"]
        self.rho_jam = data_cfg["rho_jam_per_lane"] * self.lanes
        self.v_max = data_cfg["v_max"]

        # 可学习参数
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(model_cfg["alpha_init"], dtype=torch.float32)))
        self.log_tau = torch.nn.Parameter(torch.log(torch.tensor(model_cfg["tau_init"], dtype=torch.float32)))
        self.alpha_bounds = torch.tensor(model_cfg["alpha_bounds"], dtype=torch.float32)
        self.tau_bounds = torch.tensor(model_cfg["tau_bounds"], dtype=torch.float32)

        self.ve_module = MonotoneVePiecewise(knots=model_cfg["ve_knots"], v_free=self.v_max)

        # 主干网络（使用 DeepXDE 内置 MLP 构造）
        self.net = dde.nn.FNN([2] + [model_cfg["hidden_units"]] * model_cfg["hidden_layers"] + [2], model_cfg["activation"], "Glorot normal")

    def parameters(self):
        return list(self.net.parameters()) + [self.log_alpha, self.log_tau] + list(self.ve_module.parameters())

    def freeze_Ve(self, freeze: bool):
        for p in self.ve_module.parameters():
            p.requires_grad_(not freeze)

    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), self.alpha_bounds[0], self.alpha_bounds[1])

    def tau(self):
        return torch.clamp(self.log_tau.exp(), self.tau_bounds[0], self.tau_bounds[1])

    def Ve(self, rho):
        rho_norm = (rho / self.rho_jam).clamp(0, 1)
        return self.ve_module(rho_norm)

    def forward(self, x):
        # x: numpy array (N,2) -> convert to torch for constraints then back
        xt = torch.tensor(x, dtype=torch.float32)
        out = self.net(xt)
        rho = self.rho_jam * torch.sigmoid(out[:, 0:1])
        v = self.v_max * torch.sigmoid(out[:, 1:2])
        return torch.cat([rho, v], dim=1).detach().numpy()

    def pde(self, x, y):
        """
        y: (rho, v)
        PDE residuals:
          Mass: ρ_t + (ρ v)_x = 0
          Momentum-like: (v + α ρ)_t + v (v + α ρ)_x - (Ve(ρ) - v)/τ = 0
        """
        rho = y[:, 0:1]
        v = y[:, 1:2]
        rho_t = dde.grad.jacobian(y, x, i=0, j=1)
        rho_x = dde.grad.jacobian(y, x, i=0, j=0)
        v_t = dde.grad.jacobian(y, x, i=1, j=1)
        v_x = dde.grad.jacobian(y, x, i=1, j=0)

        q = rho * v
        q_x = dde.grad.jacobian(q, x, i=0, j=0)

        alpha = self.alpha().detach().numpy()
        tau = self.tau().detach().numpy()

        # generalized velocity w = v + α ρ
        w = v + alpha * rho
        w_t = v_t + alpha * rho_t
        w_x = v_x + alpha * rho_x

        # relaxation term
        ve_val = self.Ve(torch.tensor(rho, dtype=torch.float32)).detach().numpy()
        relax = (ve_val - v) / (tau + 1e-6)

        r_mass = rho_t + q_x
        r_mom = w_t + v * w_x - relax
        return [r_mass, r_mom]

    def boundary_inlet(self, x, on_boundary):
        return on_boundary and np.isclose(x[0], self.cfg["data"]["x_min"])

    def boundary_outlet(self, x, on_boundary):
        return on_boundary and np.isclose(x[0], self.cfg["data"]["x_max"])

    def ic_condition(self, x):
        # 初值: t = 0
        return np.isclose(x[1], 0.0)

    def custom_losses(self):
        # Ve 平滑正则
        return ve_smooth_regularizer(self.ve_module, self.cfg["model"]["ve_smooth_reg"])
