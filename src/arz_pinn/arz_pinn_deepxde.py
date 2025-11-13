import deepxde as dde
from deepxde.backend import torch as bkd
import torch
from .ve_modules import MonotoneVePiecewise, ve_smooth_regularizer

class ARZPINNWrapper:
    """
    DeepXDE + PyTorch 后端的 ARZ-PINN 包装：
    - 可学习参数：alpha, tau, Ve(ρ)
    - 主干网络：FNN( (x,t) -> (rho,v) )，通过 sigmoid 映射到物理范围
    - PDE 残差：全程使用 torch 张量，避免 numpy 混用
    """
    def __init__(self, cfg):
        self.cfg = cfg
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        self.lanes = int(data_cfg["lanes"])
        self.rho_jam = float(data_cfg["rho_jam_per_lane"]) * self.lanes
        self.v_max = float(data_cfg["v_max"])

        # 可学习参数（外部传入 external_trainable_variables）
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(model_cfg["alpha_init"], dtype=torch.float32)))
        self.log_tau = torch.nn.Parameter(torch.log(torch.tensor(model_cfg["tau_init"], dtype=torch.float32)))
        self.alpha_bounds = torch.tensor(model_cfg["alpha_bounds"], dtype=torch.float32)
        self.tau_bounds = torch.tensor(model_cfg["tau_bounds"], dtype=torch.float32)

        self.ve_module = MonotoneVePiecewise(knots=model_cfg["ve_knots"], v_free=self.v_max)

        # FNN 主干
        self.net = dde.nn.FNN(
            [2] + [model_cfg["hidden_units"]] * model_cfg["hidden_layers"] + [2],
            model_cfg["activation"],
            "Glorot normal",
        )

    def external_parameters(self):
        # 仅返回“外部可训练参数”（alpha/tau/Ve），网络权重由 DeepXDE 模型自身管理
        return [self.log_alpha, self.log_tau] + list(self.ve_module.parameters())

    def freeze_Ve(self, freeze: bool):
        for p in self.ve_module.parameters():
            p.requires_grad_(not freeze)

    def alpha(self):
        return torch.clamp(self.log_alpha.exp(), self.alpha_bounds[0], self.alpha_bounds[1])

    def tau(self):
        return torch.clamp(self.log_tau.exp(), self.tau_bounds[0], self.tau_bounds[1])

    def Ve(self, rho):  # rho: torch tensor
        rho_norm = (rho / self.rho_jam).clamp(0.0, 1.0)
        return self.ve_module(rho_norm)

    def pde(self, x, y):
        """
        y = (rho, v)
        Mass:     rho_t + (rho*v)_x = 0
        Momentum: (v + α ρ)_t + v (v + α ρ)_x - (Ve(ρ) - v)/τ = 0
        """
        rho = y[:, 0:1]
        v = y[:, 1:2]

        rho_t = dde.grad.jacobian(y, x, i=0, j=1)
        rho_x = dde.grad.jacobian(y, x, i=0, j=0)
        v_t = dde.grad.jacobian(y, x, i=1, j=1)
        v_x = dde.grad.jacobian(y, x, i=1, j=0)

        q = rho * v
        q_x = dde.grad.jacobian(q, x, i=0, j=0)

        alpha = self.alpha()
        tau = self.tau()

        w = v + alpha * rho
        w_t = v_t + alpha * rho_t
        w_x = v_x + alpha * rho_x

        ve_val = self.Ve(rho)
        relax = (ve_val - v) / (tau + 1e-6)

        r_mass = rho_t + q_x
        r_mom = w_t + v * w_x - relax
        return [r_mass, r_mom]

    # DeepXDE 边界/初值条件辅助
    def boundary_inlet(self, x, on_boundary):
        return on_boundary and bkd.numpy(x[0]) == self.cfg["data"]["x_min"]

    def boundary_outlet(self, x, on_boundary):
        return on_boundary and bkd.numpy(x[0]) == self.cfg["data"]["x_max"]

    def ic_condition(self, x):
        return bkd.numpy(x[1]) == 0.0

    def custom_losses(self):
        return ve_smooth_regularizer(self.ve_module, self.cfg["model"]["ve_smooth_reg"])
