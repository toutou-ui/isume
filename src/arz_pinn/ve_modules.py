import torch
import torch.nn.functional as F

class MonotoneVePiecewise(torch.nn.Module):
    """
    单调递减 Ve(ρ) 分段线性: 通过正值增量累积下降。
    rho_norm ∈ [0,1]; 输出速度单位 = m/s
    """
    def __init__(self, knots: int, v_free: float):
        super().__init__()
        self.knots = knots
        self.v_free = torch.nn.Parameter(torch.tensor(float(v_free)))
        # raw_deltas 经过 softplus 保证正，并映射成总下降
        self.raw_deltas = torch.nn.Parameter(torch.randn(knots))

    def forward(self, rho_norm: torch.Tensor):
        rho_norm = rho_norm.clamp(0.0, 1.0)
        deltas = F.softplus(self.raw_deltas)
        total = deltas.sum() + 1e-6
        weights = deltas / total
        # 生成 knot 值：从 v_free 累积往下
        knot_vals = self.v_free * (1 - torch.cumsum(weights, dim=0).clamp(0, 0.98))
        grid = torch.linspace(0.0, 1.0, self.knots, device=rho_norm.device)
        idx = torch.clamp(torch.searchsorted(grid, rho_norm, right=True) - 1, 0, self.knots - 2)
        x0 = grid[idx]; x1 = grid[idx + 1]
        y0 = knot_vals[idx]; y1 = knot_vals[idx + 1]
        t = ((rho_norm - x0) / (x1 - x0 + 1e-6)).clamp(0, 1)
        return y0 * (1 - t) + y1 * t


def ve_smooth_regularizer(module: MonotoneVePiecewise, weight: float):
    if not isinstance(module, MonotoneVePiecewise):
        return torch.tensor(0.0, device=next(module.parameters()).device)
    d = torch.nn.functional.softplus(module.raw_deltas)
    if d.numel() < 3:
        return torch.tensor(0.0, device=d.device)
    d2 = d[:-2] - 2 * d[1:-1] + d[2:]
    return weight * torch.mean(d2 ** 2)
