import torch

def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((a - b) ** 2)).item()

def wavefront_position(rho_grid: torch.Tensor, x_coords: torch.Tensor, thr: float):
    pos = []
    for t in range(rho_grid.shape[0]):
        idx = torch.nonzero(rho_grid[t] >= thr, as_tuple=False)
        if idx.numel() == 0:
            pos.append(x_coords[0].item())
        else:
            xmax = int(torch.max(idx))
            pos.append(x_coords[xmax].item())
    return torch.tensor(pos)

def wavefront_error(rho_pred: torch.Tensor, rho_true: torch.Tensor, x_coords: torch.Tensor, thr: float):
    wp = wavefront_position(rho_pred, x_coords, thr)
    wt = wavefront_position(rho_true, x_coords, thr)
    return torch.mean(torch.abs(wp - wt)).item()

def multi_horizon_metrics(pred: dict, true: dict, horizons: list):
    out = {}
    for h in horizons:
        # TODO: 根据 h 切片
        out[h] = {
            "rmse_rho": rmse(pred["rho"], true["rho"]),
            "rmse_v": rmse(pred["v"], true["v"]),
            "rmse_q": rmse(pred["q"], true["q"])
        }
    return out
