"""
NGSIM 轨迹预处理（可运行版本）：
- 基于“样本驻留时间”近似的 Edie 栅格化：k = Σ T / (ΔxΔt), v = 栅格均值, q = k*v
- 输出 ρ/v/q 栅格、x/t 坐标、IC/BC 与 raw_dt
注意：若原始采样间隔不均匀，使用中位数近似 raw_dt
"""
from typing import Dict
import numpy as np
import pandas as pd
import torch

def load_raw_trajectories(path: str) -> pd.DataFrame:
    """
    预期列：vehicle_id, t(秒), x(米，沿线), speed(米/秒), [lane], [length]
    """
    df = pd.read_csv(path)
    need = {"vehicle_id", "t", "x", "speed"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    return df

def infer_raw_dt(df: pd.DataFrame) -> float:
    raw = []
    for _, g in df.groupby("vehicle_id"):
        if len(g) >= 2:
            dt = np.diff(np.sort(g["t"].values))
            if len(dt) > 0:
                raw.append(np.median(dt))
    return float(np.median(raw)) if raw else 0.1

def edie_grid(df: pd.DataFrame, dx: float, dt: float, x_min: float, x_max: float, t_min: float, t_max: float) -> Dict:
    raw_dt = infer_raw_dt(df)

    x_edges = np.arange(x_min, x_max + 1e-6, dx)
    t_edges = np.arange(t_min, t_max + 1e-6, dt)
    X = len(x_edges) - 1
    T = len(t_edges) - 1

    m = (df["x"].between(x_min, x_max)) & (df["t"].between(t_min, t_max))
    d = df.loc[m, ["vehicle_id", "t", "x", "speed"]].copy()

    i_x = np.clip(np.searchsorted(x_edges, d["x"].values, side="right") - 1, 0, X - 1)
    i_t = np.clip(np.searchsorted(t_edges, d["t"].values, side="right") - 1, 0, T - 1)

    T_sum = np.zeros((T, X), dtype=np.float64)
    V_sum = np.zeros((T, X), dtype=np.float64)
    V_cnt = np.zeros((T, X), dtype=np.int64)

    for ix, it, v in zip(i_x, i_t, d["speed"].values):
        T_sum[it, ix] += raw_dt
        V_sum[it, ix] += v
        V_cnt[it, ix] += 1

    rho = (T_sum / (dx * dt)) * 1000.0  # veh/km
    v = np.divide(V_sum, np.maximum(V_cnt, 1), where=V_cnt > 0)  # m/s
    q = rho * v

    x_coords = (x_edges[:-1] + x_edges[1:]) / 2.0
    t_coords = (t_edges[:-1] + t_edges[1:]) / 2.0

    return {
        "rho_grid": torch.tensor(rho, dtype=torch.float32),
        "v_grid": torch.tensor(v, dtype=torch.float32),
        "q_grid": torch.tensor(q, dtype=torch.float32),
        "x_coords": torch.tensor(x_coords, dtype=torch.float32),
        "t_coords": torch.tensor(t_coords, dtype=torch.float32),
        "raw_dt": torch.tensor(raw_dt, dtype=torch.float32),
    }

def build_ic_bc(grid: Dict) -> Dict:
    rho = grid["rho_grid"]
    v = grid["v_grid"]
    q = grid["q_grid"]
    ic_rho = rho[0].clone()
    ic_v = v[0].clone()
    inlet_q = q[:, 0].clone()
    inlet_v = v[:, 0].clone()
    outlet_v = v[:, -1].clone()
    return {"ic_rho": ic_rho, "ic_v": ic_v, "inlet_q": inlet_q, "inlet_v": inlet_v, "outlet_v": outlet_v}

def save_npz(path: str, grid: Dict, icbc: Dict):
    np.savez_compressed(
        path,
        rho_grid=grid["rho_grid"].numpy(),
        v_grid=grid["v_grid"].numpy(),
        q_grid=grid["q_grid"].numpy(),
        x_coords=grid["x_coords"].numpy(),
        t_coords=grid["t_coords"].numpy(),
        raw_dt=np.array(grid["raw_dt"].numpy()),
        ic_rho=icbc["ic_rho"].numpy(),
        ic_v=icbc["ic_v"].numpy(),
        inlet_q=icbc["inlet_q"].numpy(),
        inlet_v=icbc["inlet_v"].numpy(),
        outlet_v=icbc["outlet_v"].numpy(),
    )
