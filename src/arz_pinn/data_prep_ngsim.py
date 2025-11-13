"""
NGSIM 轨迹预处理 (占位): 需要你实现
功能: 读取单车轨迹 -> 网格化 (Edie) -> 构造 IC/BC + 监督数据张量
"""
import pandas as pd
import numpy as np
import torch
from typing import Dict

def load_raw_trajectories(path: str) -> pd.DataFrame:
    # 预期列: vehicle_id, t(sec), x(m沿线), speed(m/s), lane, length(m)
    return pd.read_csv(path)

def edie_grid(df: pd.DataFrame, dx: float, dt: float, x_min: float, x_max: float, t_min: float, t_max: float):
    """
    TODO: 真实实现:
      1. 建立 x 与 t 网格
      2. 对每格统计占有时间 -> ρ
      3. 平均速度 -> v
      4. q = ρ v 或计数过边
    返回: dict 包含 rho_grid, v_grid, q_grid, x_coords, t_coords (torch.Tensor)
    """
    return {
        "rho_grid": torch.zeros(10, 10),
        "v_grid": torch.zeros(10, 10),
        "q_grid": torch.zeros(10, 10),
        "x_coords": torch.linspace(x_min, x_max, 10),
        "t_coords": torch.linspace(t_min, t_max, 10)
    }

def build_supervision_tensors(grid: Dict):
    # 展平监督 (示例): 你可以根据需要返回 mask 和观测值
    return {
        "rho_obs": grid["rho_grid"],
        "v_obs": grid["v_grid"],
        "q_obs": grid["q_grid"]
    }

def build_ic_bc(grid: Dict) -> Dict:
    # IC: t=0 切片; BC: x=0 与 x=L 的 q/v 时间序列
    return {
        "ic_rho": grid["rho_grid"][0],      # 简化示例
        "ic_v": grid["v_grid"][0],
        "inlet_q": grid["q_grid"][:, 0],
        "inlet_v": grid["v_grid"][:, 0],
        "outlet_v": grid["v_grid"][:, -1]
    }
