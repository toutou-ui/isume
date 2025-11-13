"""
PeMS 预处理 (占位):
输入: station_id, timestamp, flow(veh/interval), speed(km/h), (occupancy %) ...
输出: weak density, IC/BC, 监督张量
"""
import pandas as pd
import torch
from typing import Dict

def load_pems(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def weak_density(df: pd.DataFrame, eps: float = 0.5):
    # 单位转换: speed km/h -> m/s
    v = df["speed"].values / 3.6
    q = df["flow"].values  # 假定已经是 veh/s 或需除以间隔
    rho = q / (v + eps)
    return rho

def build_supervision(df: pd.DataFrame):
    # 返回张量化的 flow, speed, weak rho
    # TODO: 根据站点与时间展开空间轴 (插值站点位置)
    flow = torch.tensor(df["flow"].values, dtype=torch.float32)
    speed = torch.tensor(df["speed"].values / 3.6, dtype=torch.float32)
    rho_w = torch.tensor(weak_density(df), dtype=torch.float32)
    return {"flow_obs": flow, "speed_obs": speed, "rho_weak": rho_w}

def build_ic_bc(df: pd.DataFrame):
    # 简化: 第一时间片作为 IC, 最上游站点作为 inlet
    return {
        "ic_rho": torch.tensor([], dtype=torch.float32),  # TODO
        "ic_v": torch.tensor([], dtype=torch.float32),
        "inlet_q": torch.tensor([], dtype=torch.float32),
        "inlet_v": torch.tensor([], dtype=torch.float32),
        "outlet_v": torch.tensor([], dtype=torch.float32)
    }
