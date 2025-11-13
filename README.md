# ARZ-PINN Short-Term Traffic Evolution (DeepXDE Version)

本仓库提供一个基于 [DeepXDE](https://github.com/lululxvi/deepxde) 的 ARZ-PINN 脚手架，用于高速/快速路短期 (5–30 min) 密度 ρ、速度 v 与流量 q=ρ·v 的物理一致演化预测。最大化复用成熟 PINN 框架，只保留必要的 ARZ 双 PDE、单调可学习平衡速度 Ve(ρ) 与多阶段训练逻辑。

## 特性
- ARZ 双方程：质量守恒 + 广义速度（带压力 α、松弛时间 τ）
- 单调可学习 Ve(ρ)：分段线性递减，带平滑正则
- 三阶段训练：固定 Ve → 解锁 Ve → RAR（残差自适应重采样）
- 数据融合：轨迹强监督 (ρ,v,q) + 环检弱监督 (q,v,ρ≈q/v)
- 评测：多步 RMSE、拥堵波前定位误差、PDE 残差
- 可扩展到 SUMO 仿真长窗与 PeMS 轻量迁移

## 目录结构
```
configs/
  config_ngsim.yaml
  config_pems.yaml
src/arz_pinn/
  ve_modules.py
  arz_pinn_deepxde.py
  data_prep_ngsim.py
  data_prep_pems.py
  train_deepxde.py
  eval.py
  eval_main.py
scripts/
  run_train.sh
  run_eval.sh
  generate_ngsim_cache.py
requirements.txt
LICENSE
README.md
```

## 快速开始
1) 安装依赖
```bash
pip install -r requirements.txt
```

2) 准备数据缓存（推荐 NGSIM 先做一个小窗示例）
- 将原始轨迹 CSV 放在 `data/ngsim/raw/`（列名至少包含：`vehicle_id,t,x,speed`，单位：秒/米/米每秒；0.1s 或 0.5s 采样均可）
- 生成缓存（Edie 网格化 + IC/BC）：
```bash
python scripts/generate_ngsim_cache.py \
  --raw data/ngsim/raw/trajectories.csv \
  --out data/ngsim/processed/grid_cache.npz \
  --dx 30 --dt 1 --x-min 0 --x-max 1000 --t-min 0 --t-max 600
```

3) 训练（DeepXDE，三阶段 + 可选 RAR）
```bash
bash scripts/run_train.sh
```

4) 评测与可视化
```bash
bash scripts/run_eval.sh
# 或：
python src/arz_pinn/eval_main.py \
  --config configs/config_ngsim.yaml \
  --ckpt checkpoints/ngsim/final_model.pt \
  --cache data/ngsim/processed/grid_cache.npz \
  --out outputs/ngsim/metrics.csv
```

## 与成熟仓库的复用
- PINN 主体：DeepXDE（RAR/边界/初值/采样/训练循环），我们仅提供 ARZ PDE 残差与可学习 Ve(ρ) 的 glue。
- 图时空深度基线：建议用 [LibCity](https://github.com/LibTraffic/Bigscity-LibCity) 直接复现 DCRNN/GraphWaveNet/GMAN/MTGNN 等（本仓库不复制其代码，推荐独立运行后导入指标对比）。

## 数据缓存格式（NGSIM 示例，npz）
- `rho_grid[T,X]`、`v_grid[T,X]`、`q_grid[T,X]`
- `x_coords[X]`（米）、`t_coords[T]`（秒）
- `raw_dt`（原始轨迹采样间隔秒）
- `ic_rho[X], ic_v[X], inlet_q[T], inlet_v[T], outlet_v[T]`

## 许可
MIT（本仓库代码）+ 遵守各数据原始许可。请勿直接将大体量第三方数据入库，建议用下载脚本或外部数据目录挂载。

---