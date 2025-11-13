# ARZ-PINN Short-Term Traffic Evolution (DeepXDE Version)

本仓库提供一个基于 DeepXDE 的 ARZ-PINN 脚手架，用于高速/快速路短期 (5–30 min) 密度 ρ、速度 v 与流量 q=ρ·v 的物理一致演化预测。相比纯自定义实现，本版最大化复用成熟 PINN 框架，只保留必要的 ARZ 双 PDE、单调可学习平衡速度 Ve(ρ) 与多阶段训练逻辑。

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
scripts/
  run_train.sh
  run_eval.sh
requirements.txt
LICENSE (待添加)
README.md
```

## 快速开始
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 准备数据：
   - 放置原始 NGSIM/PeMS 文件到 `./data/...`
   - 运行 `data_prep_ngsim.py` / `data_prep_pems.py` 生成缓存（TODO 处需你补实现）
3. 训练：
   ```bash
   bash scripts/run_train.sh
   ```
4. 评测：
   ```bash
   bash scripts/run_eval.sh
   ```

## 三阶段配置 (示例)
- PhaseA（epochs=2000）：`learn_Ve=false`，只训练主干 + α, τ
- PhaseB（epochs=2500）：`learn_Ve=true`
- PhaseC（epochs=1500）：`learn_Ve=true + use_rar=true` 开启残差重采样

## 需要你补充的部分
- `data_prep_ngsim.py` / `data_prep_pems.py` 中的 TODO：数据读取、网格化 (Edie)、弱密度估计、IC/BC 构造。
- 根据实际路段调整 `config_ngsim.yaml` 的 `x_max`, `t_pred`, `v_max`, `rho_jam_per_lane`, 车道数等。

## 引用
- DeepXDE: Lu et al., "DeepXDE: A Deep Learning Library for Solving Differential Equations", SIAM J. Sci. Comput. 2021.
- NGSIM / PeMS 数据源官方说明。

## 许可
默认研究使用；请遵守各数据集自身的许可证。你可以添加 MIT / Apache-2.0 等许可到 LICENSE.

---
