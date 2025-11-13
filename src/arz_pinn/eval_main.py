import os
import torch
import yaml
import numpy as np
from src.arz_pinn.eval import multi_horizon_metrics

"""
示例评测主入口：
- 加载模型 checkpoint
- 重新构建测试网格（x,t）
- 生成预测 rho,v,q 并与缓存中的 ground truth 比较
- 输出 CSV 结果

注意：这个示例依赖 data_prep_* 生成的缓存数据格式。请根据你的缓存实现调整加载逻辑。
"""

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def load_cache(cache_path):
    # TODO: 加载你在 data_prep 中存储的测试网格和真值
    # 期望返回 dict 包含 'rho_true','v_true','q_true','x_coords','t_coords'
    return {}

def predict_from_model(checkpoint_path, cfg):
    # TODO: 根据 checkpoint 恢复模型并预测。示例只返回空 dict
    return { 'rho': None, 'v': None, 'q': None }

def main(cfg_path, checkpoint_path, cache_path, out_csv):
    cfg = load_cfg(cfg_path)
    cache = load_cache(cache_path)
    pred = predict_from_model(checkpoint_path, cfg)
    metrics = multi_horizon_metrics(pred, cache, cfg['eval']['horizons'])
    # TODO: 写 CSV
    print(metrics)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cache', required=True)
    ap.add_argument('--out', required=False, default='metrics.csv')
    args = ap.parse_args()
    main(args.config, args.ckpt, args.cache, args.out)
