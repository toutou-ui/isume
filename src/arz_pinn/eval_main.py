import os
import yaml
import numpy as np
import torch
import deepxde as dde
from .arz_pinn_deepxde import ARZPINNWrapper
from .eval import multi_horizon_metrics, wavefront_error

def load_cfg(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def load_cache(npz_path: str):
    z = np.load(npz_path)
    return {
        "rho": torch.tensor(z["rho_grid"], dtype=torch.float32),
        "v": torch.tensor(z["v_grid"], dtype=torch.float32),
        "q": torch.tensor(z["q_grid"], dtype=torch.float32),
        "x": torch.tensor(z["x_coords"], dtype=torch.float32),
        "t": torch.tensor(z["t_coords"], dtype=torch.float32),
    }

def predict_grid(wrapper: ARZPINNWrapper, model: dde.Model, x_coords: torch.Tensor, t_coords: torch.Tensor):
    X = x_coords.shape[0]
    T = t_coords.shape[0]
    xx, tt = torch.meshgrid(x_coords, t_coords, indexing="xy")  # [X,T]
    xt = torch.stack([xx.flatten(), tt.flatten()], dim=1).numpy()
    y = model.net(xt)  # DeepXDE net forward
    y = torch.tensor(y, dtype=torch.float32)
    rho = torch.sigmoid(y[:, 0]) * wrapper.rho_jam
    v = torch.sigmoid(y[:, 1]) * wrapper.v_max
    q = rho * v
    rho = rho.view(X, T).T.contiguous()
    v = v.view(X, T).T.contiguous()
    q = q.view(X, T).T.contiguous()
    return {"rho": rho, "v": v, "q": q}

def main(cfg_path: str, ckpt_path: str, cache_path: str, out_csv: str):
    dde.backend.set_default_backend("pytorch")

    cfg = load_cfg(cfg_path)
    cache = load_cache(cache_path)
    wrapper = ARZPINNWrapper(cfg)

    x_min, x_max = cfg["data"]["x_min"], cfg["data"]["x_max"]
    geom = dde.geometry.Interval(x_min, x_max)
    timedomain = dde.geometry.TimeDomain(0.0, cfg["data"]["t_pred"])
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def pde_func(x, y):
        return wrapper.pde(x, y)

    data = dde.data.TimePDE(geomtime, pde_func, [], num_domain=128, num_boundary=0, num_initial=0)
    model = dde.Model(data, wrapper.net)

    model.restore(ckpt_path, verbose=0)

    preds = predict_grid(wrapper, model, cache["x"], cache["t"])
    metrics = multi_horizon_metrics(preds, {"rho": cache["rho"], "v": cache["v"], "q": cache["q"]}, cfg["eval"]["horizons"])
    wf_err = wavefront_error(preds["rho"], cache["rho"], cache["x"], thr=cfg["eval"]["wavefront_rho_thr_frac"] * (cfg["data"]["rho_jam_per_lane"] * cfg["data"]["lanes"]))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["horizon_sec", "rmse_rho", "rmse_v", "rmse_q"])
        for h in cfg["eval"]["horizons"]:
            w.writerow([h, metrics[h]["rmse_rho"], metrics[h]["rmse_v"], metrics[h]["rmse_q"]])
        w.writerow([])
        w.writerow(["wavefront_abs_error_m", wf_err])

    print(f"Saved metrics to {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.config, args.ckpt, args.cache, args.out)