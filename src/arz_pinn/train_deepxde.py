"""
三阶段 DeepXDE 训练脚本（PyTorch 后端）：
- PhaseA: 固定 Ve
- PhaseB: 解锁 Ve
- PhaseC: 启动 RAR
- additional_losses: 从缓存网格抽样监督 q, v, ρ_weak (ρ≈q/v)
"""
import os, yaml, random
import numpy as np
import torch
import deepxde as dde
from .arz_pinn_deepxde import ARZPINNWrapper

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def build_geometry(cfg):
    x_min, x_max = cfg["data"]["x_min"], cfg["data"]["x_max"]
    t_min, t_max = 0.0, cfg["data"]["t_pred"]
    geom = dde.geometry.Interval(x_min, x_max)
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    return dde.geometry.GeometryXTime(geom, timedomain)

def load_ngsim_cache(cache_path: str):
    if not cache_path or not os.path.exists(cache_path):
        return None
    z = np.load(cache_path)
    out = {k: torch.tensor(z[k], dtype=torch.float32) for k in ["rho_grid", "v_grid", "q_grid"]}
    out["x_coords"] = torch.tensor(z["x_coords"], dtype=torch.float32)
    out["t_coords"] = torch.tensor(z["t_coords"], dtype=torch.float32)
    return out

def sample_supervision_points(cache: dict, n_points: int = 8192):
    T, X = cache["rho_grid"].shape
    idx_t = torch.randint(0, T, (n_points,))
    idx_x = torch.randint(0, X, (n_points,))
    xs = cache["x_coords"][idx_x]
    ts = cache["t_coords"][idx_t]
    xt = torch.stack([xs, ts], dim=1)
    q = cache["q_grid"][idx_t, idx_x]
    v = cache["v_grid"][idx_t, idx_x]
    rho_w = cache["rho_grid"][idx_t, idx_x]
    mask_q = torch.isfinite(q)
    mask_v = torch.isfinite(v)
    mask_rho = torch.isfinite(rho_w)
    return {"xt": xt, "q_obs": q, "v_obs": v, "rho_weak": rho_w, "mask_q": mask_q, "mask_v": mask_v, "mask_rho": mask_rho}

def huber(res, delta: float):
    absr = torch.abs(res)
    quad = torch.minimum(absr, torch.tensor(delta, device=res.device))
    lin = absr - quad
    return 0.5 * quad**2 + delta * lin

def main(config_path: str):
    dde.backend.set_default_backend("pytorch")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    wrapper = ARZPINNWrapper(cfg)
    geomtime = build_geometry(cfg)

    def pde_func(x, y):
        return wrapper.pde(x, y)

    ic_rho = dde.IC(geomtime, lambda x: np.zeros((1,)), wrapper.ic_condition, component=0)
    ic_v = dde.IC(geomtime, lambda x: np.zeros((1,)), wrapper.ic_condition, component=1)
    bc_in_rho = dde.DirichletBC(geomtime, lambda x: np.zeros((1,)), wrapper.boundary_inlet, component=0)
    bc_out_v = dde.NeumannBC(geomtime, lambda x: np.zeros((1,)), wrapper.boundary_outlet, component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        [ic_rho, ic_v, bc_in_rho, bc_out_v],
        num_domain=cfg["train"]["batch_collocation"],
        num_boundary=200,
        num_initial=200,
    )

    net = wrapper.net
    model = dde.Model(data, net)

    cache_path = os.path.join(cfg["data"]["cache_dir"], "grid_cache.npz")
    sup_cache = load_ngsim_cache(cache_path) if os.path.exists(cache_path) else None
    sup_points = sample_supervision_points(sup_cache, n_points=8192) if sup_cache else None

    W = cfg["loss_weights"]
    def additional_losses():
        l_smooth = wrapper.custom_losses() * W["w_mono"]
        l_sup = torch.tensor(0.0)
        if sup_points is not None:
            xt = sup_points["xt"]
            yhat = model.net(xt.numpy())
            yhat = torch.tensor(yhat, dtype=torch.float32)
            rho_hat = torch.sigmoid(yhat[:, 0]) * wrapper.rho_jam
            v_hat = torch.sigmoid(yhat[:, 1]) * wrapper.v_max
            q_hat = rho_hat * v_hat

            if "q_obs" in sup_points:
                lq = huber(q_hat[sup_points["mask_q"]] - sup_points["q_obs"][sup_points["mask_q"]], W["huber_delta"]).mean()
                l_sup = l_sup + W["w_data_q"] * lq
            if "v_obs" in sup_points:
                lv = huber(v_hat[sup_points["mask_v"]] - sup_points["v_obs"][sup_points["mask_v"]], W["huber_delta"]).mean()
                l_sup = l_sup + W["w_data_v"] * lv
            if "rho_weak" in sup_points:
                lr = huber(rho_hat[sup_points["mask_rho"]] - sup_points["rho_weak"][sup_points["mask_rho"]], W["huber_delta"]).mean()
                l_sup = l_sup + W["w_data_rho_weak"] * lr
        return l_smooth + l_sup

    model.compile(
        "adam",
        lr=cfg["train"]["lr"],
        loss_weights=[W["w_mass"], W["w_mom"]],
        external_trainable_variables=wrapper.external_parameters(),
        additional_loss=additional_losses,
    )

    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    for phase in cfg["train"]["phases"]:
        wrapper.freeze_Ve(freeze=not phase["learn_Ve"])
        callbacks = [dde.callbacks.ModelCheckpoint(os.path.join(ckpt_dir, f"{phase['name']}"), save_better_only=False)]
        if phase["use_rar"]:
            callbacks.append(dde.callbacks.PDEPointResampler(period=cfg["train"]["rar_every"]))
        model.train(epochs=phase["epochs"], callbacks=callbacks)

    model.save(os.path.join(ckpt_dir, "final_model"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
"""