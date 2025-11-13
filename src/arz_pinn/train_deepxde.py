"""
三阶段 DeepXDE 训练脚本:
- PhaseA: 固定 Ve
- PhaseB: 解锁 Ve
- PhaseC: 启动 RAR
需要你在 TODO 位置接入真实数据监督 (通过 additional_loss)
"""
import os, yaml, random
import numpy as np
import torch
import deepxde as dde
from arz_pinn_deepxde import ARZPINNWrapper

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def build_geometry(cfg):
    x_min, x_max = cfg["data"]["x_min"], cfg["data"]["x_max"]
    t_min, t_max = 0.0, cfg["data"]["t_pred"]
    geom = dde.geometry.Interval(x_min, x_max)
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    return dde.geometry.GeometryXTime(geom, timedomain)

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

    wrapper = ARZPINNWrapper(cfg)

    # PDE & geometry
    geomtime = build_geometry(cfg)

    def pde_func(x, y):
        return wrapper.pde(x, y)

    # 初值 / 边界条件占位：用户需用实际数据替换
    ic = dde.IC(geomtime,
                lambda x: np.zeros((1,)),
                wrapper.ic_condition,
                component=0)  # rho ic
    ic_v = dde.IC(geomtime,
                  lambda x: np.zeros((1,)),
                  wrapper.ic_condition,
                  component=1)  # v ic

    bc_in_rho = dde.DirichletBC(geomtime,
                                lambda x: np.zeros((1,)),
                                wrapper.boundary_inlet,
                                component=0)
    bc_out_v = dde.NeumannBC(geomtime,
                             lambda x: np.zeros((1,)),
                             wrapper.boundary_outlet,
                             component=1)

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        [ic, ic_v, bc_in_rho, bc_out_v],
        num_domain=cfg["train"]["batch_collocation"],
        num_boundary=200,
        num_initial=200
    )

    net = wrapper.net
    model = dde.Model(data, net)

    # 自定义 loss：添加 Ve 平滑 & 数据监督 (TODO)
    def additional_losses():
        l_smooth = wrapper.custom_losses()
        # TODO: 你的轨迹/环检监督项（例如 q, v, weak ρ），可从缓存读入并对当前点插值
        return l_smooth

    model.compile("adam", lr=cfg["train"]["lr"], loss_weights=[cfg["loss_weights"]["w_mass"], cfg["loss_weights"]["w_mom"],]
                  external_trainable_variables=wrapper.parameters(),
                  additional_loss=additional_losses)

    ckpt_dir = cfg["logging"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    start_epoch = 0
    for phase in cfg["train"]["phases"]:
        learn_Ve = phase["learn_Ve"]
        use_rar = phase["use_rar"]
        wrapper.freeze_Ve(freeze=not learn_Ve)

        callbacks = [dde.callbacks.ModelCheckpoint(os.path.join(ckpt_dir, f"{phase['name']}"), save_better_only=False)]
        if use_rar:
            callbacks.append(dde.callbacks.PDEPointResampler(period=cfg["train"]["rar_every"]))

        model.train(epochs=phase["epochs"], callbacks=callbacks)
        start_epoch += phase["epochs"]

    # 最终保存
    model.save(os.path.join(ckpt_dir, "final_model"))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(args.config)
