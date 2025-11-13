#!/usr/bin/env python
import argparse, os
from src.arz_pinn.data_prep_ngsim import load_raw_trajectories, edie_grid, build_ic_bc, save_npz

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw trajectories csv (vehicle_id,t,x,speed[,lane,length])")
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--dx", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--x-min", type=float, default=0.0)
    ap.add_argument("--x-max", type=float, required=True)
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = load_raw_trajectories(args.raw)
    grid = edie_grid(df, args.dx, args.dt, args.x_min, args.x_max, args.t_min, args.t_max)
    icbc = build_ic_bc(grid)
    save_npz(args.out, grid, icbc)
    print(f"Saved cache to {args.out}")

if __name__ == "__main__":
    main()