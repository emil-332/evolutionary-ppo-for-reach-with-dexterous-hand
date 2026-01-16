import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def resolve_run_dir(run_or_summary: str) -> Path:
    p = Path(run_or_summary)
    if p.is_dir():
        sp = p / "summary.csv"
        if not sp.exists():
            raise FileNotFoundError(f"No summary.csv found in directory: {sp}")
        return p
    if p.is_file() and p.name == "summary.csv":
        return p.parent
    raise FileNotFoundError(f"Path not found or not a run dir / summary.csv: {run_or_summary}")


def load_individual_loss(run_dir: Path, gen: int, ind: int, mode: str = "last") -> float:
    """
    Read loss from results/eppo/<run>/gen_<g>/ind_<i>/metrics.csv.

    mode:
      "last": last row's loss
      "mean": mean loss over all rows
    """
    p = run_dir / f"gen_{gen}" / f"ind_{ind}" / "metrics.csv"
    if not p.exists():
        return float("nan")

    df = pd.read_csv(p)
    if "loss" not in df.columns or df["loss"].isna().all():
        return float("nan")

    if mode == "mean":
        return float(df["loss"].mean())

    return float(df["loss"].iloc[-1])


def main():
    ap = argparse.ArgumentParser(description="Plot EPPO summary.csv + (optional) aggregated PPO loss from per-individual metrics.csv")
    ap.add_argument("--run", required=True, help="Path to EPPO run dir")
    ap.add_argument("--save", default=None, help="Directory to save PNG")
    ap.add_argument("--no-show", action="store_true", help="Do not open plot window")

    # Loss options
    ap.add_argument("--no-loss", action="store_true")
    ap.add_argument("--loss-agg", choices=["all", "elite"], default="all",
                    help="Aggregate loss over all individuals or elites only")
    ap.add_argument("--loss-mode", choices=["last", "mean"], default="last")

    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run)
    summary_path = run_dir / "summary.csv"
    s = pd.read_csv(summary_path)

    required_cols = ["gen", "ind", "env_steps", "train_success_rate", "train_avg_min_goal_dist", "is_elite"]
    missing = [c for c in required_cols if c not in s.columns]
    if missing:
        raise ValueError(f"summary.csv missing required columns: {missing}")

    for c in ["env_steps", "train_success_rate", "train_avg_min_goal_dist", "is_elite"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")
    s["train_avg_min_goal_dist"] = s["train_avg_min_goal_dist"].replace([float("inf"), -float("inf")], float("nan"))

    g = s.groupby("gen")
    x_env_steps = g["env_steps"].max().to_numpy()

    y_succ = g["train_success_rate"].mean().to_numpy()
    y_dist = g["train_avg_min_goal_dist"].mean().to_numpy()

    y_loss = None
    if not args.no_loss:
        losses = []
        gens = sorted(s["gen"].unique())

        for g in gens:
            df_g = s[s["gen"] == g].copy()
            if args.loss_agg == "elite":
                df_g = df_g[df_g["is_elite"] == 1]

            loss_vals = []
            for _, r in df_g.iterrows():
                ind = int(r["ind"])
                loss_vals.append(load_individual_loss(run_dir, int(g), ind, mode=args.loss_mode))

            losses.append(pd.Series(loss_vals, dtype="float64").mean())

        y_loss = pd.Series(losses, dtype="float64")

    nrows = 2 if (args.no_loss or y_loss is None) else 3
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows), sharex=True)
    if nrows == 1:
        axs = [axs]

    axs[0].plot(x_env_steps, y_succ, linewidth=1.5)
    axs[0].set_ylabel("Success rate")
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].set_title(f"EPPO Training Success ({run_dir.name})")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(x_env_steps, y_dist, linewidth=1.5)
    axs[1].set_ylabel("L2 distance")
    axs[1].set_title("EPPO Training Avg Min Goal Distance (avg over individuals)")
    axs[1].grid(True, alpha=0.3)

    if nrows == 3 and y_loss is not None:
        axs[2].plot(x_env_steps, y_loss, linewidth=1.5)
        axs[2].set_ylabel("PPO total loss")
        axs[2].set_title(f"EPPO Training Loss (loss_agg={args.loss_agg}, loss_mode={args.loss_mode})")
        axs[2].grid(True, alpha=0.3)
        axs[2].set_xlabel("Environment steps (training only)")
    else:
        axs[1].set_xlabel("Environment steps (training only)")

    fig.tight_layout()

    if args.save is not None:
        out_dir = Path(args.save)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = "success_dist" if args.no_loss else f"success_dist_loss_{args.loss_agg}_{args.loss_mode}"
        out_path = out_dir / f"{run_dir.name}_eppo_{suffix}.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to: {out_path}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
