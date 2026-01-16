import argparse
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def moving_average(series: pd.Series, w: int = 10) -> pd.Series:
    if w <= 1:
        return series
    return series.rolling(window=w, min_periods=1).mean()


def exp_moving_average(series: pd.Series, alpha: float = 0.1) -> pd.Series:
    if alpha <= 0 or alpha > 1:
        return series
    return series.ewm(alpha=alpha, adjust=False).mean()


def resolve_csv_path(csv_or_dir: str) -> str:
    p = Path(csv_or_dir)
    if p.is_dir():
        candidate = p / "metrics.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"Directory given but no metrics.csv found: {candidate}")
        return str(candidate)
    if p.is_file():
        return str(p)
    raise FileNotFoundError(f"Path not found: {csv_or_dir}")


def pick_x(df: pd.DataFrame):
    if "env_steps" in df.columns:
        return df["env_steps"], "Environment steps"
    if "epoch" in df.columns:
        return df["epoch"], "Epoch"
    return df.index, "Index"


def pick_success_col(df: pd.DataFrame):
    for c in ["success_rate", "success_rate", "succ", "success"]:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _plot_series(ax, x, y: pd.Series, label: str, smooth: int, ema_alpha: float | None):
    ax.plot(x, y, label=label, linewidth=1.0, alpha=0.8)
    if smooth and smooth > 1:
        ax.plot(x, moving_average(y, smooth), label=f"{label} (MA{smooth})", linewidth=2.0)
    if ema_alpha is not None:
        ax.plot(x, exp_moving_average(y, ema_alpha), label=f"{label} (EMA a={ema_alpha})", linewidth=2.0)


def print_summary(df: pd.DataFrame):
    def fmt(v):
        try:
            return f"{float(v):.4f}"
        except Exception:
            return str(v)

    cols = [
        "avg_ep_return",
        "success_rate",
        "avg_final_goal_dist",
        "avg_min_goal_dist",
        "policy_loss",
        "value_loss",
    ]
    present = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not present:
        return

    for c in present:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        print(
            f"{c:20s}  start={fmt(s.iloc[0])}  end={fmt(s.iloc[-1])}  "
            f"min={fmt(s.min())}  max={fmt(s.max())}  mean={fmt(s.mean())}"
        )


def main(
    csv_or_dir: str,
    smooth: int = 10,
    ema_alpha: float | None = None,
    save_dir: str | None = None,
    show: bool = True,
    logy_loss: bool = False,
):
    csv_path = resolve_csv_path(csv_or_dir)
    df = pd.read_csv(csv_path)

    x, x_label = pick_x(df)
    run_name = Path(csv_path).parent.name

    print(f"Loaded: {csv_path}")
    print_summary(df)

    has_loss = ("loss" in df.columns) and df["loss"].notna().any() and np.isfinite(df["loss"].dropna()).any()
    nrows = 3 if has_loss else 2
    fig, axs = plt.subplots(nrows, 1, figsize=(10, 10) if nrows == 3 else (10, 7), sharex=True)

    success_col = pick_success_col(df)
    if success_col is not None:
        _plot_series(axs[0], x, df[success_col], "success_rate", smooth, ema_alpha)
        axs[0].set_ylabel("Success rate")
        axs[0].set_ylim(-0.05, 1.05)
        axs[0].set_title(f"Success ({run_name})")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()
    else:
        axs[0].set_title(f"Success ({run_name}) - no success column found")
        axs[0].grid(True, alpha=0.3)

    if "avg_min_goal_dist" in df.columns and df["avg_min_goal_dist"].notna().any():
        _plot_series(
            axs[1],
            x,
            df["avg_min_goal_dist"],
            "avg_distance_to_goal",
            smooth,
            ema_alpha,
        )
        axs[1].set_ylabel("L2 distance")
        axs[1].set_title("Average distance to goal (min over episode)")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()
    else:
        axs[1].set_title("Average distance to goal - avg_min_goal_dist not found")
        axs[1].grid(True, alpha=0.3)

    if has_loss:
        _plot_series(axs[2], x, df["loss"], "ppo_total_loss", smooth, ema_alpha)
        if logy_loss:
            axs[2].set_yscale("log")
        axs[2].set_ylabel("Loss")
        axs[2].set_title("Total PPO loss (policy + value_coef*value - entropy_coef*entropy)")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()

    axs[-1].set_xlabel(x_label)

    fig.tight_layout()

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path / f"{run_name}_success_dist_loss.png", dpi=150)
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to metrics.csv OR a run directory containing metrics.csv",
    )
    parser.add_argument("--smooth", type=int, default=10, help="Moving average window (use 1 for none)")
    parser.add_argument("--ema", type=float, default=None, help="EMA alpha; omit for none")
    parser.add_argument("--save", type=str, default=None, help="Directory to save PNG")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot window")
    parser.add_argument("--logy-loss", action="store_true", help="Log scale y for loss")

    args = parser.parse_args()

    main(
        args.csv,
        smooth=args.smooth,
        ema_alpha=args.ema,
        save_dir=args.save,
        show=not args.no_show,
        logy_loss=args.logy_loss,
    )
