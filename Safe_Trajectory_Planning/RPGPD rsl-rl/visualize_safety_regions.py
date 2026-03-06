"""
Visualize obstacle safety regions for DynObs maps.

Usage:
  python visualize_safety_regions.py \
    --config config_pointnav_RPGPD_DynObs.json \
    --seed 0 \
    --out safety_regions.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# Ensure local repo code is imported before any site-packages copy.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mujoco_env import MujocoPointNavConfig, MujocoPointNavEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config_pointnav_RPGPD_DynObs.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="safety_regions.png")
    p.add_argument("--show", action="store_true", help="Show interactive figure")
    return p.parse_args()


def load_config(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _xy_from_arr(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[:2].copy()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    env_cfg = MujocoPointNavConfig(
        env_id=str(cfg["env"]["env_id"]),
        max_steps=int(cfg["env"].get("max_steps", 1000)),
        action_scale=float(cfg["env"].get("action_scale", 1.0)),
        env_config_overrides=dict(cfg["env"].get("env_config_overrides", {})),
        env_kwargs=dict(cfg["env"].get("env_kwargs", {})),
    )

    env = MujocoPointNavEnv(env_cfg, render_mode=None)
    try:
        env.reset(seed=int(args.seed))
        builder = env._env.unwrapped  # pylint: disable=protected-access
        task = builder.task

        xmin, ymin, xmax, ymax = [float(v) for v in task.placements_conf.extents]

        pillar_xy = np.asarray([_xy_from_arr(p) for p in task.pillars.pos], dtype=np.float64)
        gremlin_xy = np.asarray([_xy_from_arr(p) for p in task.gremlin_vels.pos], dtype=np.float64)
        start_xy = _xy_from_arr(task.agent.pos)
        goal_xy = _xy_from_arr(task.goal.pos)

        pillar_body_r = float(task.pillars.size)
        gremlin_body_r = float(task.gremlin_vels.size)
        # Additional safety margin used by DynObsLevel3.calculate_cost.
        safe_radius = float(getattr(task, "safe_radius", 0.0))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{env_cfg.env_id} safety regions (seed={args.seed})")
        ax.grid(True, alpha=0.25)

        # Plot start/goal.
        ax.scatter([start_xy[0]], [start_xy[1]], marker="o", color="tab:green", s=70, label="Start")
        ax.scatter([goal_xy[0]], [goal_xy[1]], marker="*", color="gold", edgecolor="black", s=120, label="Goal")

        # Pillars: body + cost-safe-radius.
        for i, p in enumerate(pillar_xy):
            body = Circle((p[0], p[1]), pillar_body_r, edgecolor="tab:blue", facecolor="tab:blue", alpha=0.45)
            if safe_radius > 0.0:
                safe = Circle(
                    (p[0], p[1]),
                    pillar_body_r + safe_radius,
                    edgecolor="tab:orange",
                    facecolor="none",
                    linewidth=1.6,
                    linestyle="-",
                )
                ax.add_patch(safe)
            ax.add_patch(body)
            if i == 0:
                if safe_radius > 0.0:
                    safe.set_label("Cost boundary (body + safe_radius)")
                body.set_label("Pillar body")

        # Gremlins: body + cost-safe-radius.
        for i, g in enumerate(gremlin_xy):
            body = Circle((g[0], g[1]), gremlin_body_r, edgecolor="tab:red", facecolor="tab:red", alpha=0.45)
            if safe_radius > 0.0:
                safe = Circle(
                    (g[0], g[1]),
                    gremlin_body_r + safe_radius,
                    edgecolor="tab:orange",
                    facecolor="none",
                    linewidth=1.6,
                    linestyle="-",
                )
                ax.add_patch(safe)
            ax.add_patch(body)
            if i == 0:
                body.set_label("Gremlin body")

        ax.legend(loc="best")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        print(f"[Saved] {out.resolve()}")

        if args.show:
            plt.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()
