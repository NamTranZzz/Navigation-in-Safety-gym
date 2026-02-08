
import argparse
import time
from typing import Optional, List

import numpy as np

# pip install safety-gymnasium mujoco gymnasium matplotlib
import safety_gymnasium


def try_get_geom_positions(env, name_contains: Optional[List[str]] = None):
    """
    Best-effort: extract MuJoCo geom names and positions from env.
    Works if env exposes env.unwrapped.model and env.unwrapped.data (MuJoCo).
    """
    try:
        model = env.unwrapped.model
        data = env.unwrapped.data
    except Exception:
        return None

    # MuJoCo: model.ngeom, data.geom_xpos (ngeom, 3)
    try:
        ngeom = model.ngeom
        xpos = np.array(data.geom_xpos)  # (ngeom, 3)
        out = []
        for i in range(ngeom):
            # name retrieval differs slightly across mujoco versions
            try:
                name = model.geom(i).name
            except Exception:
                try:
                    name = model.names[model.name_geomadr[i] :].split(b"\x00", 1)[0].decode()
                except Exception:
                    name = f"geom_{i}"

            if name_contains is not None:
                if not any(k in name for k in name_contains):
                    continue

            out.append((name, xpos[i].copy()))
        return out
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="SafetyPointButton1-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--render_mode", type=str, default="human", choices=["human", "rgb_array"])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--policy", type=str, default="random", choices=["random", "small_random"])
    p.add_argument("--print_geoms_every", type=int, default=0,
                   help="0 disables. Otherwise print filtered geom positions every N steps.")
    p.add_argument("--geom_filter", type=str, default="hazard,gremlin,button,wall",
                   help="Comma-separated substrings to filter geoms to print (best-effort).")
    args = p.parse_args()

    # Create env. For rendering, Gymnasium expects render_mode set at make() time. :contentReference[oaicite:1]{index=1}
    env = safety_gymnasium.make(args.env_id, render_mode=args.render_mode)

    print("env_id:", args.env_id)
    print("action_space:", env.action_space)
    print("obs_space:", env.observation_space)

    # Optional: if you want a specific camera, some Safety-Gymnasium envs accept camera_name via env.render
    # (e.g., camera_name="human"). :contentReference[oaicite:2]{index=2}

    obs, info = env.reset(seed=args.seed)

    # If using rgb_array, show frames in matplotlib
    use_matplotlib = (args.render_mode == "rgb_array")
    if use_matplotlib:
        import matplotlib.pyplot as plt
        plt.ion()
        frame = env.render()  # returns an RGB frame array in rgb_array mode :contentReference[oaicite:3]{index=3}
        im = plt.imshow(frame)
        plt.title(args.env_id)
        plt.axis("off")

    geom_keys = [s.strip() for s in args.geom_filter.split(",") if s.strip()]

    dt = 1.0 / max(1e-6, args.fps)
    for t in range(args.steps):
        # --- Choose action ---
        if args.policy == "random":
            action = env.action_space.sample()
        else:
            # small_random: jitter around 0 so the point robot drifts around
            # (useful if action_space sample is too aggressive)
            low = env.action_space.low
            high = env.action_space.high
            action = np.random.uniform(low, high).astype(np.float32)
            action *= 0.25

        # Safety-Gymnasium step signature includes an extra `cost` return. :contentReference[oaicite:4]{index=4}
        obs, reward, cost, terminated, truncated, info = env.step(action)

        # --- Render ---
        if use_matplotlib:
            frame = env.render()
            im.set_data(frame)
            import matplotlib.pyplot as plt
            plt.pause(0.001)
        else:
            # In human mode the window updates continuously (env.render() returns None). :contentReference[oaicite:5]{index=5}
            env.render()

        # --- Optional: print obstacles/map elements positions (best-effort) ---
        if args.print_geoms_every and (t % args.print_geoms_every == 0):
            geoms = try_get_geom_positions(env, name_contains=geom_keys)
            if geoms is None:
                print("[geom dump] MuJoCo geom extraction not available in this install.")
            else:
                print(f"\n[geom dump @ step {t}] (filtered by {geom_keys})")
                for name, pos in geoms[:50]:
                    print(f"  {name:25s}  pos={pos}")

        # Slow down so it’s watchable
        time.sleep(dt)

        # --- Episode reset ---
        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("done")


if __name__ == "__main__":
    main()