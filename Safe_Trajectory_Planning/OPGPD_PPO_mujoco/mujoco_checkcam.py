import safety_gymnasium
import mujoco


def _find_model(obj, max_depth=4):
    seen = set()
    queue = [(obj, "env")]
    while queue:
        cur, path = queue.pop(0)
        if cur is None or id(cur) in seen:
            continue
        seen.add(id(cur))
        model = getattr(cur, "model", None)
        if model is not None:
            return model, path + ".model"
        sim = getattr(cur, "sim", None)
        if sim is not None and getattr(sim, "model", None) is not None:
            return sim.model, path + ".sim.model"
        if max_depth <= 0:
            continue
        for attr in ("env", "_env", "unwrapped", "task", "_task", "world", "_world", "builder", "_builder"):
            if hasattr(cur, attr):
                queue.append((getattr(cur, attr), f"{path}.{attr}"))
        max_depth -= 1
    return None, None


env = safety_gymnasium.make("SafetyPointGoal1-v0", render_mode="rgb_array")
model, path = _find_model(env)
if model is None:
    print("Could not locate MuJoCo model. Env type:", type(env))
    print("Unwrapped type:", type(env.unwrapped))
    print("Unwrapped attrs:", [a for a in dir(env.unwrapped) if "model" in a or "sim" in a or "world" in a])
    raise AttributeError("Could not locate MuJoCo model on env; check Safety-Gymnasium wrapper chain.")
print("Found model via", path)

print("ncam =", model.ncam)
for cam_id in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
    print(f"{cam_id}: {name}")

env.close()
