# RPGPD (rsl-rl) for Safety-Gymnasium PointNav

This folder runs **RPGPD** on top of `rsl-rl`:
- `rsl_rl.runners.OnPolicyRunner` handles rollout/checkpoint lifecycle.
- A custom RPGPD hook overrides PPO update with primal-dual logic.

## Files
- `train_rpgpd_pointnav.py`: single-process training entrypoint.
- `train_rpgpd_pointnav_parallel.py`: subprocess-parallel env training.
- `train_parallel.py`: thin alias to `train_rpgpd_pointnav_parallel.py`.
- `rpgpd_agent.py`: vec-env adapter + RPGPD trainer hook.
- `rpgpd_agent_parallel.py`: parallel vec-env adapter.
- `eval_rpgpd_pointnav.py`: standard checkpoint evaluation.
- `eval_rpgpd_pointnav_rejection.py`: rejection evaluation (`EpCost0 <= cost_limits[0]`).
- `config_pointnav_RPGPD_*.json`: task configs.

## RPGPD + PPO objective implemented
- Primal (policy) update uses PPO clipping on Lagrangian advantage:
\[
A_L(s,a)=A_r(s,a)-\lambda^\top A_c(s,a)
\]
\[
\max_\theta\ \mathbb{E}\left[\min\left(r_t(\theta)A_L,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_L\right)\right]
\]
- Dual update (projected ascent + shrinkage):
\[
\lambda \leftarrow \Pi_{[0,\lambda_{\max}]}\left((1-\eta_\lambda\tau)\lambda+\eta_\lambda\cdot(J_c-d)\right)
\]

## Important design choices
- Reward/cost separation is enforced in training (no cost-based reward shaping).
- Training reward uses unshaped env reward.
- Separate reward and cost critics with separate GAE.
- Multi-constraint costs are supported from `cmdp.cost_limits`.

## Train
```bash
python train_rpgpd_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --device cpu \
  --num_envs 12 \
  --max_iterations 300 \
  --log_dir runs_pointnav_rpgpd_rsl

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_rpgpd_pointnav_parallel.py \
  --config config_pointnav_RPGPD_DynObs_Dense.json \
  --device cpu \
  --num_envs 20 \
  --max_iterations 300 \
  --log_dir runs_pointnav_rpgpd_rsl \
  --start_method spawn
```

## Evaluate
```bash
python eval_rpgpd_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --checkpoint runs_pointnav_rpgpd_rsl/run_YYYYMMDD_HHMMSS/model_299.pt \
  --num_episodes 10 \
  --device cpu
```

## Rejection Evaluate
```bash
python eval_rpgpd_pointnav_rejection.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --checkpoint runs_pointnav_rpgpd_rsl/run_YYYYMMDD_HHMMSS/model_299.pt \
  --num_episodes 10 \
  --max_attempts 1000 \
  --device cpu
```
