# P3O (rsl-rl) for Safety-Gymnasium PointNav

This folder runs **P3O** using:
- `rsl_rl.runners.OnPolicyRunner` for rollout/checkpoint infrastructure.
- A custom P3O update hook that replaces PPO loss with the P3O objective.

## Files
- `train_p3o_pointnav.py`: training entrypoint.
- `p3o_agent.py`: vectorized env adapter + P3O update hook.
- `eval_p3o_pointnav.py`: standard checkpoint evaluation.
- `eval_p3o_pointnav_rejection.py`: rejection evaluation (accept only episodes with `EpCost0 <= cost_limits[0]`).
- `config_pointnav_P3O_*.json`: task-specific configs.

## P3O objective implemented
\[
\max_\theta\ L_R^{clip}(\theta) - \kappa\sum_i \mathrm{ReLU}(L_{C_i}^{clip}(\theta))
\]
with separate reward/cost critics and separate reward/cost GAE.

## Important design choices
- Reward-cost separation is enforced for training:
  - Training reward uses unshaped env reward.
- Number of cost constraints is inferred from env config (`cost_limits`) and handled generically.

## Train
```bash
python train_p3o_pointnav.py \
  --config config_pointnav_P3O_DynObs_dense.json \
  --device cpu \
  --num_envs 12 \
  --max_iterations 300 \
  --log_dir runs_pointnav_p3o_rsl

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_p3o_pointnav_parallel.py \
  --config config_pointnav_P3O_DynObs_Dense.json \
  --device cpu \
  --num_envs 20 \
  --max_iterations 300 \
  --log_dir runs_pointnav_p3o_rsl_parallel \
  --start_method spawn
```

## Evaluate
```bash
python eval_p3o_pointnav.py \
  --config config_pointnav_P3O_DynObs_dense.json \
  --checkpoint runs_pointnav_p3o_rsl_parallel/run_20260305_183616/model_299.pt \
  --num_episodes 10 \
  --device cpu \
  --camera_id 1 \
  --render_width 1920 \
  --render_height 1080 \
  --save_mp4_all runs_pointnav_ppo_rsl/eval_all_episodes.mp4 \
  --mp4_macro_block_size 1 \
  --mp4_crf 17 \
  --mp4_preset slow
```

## Rejection evaluate
```bash
python eval_p3o_pointnav_rejection.py \
  --config config_pointnav_P3O_DynObs.json \
  --checkpoint runs_pointnav_p3o_rsl/run_YYYYMMDD_HHMMSS/model_299.pt \
  --num_episodes 10 \
  --max_attempts 1000 \
  --device cpu
```
