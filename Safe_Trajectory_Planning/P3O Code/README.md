p3o_nav2d

Files:
- nav_env.py          2D navigation dynamics-only environment (static pillars, static hazards, dynamic obstacles)
- cmdp_wrapper.py     reward/cost wrapper (goal-directed reward, distance-based obstacle cost)
- p3o_agent.py        P3O implementation (fixed kappa=20, PPO-style clipping + multi-constraint penalty)
- train_p3o_nav2d.py  runnable training entrypoint + checkpointing + metric plots
- eval_p3o_nav2d.py   evaluation entrypoint (multiple seeds, optional parallel, optional animation)
- config_nav_p3o.json config for current defaults

Quick start (fast sanity run):
  python train_p3o_nav2d.py --config config_nav_p3o.json --epochs 3 --steps_per_epoch 4000

Paper-default scale (very large):
  python train_p3o_nav2d.py --config config_nav_p3o.json --total_steps 10000000

Evaluate a checkpoint:
  python eval_p3o_nav2d.py --config config_nav_p3o.json --checkpoint runs_nav2d/ckpt_epoch_001.pt --num_episodes 10 --parallel 4

Notes:
- This code intentionally avoids gym/gymnasium dependencies for portability.
- If you want a gymnasium.Env-compatible wrapper later, we can add that with minimal changes.


Notes from Nam:
python train_p3o_nav2d.py --config config_nav_p3o.json --epochs 10 --steps_per_epoch 4000 --eval_episodes 100 --eval_parallel 4 --checkpoint_every 5 --live

python P3O Code/train_p3o_nav2d.py --config P3O Code/config_nav_p3o.json --epochs 20 --checkpoint_every 5 --live


 python train_p3o_nav2d.py \
  --config config_nav_p3o.json \
  --epochs 1000 \
  --num_roll_out 15 \
  --max_ep_len 1000 \
  --rollout_parallel 16 \
  --checkpoint_every 20 \
  --device mps \
  --eval_episodes 5


python eval_p3o_nav2d.py \
  --config config_nav_p3o.json \
  --checkpoint runs_nav2d/ckpt_20260113_022622_epoch_1000.pt \
  --num_episodes 5 \
  --max_steps 1000 \
  --save_mp4_all
