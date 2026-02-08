P3O MuJoCo Point Navigation

Files:
- mujoco_env.py              Safety-Gymnasium point navigation wrapper (MuJoCo)
- cmdp_wrapper.py            reward/cost wrapper (uses env reward + cost)
- p3o_agent.py               P3O implementation (fixed kappa=20, PPO-style clipping + multi-constraint penalty)
- train_p3o_pointnav.py      runnable training entrypoint + checkpointing + metric plots
- eval_p3o_pointnav.py       evaluation entrypoint (multiple seeds, optional parallel, optional video)
- config_pointnav_p3o.json   config for current defaults

Dependencies:
- pip install safety-gymnasium mujoco gymnasium
- (optional for video) pip install imageio imageio-ffmpeg

Color overrides:
- Set `env.color_overrides` in `config_pointnav_p3o.json` with substrings mapped to RGBA.
- Example keys: `goal`, `pillar`, `agent`, `robot`, `hazard`, `wall`, `obstacle`.

Environment overrides:
- Set `env.env_config_overrides` in `config_pointnav_p3o.json` to pass a config dict into `safety_gymnasium.make(...)`.
- Use this to change counts like hazards/obstacles/pillars if the env exposes those keys.
- To discover valid keys for an env: `python -c "import safety_gymnasium as sg; e=sg.make('SafetyPointGoal1-v0'); print(e.unwrapped.config); e.close()"`.

Quick start (fast sanity run):
  python train_p3o_pointnav.py --config config_pointnav_p3o.json --epochs 3 --steps_per_epoch 4000

Paper-default scale (very large):
   python train_p3o_pointnav.py \
  --config config_pointnav_p3o.json \
  --epochs 100 \
  --num_roll_out 15 \
  --max_ep_len 1000 \
  --rollout_parallel 16 \
  --checkpoint_every 20 \
  --device mps \
  --eval_episodes 5 \
  --resume_checkpoint runs_pointnav/ckpt_20260122_140002_epoch_300.pt \
  --live

python train_p3o_pointnav.py \
  --config config_pointnav_p3o.json \
  --epochs 100 \
  --num_roll_out 15 \
  --max_ep_len 1000 \
  --rollout_parallel 16 \
  --checkpoint_every 20 \
  --device mps \
  --eval_episodes 5 \
  --init_checkpoint runs_pointnav/ckpt_20260122_140002_epoch_300.pt


Evaluate a checkpoint:
  python eval_p3o_pointnav.py --config config_pointnav_p3o.json --checkpoint runs_pointnav/ckpt_epoch_001.pt --num_episodes 10 --parallel 4

Save a video rollout:
  python eval_p3o_pointnav.py --config config_pointnav_p3o.json \
  --checkpoint runs_pointnav/ckpt_20260122_140002_epoch_300.pt \
  --num_episodes 5 \
  --save_mp4_each --render_width 1920 --render_height 1080 \
  --camera_id 1 


python eval_p3o_pointnav.py --config config_pointnav_p3o.json \
  --checkpoint runs_pointnav/ckpt_20260123_171154_epoch_020.pt \
  --num_episodes 5 \
  --save_mp4_each --render_width 1920 --render_height 1080 \
  --camera_id 1




