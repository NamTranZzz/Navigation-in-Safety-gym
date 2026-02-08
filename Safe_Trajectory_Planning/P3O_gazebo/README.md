P3O Gazebo Point Navigation

Files:
- gazebo_env.py              Gazebo point navigation wrapper (Gymnasium)
- cmdp_wrapper.py            reward/cost wrapper (uses env reward + cost)
- p3o_agent.py               P3O implementation (fixed kappa=20, PPO-style clipping + multi-constraint penalty)
- train_p3o_pointnav.py      runnable training entrypoint + checkpointing + metric plots
- eval_p3o_pointnav.py       evaluation entrypoint (multiple seeds, optional parallel, optional video)
- config_pointnav_p3o.json   config for current defaults

Dependencies:
- pip install gymnasium
- Gazebo + your Gymnasium env package that provides the env_id
- (optional for video) pip install imageio imageio-ffmpeg

Quick start (fast sanity run):
  python train_p3o_pointnav.py --config config_pointnav_p3o.json --epochs 3 --steps_per_epoch 4000

Evaluate a checkpoint:
  python eval_p3o_pointnav.py --config config_pointnav_p3o.json --checkpoint runs_pointnav/ckpt_epoch_001.pt --num_episodes 10 --parallel 4

Save a video rollout:
  python eval_p3o_pointnav.py --config config_pointnav_p3o.json --checkpoint runs_pointnav/ckpt_epoch_001.pt --save_mp4

Notes:
- Set `env.env_id` in `config_pointnav_p3o.json` to your Gazebo Gymnasium env ID.
- Use `env.env_kwargs` to pass extra args to `gym.make(...)`.
