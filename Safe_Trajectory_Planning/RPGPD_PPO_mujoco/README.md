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
   

python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD.json \
  --epochs 100 \
  --num_roll_out 15 \
  --max_ep_len 1000 \
  --rollout_parallel 16 \
  --checkpoint_every 20 \
  --device mps \
  --eval_episodes 5 \
  --init_checkpoint runs_pointnav/ckpt_20260122_140002_epoch_300.pt


python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 100 \
  --num_roll_out 1 \
  --max_ep_len 100 \
  --rollout_parallel 16 \
  --checkpoint_every 20 \
  --device mps \
  --eval_episodes 5 \
  --live


  python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 300 \
  --num_roll_out 15 \
  --max_ep_len 1000 \
  --rollout_parallel 15 \
  --checkpoint_every 20 \
  --device mps \
  --eval_episodes 5 

   python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 200 \
  --num_roll_out 120 \
  --max_ep_len 1000 \
  --rollout_parallel 15 \
  --checkpoint_every 2 \
  --device mps \
  --eval_episodes 5 \
  --init_checkpoint runs_pointnav_rpgpd/ckpt_20260221_045957_epoch_112.pt


python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 300 \
  --num_roll_out 120 \
  --max_ep_len 1000 \
  --rollout_parallel 15 \
  --checkpoint_every 1 \
  --device mps \
  --eval_episodes 5 \
  --init_checkpoint runs_pointnav_rpgpd/ckpt_20260221_113542_epoch_052.pt


python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 300 \
  --num_roll_out 120 \
  --max_ep_len 1000 \
  --rollout_parallel 15 \
  --checkpoint_every 2 \
  --device mps \
  --eval_episodes 5 \
  --resume_checkpoint runs_pointnav_rpgpd/run_20260222_131743/ckpt_20260222_135627_epoch_020.pt


python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 300 \
  --num_roll_out 30 \
  --max_ep_len 1000\
  --rollout_parallel 15 \
  --checkpoint_every 1 \
  --device mps \
  --eval_episodes 5 \
  --init_checkpoint runs_pointnav_rpgpd/ckpt_20260221_113542_epoch_052.pt \
  --fixed_rollout_seed_range "1-120" 

python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 150 \
  --num_roll_out 30 \
  --max_ep_len 1000\
  --rollout_parallel 15 \
  --checkpoint_every 1 \
  --device mps \
  --eval_episodes 5 \
  --init_checkpoint runs_pointnav_rpgpd/ckpt_20260221_113542_epoch_052.pt \
  --fixed_rollout_seed_range "1-1" 

  python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 150 \
  --num_roll_out 60 \
  --max_ep_len 1000\
  --rollout_parallel 15 \
  --checkpoint_every 1 \
  --device mps \
  --eval_episodes 5 \
  --resume_checkpoint runs_pointnav_rpgpd/run_20260223_235742/ckpt_20260224_001417_epoch_015.pt
  --fixed_rollout_seed_range "1-1" 

python train_rpgpdppo_pointnav.py \
  --config config_pointnav_RPGPD_DynObs.json \
  --epochs 150 \
  --num_roll_out 6 \
  --max_ep_len 500\
  --rollout_parallel 6 \
  --checkpoint_every 1 \
  --device mps \
  --eval_episodes 5 \
  --fixed_rollout_seed_range "1-1" 


Evaluate a checkpoint:
  python eval_p3o_pointnav.py --config config_pointnav_p3o.json --checkpoint runs_pointnav/ckpt_epoch_001.pt --num_episodes 10 --parallel 4

Save a video rollout:
  python eval_p3o_pointnav.py --config config_pointnav_p3o.json \
  --checkpoint runs_pointnav/ckpt_20260122_140002_epoch_300.pt \
  --num_episodes 5 \
  --save_mp4_each --render_width 1920 --render_height 1080 \
  --camera_id 1 



python eval_rpgpdppo_pointnav.py --config config_pointnav_RPGPD_DynObs.json \
  --checkpoint runs_pointnav_rpgpd/ckpt_20260221_113542_epoch_052.pt \
  --num_episodes 1 \
  --save_mp4_all --render_width 1920 --render_height 1080 \
  --camera_id 1


python eval_rpgpdppo_pointnav.py --config config_pointnav_RPGPD_DynObs.json \
  --checkpoint runs_pointnav_rpgpd/run_20260221_163806/ckpt_20260221_172754_epoch_030.pt \
  --num_episodes 3 \
  --random_eval_seeds \
  --save_mp4_all --render_width 1920 --render_height 1080 \
  --camera_id 1


python eval_rpgpdppo_pointnav_rejection.py --config config_pointnav_RPGPD_DynObs.json \
  --checkpoint runs_pointnav_rpgpd/run_20260222_192922/ckpt_20260222_195216_epoch_019.pt \
  --num_episodes 3 \
  --random_eval_seeds \
  --save_mp4_all --render_width 1920 --render_height 1080 \
  --fixed_rollout_seed_range "1-1"\
  --camera_id 1 \




python eval_rpgpdppo_pointnav.py --config config_pointnav_RPGPD_DynObs.json \
  --checkpoint runs_pointnav_rpgpd/ckpt_20260221_113542_epoch_052.pt \
  --num_episodes 1 \
  --save_mp4_all --render_width 1920 --render_height 1080 \
  --camera_id 1
  --overlay_unsafe_areas

