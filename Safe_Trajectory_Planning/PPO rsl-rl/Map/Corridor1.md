# MuJoCo PointNav: Random S-Corridor Map Using 4 Pillars + 6 Rand_Gremlins (Code-Ready Spec)

This spec defines a **procedurally generated** 2D navigation task in MuJoCo. The robot must reach a goal while avoiding **safety barriers** around (1) **4 static pillars** and (2) **6 dynamic obstacles** called **`rand_gremlin`**. The pillar safety barriers are arranged to **force an S-shaped trajectory** from start to goal. Visualise safe boundies of obstacles as well.

The intended implementation is: sample geometry on reset → validate map → build MuJoCo model → run episode.

---

## 1) World and Time

### 1.1 Coordinates
- Agent and obstacles move in plane `(x,y)`; `z` fixed.
- World bounds: rectangle `x ∈ [0,W]`, `y ∈ [0,H]`.
- Defaults: `W=16.0`, `H=12.0` meters.

### 1.2 Boundary walls
- Solid outer walls on the rectangle boundary.
- Wall thickness `t_wall=0.10`, wall height `h_wall=1.0`.

### 1.3 Control step
- Control dt: `dt_ctrl = 0.05` (20 Hz) (or your choice).
- Max control steps: `T_max = 800`.

---

## 2) Entities

### 2.1 Agent
- Point robot / disc.
- Agent radius: `r_agent = 0.20`.
- Goal tolerance: `goal_tol = 0.40`.
- Episode success: `||agent_pos - goal|| ≤ goal_tol`.

### 2.2 Static pillars (N_pillar = 4)
- Cylinder geoms.
- Physical radius per pillar: `r_pillar ∈ [0.20, 0.35]` (sample independently).
- Height: `h_pillar = 1.0`.
- Each pillar has a **safety radius** `R_pillar = r_pillar + b_pillar`, with `b_pillar ∈ [0.25, 0.55]`.

### 2.3 Dynamic obstacles: `rand_gremlin` (N_gremlin = 6)
- Cylinder/disc bodies that move in plane.
- Physical radius: `r_grem ∈ [0.18, 0.30]` (sample independently).
- Speed: `v_grem ∈ [0.30, 0.90]` m/s (sample per gremlin per episode).
- Safety radius `R_grem = r_grem + b_grem`, with `b_grem ∈ [0.20, 0.45]`.

---

## 3) Safety Cost (CMDP cost signal)

At each **control step**, compute:

For every obstacle i (pillars + gremlins):
- Let `d_i = ||agent_pos - obs_i_pos||`.
- Violation indicator: `I_i = 1(d_i < R_i)`.

**Cost definition (per-step, required):**
- `cost_t = Σ_i I_i`

So if agent enters the safety barrier of any obstacle, cost increases by +1 **per step per obstacle** while inside.

Return `cost_t` in `info["cost"]` (or as separate signal for CMDP).

---

## 4) Forcing an S-Shaped Route Using 4 Pillars

We create an S-route by placing pillars as “blockers” that alternate which side the agent must pass through.

### 4.1 Choose an S pattern
Randomly pick one of two patterns with 50/50 probability:

- **Pattern A (Top → Bottom → Top)**  
- **Pattern B (Bottom → Top → Bottom)**

Define:
- `y_top = 0.78H + jitter_top`, `jitter_top ∈ [-0.05H, +0.05H]`
- `y_bot = 0.22H + jitter_bot`, `jitter_bot ∈ [-0.05H, +0.05H]`
- `y_mid = 0.50H + jitter_mid`, `jitter_mid ∈ [-0.06H, +0.06H]`

Pick three x-stations:
- `x1 ∈ [0.28W, 0.35W]`
- `x2 ∈ [0.47W, 0.53W]`
- `x3 ∈ [0.65W, 0.72W]`

Pick corridor width target:
- `cw ∈ [1.2, 2.0]`

### 4.2 Start and goal sampling (random but structured)
Start near left, goal near right. Side depends on S pattern.

If Pattern A (Top→Bottom→Top):
- Start: `x ∈ [0.8, 2.0]`, `y ∈ [0.65H, 0.90H]` (top-ish)
- Goal:  `x ∈ [W-2.0, W-0.8]`, `y ∈ [0.65H, 0.90H]` (top-ish)

If Pattern B (Bottom→Top→Bottom):
- Start: `x ∈ [0.8, 2.0]`, `y ∈ [0.10H, 0.35H]` (bottom-ish)
- Goal:  `x ∈ [W-2.0, W-0.8]`, `y ∈ [0.10H, 0.35H]` (bottom-ish)

Also require `||start-goal|| ≥ 0.60W`.

### 4.3 Pillar placement rule (4 pillars total)
Pillars are placed to create 3 “forcing events” along the x-axis:

- **Gate 1 at x1:** two pillars form a narrow gap around y_mid  
  (forces agent through center, creating first bend setup)
- **Gate 2 at x2:** one pillar blocks the “wrong” side  
  (forces lane switch)
- **Gate 3 at x3:** one pillar blocks the opposite side  
  (forces second lane switch)

#### Gate 1 (P1, P2)
Place P1 and P2 near x1 around y_mid with a safety-gap ≈ cw:

- Sample pillar radii and buffers first to get `R1`, `R2`.
- Choose a target separation along y:
  - `d_y = cw + R1 + R2`  (so safety discs leave a gap ~cw)
- Set:
  - `P1 = (x1 + u1, y_mid + d_y/2)`
  - `P2 = (x1 + u2, y_mid - d_y/2)`
  - `u1,u2 ∈ [-0.4, +0.4]` meters

#### Gate 2 (P3) at x2
Blocks the side that should NOT be taken between x2 and x3.

If Pattern A (Top→Bottom→Top):
- Between x2 and x3 agent should be **bottom**, so block **top**:
  - `P3 = (x2 + u3, y_top + v3)`
If Pattern B (Bottom→Top→Bottom):
- Between x2 and x3 agent should be **top**, so block **bottom**:
  - `P3 = (x2 + u3, y_bot + v3)`
Where `u3 ∈ [-0.5, +0.5]`, `v3 ∈ [-0.3, +0.3]`.

#### Gate 3 (P4) at x3
Blocks opposite side to force the final bend back.

If Pattern A (Top→Bottom→Top):
- After x3, agent should be **top**, so block **bottom**:
  - `P4 = (x3 + u4, y_bot + v4)`
If Pattern B (Bottom→Top→Bottom):
- After x3, agent should be **bottom**, so block **top**:
  - `P4 = (x3 + u4, y_top + v4)`
Where `u4 ∈ [-0.5, +0.5]`, `v4 ∈ [-0.3, +0.3]`.

### 4.4 Pillar constraints (rejection sampling)
After proposing pillar positions, reject unless:

1) In bounds with margin `m=0.6`:
- `m ≤ x ≤ W-m`, `m ≤ y ≤ H-m`

2) Pillars do not overlap physically:
- `dist(Pi,Pj) ≥ (r_i + r_j + 0.15)`

3) Pillars are not too close to start/goal:
- `dist(Pi,start) ≥ 2.0`
- `dist(Pi,goal) ≥ 2.0`

---

## 5) “Corridor Region” (where gremlins live)

Define a centerline polyline through 5 waypoints (S curve):
- `w0 = start`
- `w1 = (x1, y_mid)`
- `w2 = (x2, side2)` where:
  - Pattern A: `side2 = y_bot`
  - Pattern B: `side2 = y_top`
- `w3 = (x3, side3)` where:
  - Pattern A: `side3 = y_top`
  - Pattern B: `side3 = y_bot`
- `w4 = goal`

Define corridor radius:
- `corridor_radius = cw/2 + 0.6`

Define corridor set:
- `CORRIDOR = { p : distance_to_polyline(p, [w0..w4]) ≤ corridor_radius }`

Gremlins must be spawned in `CORRIDOR` and constrained to remain inside it.

---

## 6) rand_gremlin Motion: Random Walk with Holding Time (10–100 steps)

For each gremlin j maintain:
- position `p_j`
- heading `θ_j`
- holding counter `k_j`
- speed `v_j`

Initialization:
- `θ_j ~ Uniform(0,2π)`
- `k_j ~ UniformInt(10,100)`

Each control step:
1) If `k_j == 0`:
   - resample `θ_j ~ Uniform(0,2π)`
   - resample `k_j ~ UniformInt(10,100)`
2) Propose:
   - `p' = p + v_j * [cosθ_j, sinθ_j] * dt_ctrl`
3) If `p'` is outside `CORRIDOR` OR outside world bounds OR collides with a pillar:
   - resample `θ_j ~ Uniform(0,2π)`
   - set `k_j ~ UniformInt(5,20)` (short hold)
   - **do not move** this step (keep `p_j`)
4) Else:
   - set `p_j ← p'`
   - decrement `k_j -= 1`

Gremlins are collidable and can block the agent (kinematic or dynamic implementation is allowed).

---

## 7) Gremlin Spawn (random)

Spawn 6 gremlins by rejection sampling:

Sample candidate points `p`:
- Either uniformly in world until `p ∈ CORRIDOR`, or
- Sample along polyline arc-length with lateral offset.

Reject unless:
- `p ∈ CORRIDOR`
- `dist(p,start) ≥ 2.0`
- `dist(p,goal) ≥ 1.5`
- `dist(p, pillar_i) ≥ r_grem + r_pillar_i + 0.20` for all pillars
- `dist(p, gremlin_k) ≥ r_grem + r_grem_k + 0.20` for all already-placed gremlins

---

## 8) Map Validity Checks (required)

After sampling start/goal/pillars/gremlins, validate:

### 8.1 Connectivity
There must exist a path from start to goal avoiding **physical collisions**:
- blocked = pillars (physical radii) + boundary walls
- Use coarse grid A*:
  - resolution `res = 0.20` meters
  - inflate obstacles by `r_agent + 0.05`

### 8.2 S-shape enforcement (two bends)
Compute the A* path polyline and ensure it contains **at least 2 significant turns**:
- For consecutive segments, compute turning angle.
- Count turns where `angle ≥ 35°` (configurable).
- Require `turn_count ≥ 2`.

If any validity check fails, resample everything (up to `max_tries`).

---

## 9) Rewards and Termination (example; can be changed)

- Reward shaping (example):
  - `r_t = -0.01` per step
  - `+1.0` on success
- Termination:
  - success if within `goal_tol`
  - timeout if `t >= T_max`

Safety is **not** in reward; it’s in `cost_t`.

---

## 10) What Codex Should Implement

Functions / modules:

1) `generate_episode(seed, cfg)`:
   - sample pattern, cw, x1/x2/x3, y_top/y_bot/y_mid
   - sample start/goal
   - place 4 pillars via rules + rejection constraints
   - define corridor polyline and `is_in_corridor(p)`
   - place 6 gremlins via rules
   - run validity checks (A*, turn count)
   - return full episode spec

2) `distance_to_polyline(p, waypoints)` (for corridor constraint)

3) MuJoCo model builder:
   - boundary walls
   - pillars
   - agent
   - gremlins (kinematic or dynamic)

4) Step loop:
   - apply action to agent
   - update gremlins using holding-time random walk
   - compute reward, `cost_t`, done, info

---

## 11) Default Config (recommended)
- `W=16, H=12`
- `N_pillar=4`, `N_gremlin=6`
- `cw ∈ [1.2, 2.0]`
- `r_agent=0.20`
- `r_pillar ∈ [0.20, 0.35]`, `b_pillar ∈ [0.25, 0.55]`
- `r_grem ∈ [0.18, 0.30]`, `b_grem ∈ [0.20, 0.45]`
- `v_grem ∈ [0.30, 0.90]`
- holding steps `K ∈ [10, 100]`
- `goal_tol=0.40`, `T_max=800`
- cost = per-step sum of barrier intrusions
