# Go2 SuperMimic

Quadruped velocity learning on MJLab. Module boundaries follow
SuperMimic-Reproduced; learning behavior and task definitions follow
`quadruped_lab@bac831afa62cd1bb89e0d3c7837bf1766b149d5a`.
Neither reference repository is imported at runtime.

## Structure

Paths are relative to `src/tasks/supermimic/`.

| Module | Responsibility |
| --- | --- |
| `config/<robot>/env_cfg.py` | Native environment, actions, rewards, sensors and events |
| `config/<robot>/rl_cfg.py` | Robot learning defaults |
| `config/<robot>/__init__.py` | Register training and playback task configurations |
| `envs/state_io.py`, `envs/sim_context.py` | Simulator state and randomized physics |
| `envs/mdp/` | Quadruped observations, commands, rewards, terrain and termination |
| `world_model/state_spec.py` | State coordinates, physical residuals, integration and errors |
| `world_model/heads.py` | Analytic task advancement and replay event scores |
| `world_model/normalization.py` | Explicit running population statistics |
| `world_model/world_model.py` | Normalized autoregressive predictions and masked loss |
| `world_model/wm_trainer.py` | Model sampling and optimizer updates |
| `data/` | Actual collected histories, replay windows and rare-event archive |
| `policy/nets.py` | History MoE, actor and privileged critic |
| `imagination/` | Differentiable rollout, objective and actor/critic updates |
| `training/config.py`, `training/schedule.py` | Component composition and learning schedule |
| `training/loop.py` | Collection, updates, rollback, logging and checkpoints |
| `eval/`, `scripts/` | Playback, evaluation, deployment export and CLI entry points |

`TrainConfig` combines a native `env`, a `SuperMimicCfg` agent and run settings.
Environment parameters stay on their managers: history length belongs to
`env.observations["history"]`, action scales to `env.actions["joint_pos"]`, and
replay capacity/archive settings to `env.recorders["replay"].params`.
Each learning component declares its configuration beside its implementation.

New robot presets register native train/play configurations under `config/<robot>`.
The entry points discover them by their `SuperMimicCfg`; there is no robot
switch or second registry. Reusing the current task requires its quadruped state
and action conventions. Other morphologies require corresponding task heads,
state coordinates and policy dimensions.

## Robot and task

SuperMimic, loco and PIE share `src/assets/robots/unitree_go2/go2_constants.py`
and `xmls/go2.xml`. The XML owns geometry, inertias, joint limits and named
sites/sensors; native `EntityCfg` owns the initial state, actuators and contacts.
There is one robot preset. Initial height is 0.29 m, left/right hip angles are
+0.1/-0.1, thigh angle is 0.9 and calf angle is -1.8.
Hip/thigh/calf action scales are 0.25/0.5/0.5, gains are 20/20/40 and 1/1/2,
effort limits are 23.7/23.7/45.43, and armature is 0.01.
Actuator delays span zero to four physics steps, with hold probability 0.5 and
update period ten. Feet use condim 3, priority 1 and friction (1, 0.005, 0.0005);
other colliders use condim 1 and priority 0. Ground and self contacts are enabled.
All collision geoms use solref (0.01, 1). These shared control/contact defaults
also apply to loco and PIE; each task still defines its own action scale.

The geometry follows the public [Unitree mjlab Go2 asset](https://github.com/unitreerobotics/unitree_rl_mjlab/blob/main/src/assets/robots/unitree_go2/xmls/go2.xml),
with foot centers and sites at x = -0.002 m as in
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/blob/main/unitree_go2/go2.xml).
Both use cylinder head, hip and calf colliders. This intentionally replaces the
capsules in quadruped_lab's private variant, so physical trajectories differ.
Native actuator and collision configuration follows
[mjlab 1.6](https://github.com/mujocolab/mjlab/blob/v1.6.0/src/mjlab/asset_zoo/robots/unitree_go1/go1_constants.py).
Actuators are created once through mjlab; the XML adds no motors or passive joint
damping that would change the retained PD controller.

Joint order is FL, FR, RL, RR, with hip, thigh and calf per leg. Encoder
observations use `q - default + bias`; physical targets use
`default + scale * action - bias`.

Packed state contains 125 physical coordinates, 42 memory coordinates and three
commands. The 16 context coordinates contain randomized physics actually used by
the model. Actor frames have 47 coordinates, five-frame histories have 235, and
critic features have 262. The 21 reward terms, command feedback, contact timing,
terrain and curriculum retain the quadruped task definitions.

## Learning protocol

- Prefill replay before fitting the model, then alternate two and three real
  control steps per learning iteration (`agent.schedule.collect_steps=2.5`).
- Update the WM from iteration one: ten batches of 2048, with eight-step windows.
  The default network has three 1024-unit layers, ordinary output initialization,
  and input/output RunningNorm. Contact-logit output statistics remain mean zero
  and variance one. Loss averages normalized coordinate errors over valid
  transitions; reset transitions and subsequent positions are masked out.
- Start policy updates at iteration 301; ramp actor learning rate over 1000
  iterations. The eight-expert MoE uses separate history and current-frame
  RunningNorm. The critic uses RunningNorm and ELU.
- Store the actual collected noisy history in float16. Sample WM batches with
  25% priority windows and 25% archive windows when the archive is nonempty.
  Archive capacity is 100000 and keep probability is 0.02.
- Apply OU exploration to 25% of environments and independent Gaussian noise
  with standard deviation 0.1 to the remainder. Both random draws cover all
  environments before selecting the exploration rows.
- Use horizon 32, gamma 0.99 and lambda 0.95. Optimize the complete survival-weighted
  actor objective with terminal-value gradients. Critic targets use cumulative
  survival ratios; fixed targets precede statistics updates and environment-major
  minibatch flattening.
- Decay learning rates over 30000 learning iterations to a 0.03 ratio. The default
  target is 40000 iterations. Skip nonfinite gradient updates. Collapse rollback
  restores the saved policy and critic, retains the WM, halves learning rates and
  restarts the actor ramp.

Native managers own simulation, history and resets. Recorder hooks snapshot
terminal actions and flags before automatic reset clears them. Disabling
terminations also disables timeouts while retaining task-state refresh.
Checkpoint publication uses one collapse guard for periodic and final saves.
Training supports one process; distributed normalization and rollback are not
part of this protocol.

## Run

From the repository root:

```bash
.venv/bin/python src/tasks/supermimic/scripts/train.py go2-supermimic-flat \
  --device cuda:0 --env.scene.num-envs 1024 --run-name flat
.venv/bin/python src/tasks/supermimic/scripts/train.py go2-supermimic-rough \
  --device cuda:0 --run-name rough
.venv/bin/python -m src.tasks.supermimic.scripts.play go2-supermimic-rough \
  --checkpoint-file /absolute/path/ckpt_latest.pt --viewer viser
.venv/bin/python -m src.tasks.supermimic.scripts.export \
  --checkpoint-file /absolute/path/ckpt_latest.pt --output-dir /absolute/path/deploy
```

Component overrides use `--agent.world-model.*`, `--agent.wm-trainer.*`,
`--agent.policy-trainer.*` and `--agent.schedule.*`. Iterations count learning
updates, each with an average of 2.5 real control steps.

Resume with `--resume` and the same nondefault component settings. Same-format
resume restores weights, optimizers and progress; physical trajectories, replay,
terrain assignments, warmup and rollback state restart as in the baseline.
Checkpoints use `supermimic_quadruped_v3`. Earlier protocols have different
normalization, physics and replay semantics and are rejected at loading boundaries.
Development checkpoints that serialized the removed dedicated Go2 module require
their original source checkout. Start a new run for the shared collision geometry.

CLI playback uses the registered play configuration: repeat training overrides
that affect history length or action semantics. Programmatic
`make_playback(checkpoint_file)` uses the checkpoint's environment unless given an
explicit `env_cfg`. Export writes TorchScript, ONNX and JSON metadata with both
actor normalizers and action scales resolved in joint order.

Short deterministic comparisons and CPU smoke runs check behavior preservation;
they do not establish convergence of the default 40000-iteration run.
