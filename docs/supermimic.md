# Go2 SuperMimic

Go2 velocity learning on MJLab, using the SuperMimic-Reproduced training framework.
Task coordinates, observations, rewards, command sampling, contact timing, and
terrain belong to the quadruped task. Replay, world-model optimization,
short-horizon policy learning, and scheduling follow the reference implementation.

## Structure

Paths below are relative to `src/tasks/supermimic/`.

| Module | Responsibility |
| --- | --- |
| `config/go2/env_cfg.py` | Native flat/rough Go2 environment presets and manager configuration |
| `config/go2/rl_cfg.py` | Go2 `SuperMimicCfg` learning defaults |
| `config/go2/__init__.py` | Native MJLab task registration for Go2 training and playback |
| `envs/state_io.py`, `envs/sim_context.py` | Simulator features and randomized context |
| `envs/mdp/` | Velocity commands, observations, rewards, termination, terrain and progression |
| `world_model/state_spec.py` | Go2 coordinates, tensor packing, canonical inputs, integration and error coordinates |
| `world_model/heads.py` | Analytic task advancement, observations and rewards |
| `world_model/dynamics.py` | Reference `MLP` and zero-output-initialized `DynamicsMLP` |
| `world_model/world_model.py` | Prediction, compiled loss step, autoregressive Eq.1 loss and diagnostics |
| `world_model/wm_trainer.py` | `WMTrainerCfg`, uniform replay sampling and Adam updates |
| `data/replay_buffer.py` | Episode-aware ring storage, complete windows and reconstructed histories |
| `data/recorder.py` | Native recorder hooks, clean/executed actions and terminal rows |
| `policy/nets.py` | Go2 history MoE, privileged critic, fixed physical input scales |
| `imagination/compiled.py` | One compiled task prediction/reward/observation step |
| `imagination/rollout.py` | Time-first differentiable trajectories and survival probabilities |
| `imagination/objective.py` | Reference policy objective and GAE targets |
| `imagination/policy_trainer.py` | Actor BPTT followed by fixed-target critic minibatches |
| `training/loop.py` | One real environment step, fractional UTD updates, summaries and checkpoints |
| `training/config.py` | Generic `TrainConfig` environment/agent composition and `SuperMimicCfg` |
| `training/schedule.py`, `training/distributed.py` | Reference cosine schedule and explicit gradient averaging |
| `eval/`, `scripts/` | Playback, export, velocity evaluation and thin command entry points |

`TrainConfig` combines the native `env` configuration with an `agent`
(`SuperMimicCfg`) and run settings. The agent owns `world_model`, `wm_trainer`,
`policy`, `critic`, `policy_trainer`, `schedule` and action exploration noise.
Each learning component declares its own configuration beside its implementation.
Training, evaluation and command entry points use the registered configurations
without importing a robot-specific preset.

Robot settings live in `src/assets/robots/unitree_go2/go2_constants.py` as native
MJLab actuator, articulation, collision and initial-state constants.
`get_go2_robot_cfg()` returns an independent entity with initial height 0.32 m,
hip/thigh/calf gains 20/20/40, damping 1/1/2 and effort limits 23.5/23.5/45.
Every actuator group uses delay lags 0–4, hold probability 0.5 and update period 10.
Robot collisions use `contype=1`, `conaffinity=0` and foot friction 0.6 before task
startup randomization. MJLab 1.6 loads meshes directly through `MjSpec.from_file()`.

The velocity task defines `JointPositionActionCfg(scale=0.25)` directly in its
environment configuration. Deployment export reads that actual action term.
Camera, simulation and event values are also configured there.

`flat_env_cfg()` and `rough_env_cfg()` return `ManagerBasedRlEnvCfg` directly;
training constructs `ManagerBasedRlEnv` with the selected preset. Native command,
observation, reward, termination, event and recorder managers own the task lifecycle.
The velocity command keeps deterministic task memory; the observation manager owns
frame history and partial-reset backfill. Terminal features are refreshed before
scoring and captured by recorder hooks before automatic resets. Playback uses the
same environment and native viewer controls.

`Physical`, `Memory` and `Context` are named tensor views obtained with `split()`.
There are no field registries, layout dictionaries or global slice aliases.
`StateSpec` handles packed state, canonical features, integration and prediction
errors; it does not initialize robot metadata, history or checkpoint identity.
The packed state retains 125 physical features, 42 memory features and three
commands. Context contains only the 16 randomized physics parameters used by the
model: friction, encoder bias and COM offset. Terrain origins and standing
probability are no longer copied into context, and command ranges come directly
from environment settings. Observation and critic widths remain 47 and 262.

Configuration follows one path: `TrainConfig.env` is the native environment
configuration passed to `ManagerBasedRlEnv`. Scene counts, simulation settings,
actions, rewards and recorder settings use their corresponding manager fields.
Command sampling, task noise and reward-shape parameters belong to
`env.commands["velocity"]`, a `VelocityCommandCfg`; reward weights belong to
`env.rewards`. The policy and recorder read history length from
`env.observations["history"].history_length`. Replay capacity is configured as
`env.recorders["replay"].params["capacity"]`, and training exploration noise as
`TrainConfig.agent.action_noise_std`. Action scale comes directly from
`env.actions["joint_pos"].scale`. The MoE experts, gates, history encoding and actor
computations remain unchanged.

## Task registration

Go2 presets use MJLab's native registry with task IDs `go2-supermimic-flat` and
`go2-supermimic-rough`. Each ID registers training and playback environment
configurations plus its `SuperMimicCfg`. The SuperMimic train/play entry points
select registry entries with that learning configuration and run the SuperMimic
trainer. MJLab's RSL training entry point uses its own learning runner.

A robot preset follows three files under `config/<robot>/`:

- `env_cfg.py` constructs native `ManagerBasedRlEnvCfg` presets.
- `rl_cfg.py` provides a `supermimic_cfg()` learning-settings factory.
- `__init__.py` registers those environments and learning settings with
  `register_mjlab_task`.

Repository task discovery imports these packages; there is no central robot
dispatcher to extend. Another robot that shares the current quadruped state and
action conventions can reuse the learning framework through its presets. A new
morphology also needs matching state coordinates, analytic task terms and policy
dimensions.

## Reference training protocol

Each iteration collects one vectorized environment step. After warmup,
`updates_per_env_step` accumulates the update budget; every complete budget unit
runs one WM update and the configured number of policy updates. Defaults match
the reference: UTD 0.4, warmup 40 environment steps, ten WM minibatches per update,
eight-step WM windows, horizon 32, gamma 0.99 and GAE lambda 0.95.

The ring buffer uses `[environment,time]` storage, per-environment pointers,
episode identities and explicit terminal rows. Windows never cross resets or the
write pointer. A timeout is a terminal storage boundary with termination label
zero. Sampled windows and imagined trajectories use time-first tensors. Policy
history is reconstructed from logged clean states, with episode-start backfill
and fresh observation noise. There is no stored half-precision noisy history,
prioritized/archive sampler or OU exploration group.

The WM predicts task-defined residual coordinates. Its loss is
`sum_t mean_batch(sum_features(error**2)) / K`, matching the reference Eq.1.
The output layer starts at zero. Input/context/error coordinates use fixed task
scales, replacing running normalization. Supervised Go2 terrain relief retains
its explicit half-to-float rounding and recorded-pose inputs; recorded task memory
and commands are fed back between supervised steps.

Policy learning uses the reference discounted survival-weighted objective with
terminal-value gradients, and optimizes `-objective / horizon`. Critic targets
use the reference GAE recursion directly with `1-dhat`. WM and critic parameters
are frozen during actor BPTT, then the critic fits fixed targets across epochs
and shuffled minibatches. The old survival-ratio targets, actor ramp/freeze,
collapse rollback and alternate optimizer execution modes have been removed.

The necessary task adaptations are explicit: the Go2 state packs physical state,
command memory and commands into a tensor; context remains external. Contact
probabilities, analytic fall termination, terrain inputs and reward terms remain
quadruped definitions. The MoE history encoder is retained as the velocity task's
policy architecture. G1 motion libraries, FK, future-frame goals and its learned
termination BCE are not introduced into this velocity task. Real rewards and
previous-action task memory continue to use the executed action; replay separately
records the clean policy action.

## Run

Run from the repository root:

```bash
.venv/bin/python src/tasks/supermimic/scripts/train.py --help
.venv/bin/python src/tasks/supermimic/scripts/train.py go2-supermimic-flat \
  --device cuda:0 --run-name flat --env.scene.num-envs 1024
.venv/bin/python src/tasks/supermimic/scripts/train.py go2-supermimic-rough \
  --device cuda:0 --run-name rough
.venv/bin/python src/tasks/supermimic/scripts/train.py go2-supermimic-flat \
  --resume /absolute/path/ckpt_latest.pt --agent.schedule.total-iterations 400000
```

The same presets are available through `python -m src.tasks.supermimic.scripts.train`.
Supply the same preset and nondefault component options when resuming a run.
Counters, optimizer states and curricula are restored; replay and warmup refill
as in Reproduced. A run writes `params/env.yaml`, `params/agent.yaml`, `tb/`,
numbered `ckpt_N.pt` and `ckpt_latest.pt`.

Each checkpoint bundles its `TrainConfig`, including the actual native environment
configuration, alongside model and optimizer states. Playback and export require
only the checkpoint; there is no adjacent task snapshot or robot-settings copy.

CPU smoke runs should use small networks, environment counts and batches, with
`--agent.wm-trainer.compile None --agent.policy-trainer.compile None`. Default network sizes
and environment counts target GPU training. `torchrun` uses the reference per-rank
GPU pinning, parameter broadcast, gradient averaging and square-root LR scaling.
All ranks share terrain generation seeds while physics/reset seeds differ.

```bash
.venv/bin/python -m src.tasks.supermimic.scripts.play go2-supermimic-flat \
  --checkpoint-file /absolute/path/ckpt_latest.pt --viewer viser
.venv/bin/python -m src.tasks.supermimic.scripts.play go2-supermimic-rough \
  --agent zero --device cpu --steps 8 --env.scene.num-envs 2
.venv/bin/python -m src.tasks.supermimic.scripts.export \
  --checkpoint-file /absolute/path/ckpt_latest.pt --output-dir /absolute/path/deploy
```

Playback uses the selected task's native playback configuration. Repeat any
training overrides that affect policy inputs or action semantics, including
history length and action scale. Environment overrides follow the same manager paths, such as `--env.scene.num-envs`,
`--env.viewer.width`, `--env.viewer.height` and
`--env.commands.velocity.no-terminations True`. The programmatic
`make_playback(..., env_cfg=...)` helper uses the supplied native configuration;
when omitted for a trained policy, it uses the checkpoint's native configuration.

Export writes `policy.pt`, `policy.onnx` and `policy.json`, including fixed policy
scales and the native robot's action/default-position metadata. Default deployment
input is `[B,235]`, in frame-major oldest-first order, and output is `[B,12]`.
Playback routes partial resets through task memory and history. Rough playback
samples terrain rows on reset; saved command progress remains frozen.

## Validation

The native configuration refactor changes how task settings are constructed and
stored. Checkpoints created before the native task-registration layout use their original
checkout. Deployment policy inputs and outputs remain unchanged.

Validation of the preceding tensor-view refactor used temporary scripts outside
the repository. These compared native robot presets, state packing, integration,
analytic observations, commands, rewards and gradients against its preceding
implementation. Real CPU trajectories covered flat/rough terrain, startup
randomization, timeouts, partial resets and observation noise. States, contact
timers, commands, rewards, masks and RNG matched exactly with physics held fixed.
Replay checks covered terminal rows, ring wrap, sampled histories and manual-reset
episode boundaries. After 24 real steps and six updates, WM, MoE and critic weights
matched exactly, with Adam moment differences within float32 accumulation rounding.
The requested robot/control changes were then checked with fresh runs.

Those CPU smoke runs covered training, checkpoint continuation, playback, velocity
evaluation and TorchScript/ONNX export. Multi-GPU execution and long-run convergence
have not been measured. Existing `tests/` files are excluded from this reconstruction
and its replanned commits.

The native configuration refactor also matches the preceding implementation exactly
for all 1,846 compared tensors on each terrain. A 24-step CPU run with six learning
updates reproduces WM, MoE and critic weights bit for bit. Native configuration
roundtrips, checkpoint continuation, playback, evaluation, dynamic-batch export
and compiled imagination/WM gradients pass with the new configuration path.
Registry-based training also reproduces all three model weights exactly; temporary
additional registration confirms automatic train/play discovery and independent
configuration overrides without changing either entry point.

## Sources

- SuperMimic-Reproduced: `ba6105349069bf040f1fdad60873d13138d505a2`.
- quadruped_lab task semantics: `bac831afa62cd1bb89e0d3c7837bf1766b149d5a`.
- Installed MJLab 1.6.0 APIs and native Go2/Go1 asset conventions.

Runtime code does not import the local reference checkout.
