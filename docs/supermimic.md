# Go2 SuperMimic

移植采用“固定算法基准、完整复制核心、适配仿真边界、复用本地资产、正式注册任务”的方式。
以 quadruped_lab 当前 SuperMimic 为唯一算法和任务基准；loco、PIE 提供目录布局和入口惯例，
MJLab 1.6.0 提供配置工厂、环境和注册 API。没有把模型学习算法改写成 PPO。

## 代码边界

| 模块 | 目标位置 | 处理方式 |
| --- | --- | --- |
| 通用 RL 核心 | `src/tasks/supermimic/rl/core/` | 原样复制 quadruped_lab `agents/supermimic/`，保留回放、归一化、MoE、世界模型、learner、runner、日志、部署导出和原许可 |
| 任务数学 | `src/tasks/supermimic/mdp/` | 复制状态、命令、观测、奖励、终止、地形和课程，只调整算法导入 |
| 真实环境 | `src/tasks/supermimic/config/go2/env_cfgs.py`、`env.py` | 集中装配原生 ManagerBasedRlEnvCfg、地形与播放模式；adapter 提取真实转移 |
| Go2 配置与注册 | `src/tasks/supermimic/config/go2/` | 按 loco/PIE 的 `env_cfgs.py`、`rl_cfg.py`、`__init__.py` 工厂布局正式注册 |
| 完整任务配置 | `config/go2/task_cfg.py`、`config/loading.py` | 保留参考嵌套参数和 JSON 快照协议，避免重写配置改变旧协议含义 |
| 任务训练及部署 | `src/tasks/supermimic/rl/` | 复制任务 runner、训练、检查点、播放、评估、外部驱动和部署装配，更新包路径 |
| 本地机器人 | `src/assets/robots/unitree_go2/supermimic.py` | 复用既有 MJCF，独立应用参考初姿、PD、动作比例、延迟和碰撞参数 |
| 命令入口 | `scripts/supermimic/{train,play,export}.py` | 薄脚本，实际命令装配在任务 `cli.py` |

注册任务为 `go2-supermimic-flat`、`go2-supermimic-rough`。
`load_env_cfg()`、`load_rl_cfg()`、`load_runner_cls()` 均可从 MJLab 注册表取到独立配置副本。
算法配置使用 `SuperMimicRunnerCfg(RslRlBaseRunnerCfg)`，包含独立的 `algorithm` 和 `execution`。
基础类的 `num_steps_per_env` 等 RSL 专用字段不参与 SuperMimic 计算，采集频率只由
`algorithm.collect_steps` 决定。训练和播放必须使用本目录的专用入口。

## 必须保留的算法和环境约定

- 默认 8 专家 MoE；actor 每帧 47 维、五帧 235 维，动作 12 维。actor 只读取本体历史和命令、相位。
- 物理状态 125 维，critic 262 维；地形和随机化上下文只进入世界模型与 critic。
- 原预算保持不变：世界模型窗口 8 步、每轮 10 次模型更新，模型 batch 2048；想象 batch 4096、horizon 32；critic 5 epochs × 4 minibatches。
- 平均采集 2.5 个控制步，按 2/3 交替；原始回放、稀有窗口归档、OU 探索、预热、冻结模型权重的 BPTT、软存活、末端价值、lambda-return 和塌缩回滚均保留。
- 物理步长 0.005 秒，decimation=4，控制周期 0.02 秒。独立资产封装不改变 loco、PIE 的配置。
- 原生环境 `auto_reset=False`。adapter 在物理步进后读取终态，计算奖励和终止、结算课程，再手动重置结束行。`final_state` 与重置后的 `next_state` 分开保留。
- 世界模型窗口仍按参考读取下一回放槽并屏蔽 done 当步及后续，不擅自改成以 `final_state` 监督。历史按旧到新排列，重置时重复初始帧。
- 原生观测、命令、奖励、课程 manager 保持为空，由同一任务 Tensor 模型统一实现真实和想象路径；不能把裸 ManagerBasedRlEnv 当成完整可训练任务。
- 不通过 RslRlVecEnvWrapper，也不额外裁剪动作；该包装器的初始化 reset、done 合并和可选动作裁剪不符合此任务转移协议。

机器人相对参考 MJCF 的差异：`base` 改为本地 `base_link`；足端前移 2 mm；hip/calf 的部分碰撞体为 cylinder，参考为 capsule。
质量、惯量、body 位姿和关节限位已对拍一致。MJWarp 对 cylinder 碰撞会提示 MULTICCD 仅返回单接触点；
这也是本地资产与参考 capsule 的物理差异。配置增加 `asset_variant` 进入语义指纹，
因此不能把参考资产检查点直接宣称为同一任务继续训练。网络同形状的显式热启动仍按原 runner 规则处理。

## 运行

在 legged_mjlab 仓库根目录，使用已安装依赖的 Python；本地环境为 `.venv/bin/python`。
`setup.py` 已声明保存策略时必需的 `onnx` 依赖，并将核心的完整 MIT 许可加入包数据。

```bash
.venv/bin/python scripts/list_envs.py --keyword supermimic
.venv/bin/python scripts/supermimic/train.py go2-supermimic-flat --help
.venv/bin/python scripts/supermimic/train.py go2-supermimic-flat --device cuda:0
.venv/bin/python scripts/supermimic/train.py go2-supermimic-rough --device cuda:0
```

通过 `config/go2/rl_cfg.py` 的工厂修改新训练算法默认值，通过 `task_cfg.py` 修改任务默认值。
单次实验推荐使用局部 JSON 覆盖，以保留可追溯配置。例如 `experiment.json`：

```json
{
  "env": {"num_envs": 1024, "seed": 1},
  "algo": {"collect_steps": 2.5},
  "commands": {"initial_ranges": [[-1.0, 1.0], [-1.0, 1.0], [-2.0, 2.0]]}
}
```

```bash
.venv/bin/python scripts/supermimic/train.py go2-supermimic-flat \
  --device cuda:0 --task-config /absolute/path/experiment.json --run-name baseline
```

未给 JSON 或恢复来源时，使用注册表的算法默认值；给出任务 JSON 时以其覆盖任务默认值；
恢复时优先恢复完整快照。命令行 `--num-envs`、`--seed` 是最后的显式覆盖。
新训练默认 4096 环境。CPU 验证需要显式减少环境、网络、回放和 batch，关闭编译和 fused Adam；
默认 GPU 预算不适合直接在 CPU 跑；只检查环境时可使用下方的 zero 策略播放命令。

运行产物位于 `logs/supermimic/go2_supermimic_{flat,rough}/时间戳/`，包含：

- `model_N.pt`、当前 `policy.onnx` 和完整 `task_config.json`；
- `params/env.yaml`、`agent.yaml`、`run.yaml`、`git.diff` 及 TensorBoard 事件文件。

```bash
.venv/bin/python scripts/supermimic/play.py go2-supermimic-flat \
  --checkpoint-file /absolute/path/model_1000.pt --viewer viser
.venv/bin/python scripts/supermimic/play.py go2-supermimic-flat \
  --agent zero --device cpu --steps 50
.venv/bin/python scripts/supermimic/train.py go2-supermimic-flat \
  --resume /absolute/path/model_1000.pt --continue-run True --max-iterations 2000
.venv/bin/python scripts/supermimic/export.py \
  --checkpoint-file /absolute/path/model_1000.pt --output-dir /absolute/path/deploy
```

`--resume` 默认只热启动参数，从零重新计轮；`--continue-run True` 恢复迭代号和累计计数。
`--max-iterations` 是目标总轮数。回放、物理状态和随机数轨迹不恢复；不等价于逐步无缝续跑。
任务装配层额外检查继续训练的语义指纹，拒绝同维度但奖励、命令或资产不同的接续；跨语义请显式热启动。
播放默认关闭超时重置、保留跌倒重置，支持 zero/random/trained、native/viser、录像和有限步无窗口运行。
播放和部署导出需要检查点同目录的完整任务快照。
导出 PT/ONNX/JSON 包含双路归一化、MoE 编码器、动作 MLP 和关节、动作、相位元数据。
部署图输入 `[B,235]`、输出 `[B,12]`；相位、heading 和历史维护由调用方负责。

## 参考依据与选择

| 来源 | 核查版本 | 采用方式 |
| --- | --- | --- |
| 本地 quadruped_lab | `bac831afa62cd1bb89e0d3c7837bf1766b149d5a` | 唯一算法/任务基准，直接复制可复用部分 |
| 本地 legged_mjlab | 移植前 `ee6489b8db0439073f6088f453e99b1af43f4834` | loco/PIE 任务目录、Go2 资产、注册与薄脚本惯例 |
| MJLab 官方源码 | 本地安装 1.6.0 | 实读 `tasks/registry.py`、`utils/lab_api/tasks/importer.py`、env/wrapper/viewer 接口；[官方项目](https://github.com/mujocolab/mjlab) |
| SuperMimic-Reproduced | `ba6105349069bf040f1fdad60873d13138d505a2` | G1 动作跟踪、10 帧历史，只参考边界；不混入其模型和回放 |
| supermimic-mjlab | `9aab0a588c6941bf9862a2d444b711fd73e2b832` | 交叉核对手动重置和仿真边界；不引入其动作库 |
| Uni-SuperMimic | `ac19b73a4f919fb2ea37dbad860b012315497139` | 核对 registry/MDP 协议思想；其 mask、value normalization 等不替换指定基准 |

这些本地参考项目位于 `/home/l/文档/supermimic/`。移植后的运行不依赖该目录。
复制核心的完整 MIT 许可见 `src/tasks/supermimic/rl/core/LICENSE`。

## 验证范围

```bash
MPLCONFIGDIR=/tmp/legged-supermimic-mpl \
WARP_CACHE_PATH=/tmp/legged-supermimic-warp \
OMP_NUM_THREADS=1 \
.venv/bin/python scripts/supermimic/play.py go2-supermimic-flat \
  --agent zero --device cpu --steps 8
```

两个缓存环境变量将 Matplotlib 和 Warp 缓存写入临时目录，适用于只读 HOME 的运行沙箱。
当前仓库未包含专用测试源码；以上命令检查真实环境 reset/step，各脚本的 `--help` 可检查命令入口。
小规模 CPU 冒烟验证不能替代长时间训练、收敛或真机验证；GPU 的编译、fused Adam、
默认 4096 环境规模以及训练性能仍需实际 GPU 验证。
