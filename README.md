# Legged MJLab

基于 MJLab、MuJoCo 和 PyTorch 的四足机器人运动学习项目，面向 Unitree Go2，提供本体感知、视觉感知和世界模型三类训练流程，支持训练、策略回放与导出。

## 算法与任务

当前提供 16 个 Go2 任务，按 `loco`、`pie`、`supermimic` 分组。任务注册时指定环境、算法和训练器，使用对应目录下的统一脚本启动。

| 算法 | 任务编号 | 算法文档 |
| --- | --- | --- |
| HIM | `go2-him-flat`、`go2-him-rough` | [本体运动控制](docs/loco.md) |
| HIM + AMP | `go2-him-amp-flat`、`go2-him-amp-rough` | [本体运动控制](docs/loco.md) |
| DreamWaQ | `go2-dwaq-flat`、`go2-dwaq-rough` | [本体运动控制](docs/loco.md) |
| DreamWaQ + AMP | `go2-dwaq-amp-flat`、`go2-dwaq-amp-rough` | [本体运动控制](docs/loco.md) |
| CTS | `go2-cts-flat`、`go2-cts-rough` | [本体运动控制](docs/loco.md) |
| CTS + AMP | `go2-cts-amp-flat`、`go2-cts-amp-rough` | [本体运动控制](docs/loco.md) |
| PIE / PIE + AMP | `go2-pie-rough`、`go2-pie-amp-rough` | [视觉运动控制](docs/pie.md) |
| SuperMimic | `go2-supermimic-flat`、`go2-supermimic-rough` | [世界模型与策略学习](docs/supermimic.md) |

- **本体运动控制**：从观测历史估计速度与运动隐变量，通过 PPO 学习速度跟踪；AMP 为策略增加参考动作约束。
- **视觉运动控制**：融合深度图与本体历史，联合学习状态估计和运动策略。
- **世界模型学习**：从仿真数据学习动力学，在可微想象轨迹中优化策略。

## 安装

建议使用 Python 3.11 和 NVIDIA GPU。主要依赖由 `setup.py` 管理，包括 MJLab 1.6.0、MuJoCo 3.11.0 和 PyTorch ≥ 2.7；GPU 驱动需与 PyTorch 匹配。

在项目根目录创建环境并安装：

```bash
conda create -n mjlab python=3.11
conda activate mjlab
pip install -e .
```

## 使用

基本流程为：**训练 → 回放验证 → 策略导出**。以下命令均在项目根目录、激活环境后执行。

### 1. 查看任务(loco+pie)

```bash
python scripts/list_envs.py
```

### 2. 训练

按需选择一种算法，任务编号可替换为上表中的对应任务。

```bash
# 本体运动控制
python scripts/loco/train.py go2-him-flat

# 视觉运动控制
python scripts/pie/train.py go2-pie-rough

# supermimic
python scripts/supermimic/train.py go2-supermimic-flat
```

### 3. 回放与导出

回放默认按所选任务的日志配置加载最新本地检查点，也可通过 `--checkpoint-file` 指定文件。各模块的导出时机与输出格式见算法文档。

```bash
python scripts/loco/play.py go2-him-flat
python scripts/pie/play.py go2-pie-rough
python scripts/supermimic/play.py go2-supermimic-flat
```

各入口支持 `--help`。并行环境数、恢复训练、观测定义和具体算法说明见上表文档。训练输出位于 `logs/`，使用 `tensorboard --logdir logs` 查看训练曲线。

## 目录结构

```text
scripts/              # 训练与回放入口
src/assets/           # 机器人模型与参考动作
src/tasks/loco/       # HIM、DreamWaQ、CTS 及 AMP 变体
src/tasks/pie/        # 视觉状态估计与策略学习
src/tasks/supermimic/ # 世界模型与可微策略学习
docs/                 # 算法与使用文档
```

## Acknowledgements

- [MJLab](https://github.com/mujocolab/mjlab)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [parkour_mjlab](https://github.com/yhx1203/parkour_mjlab.git)
- [My_unitree_go2_gym](https://github.com/yusongmin1/My_unitree_go2_gym.git)
- [HIMLoco](https://github.com/InternRobotics/HIMLoco.git)
- [DreamWaQ](https://github.com/Manaro-Alpha/DreamWaQ.git)
- [HIMLoco](https://github.com/InternRobotics/HIMLoco.git)
- [go2_rl_lab](https://github.com/wty-yy/go2_rl_lab.git)