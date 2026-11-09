# SuperMimic：世界模型与策略学习

[项目首页](../README.md) · [Loco](loco.md) · [PIE](pie.md)

本项目中的 SuperMimic 用于 Go2 速度跟踪：从真实仿真数据学习动力学，在可微世界模型中展开想象轨迹，通过回报梯度优化历史策略。奖励、观测、指令和终止条件由解析任务模块计算。

## 观测与控制

单帧为 47 维，前 45 维顺序与 loco 一致，末尾追加步态相位：

| 观测项 | 维度 |
| --- | --- |
| 速度指令 `vx, vy, yaw_rate` | 3 |
| 机体角速度 | 3 |
| 投影重力 | 3 |
| `joint_pos - default_joint_pos + encoder_bias` | 12 |
| 关节速度 | 12 |
| 上一步未缩放动作 | 12 |
| 步态相位 `sin, cos` | 2 |

历史为 6 帧，包含当前帧，形状为 `[N, 6, 47]`，旧帧在前、当前帧在后；展平为 282 维。重置使用初始帧填满历史。

策略网络观测保持原始量纲，噪声在组装后的向量上按各项幅度添加。指令、上一步动作和步态相位不加噪，站立时相位两维均为零。策略内部的历史与当前帧分别使用运行统计归一化。

世界模型的打包状态为 170 维：物理状态 125、任务记忆 42、指令 3；另带 16 维随机物理上下文。价值网络使用 262 维特权特征，其中包含物理状态、任务状态和地形信息，部署时不需要该输入。

## 学习流程

```text
仿真采集 → 经验回放缓存 → 动力学模型拟合
                ↓              ↓
           采样起始状态 → 可微想象轨迹 → 策略与价值网络更新
```

### 世界模型

世界模型使用 `1024, 1024, 1024` 的 MLP，从规范化状态特征、执行动作和物理上下文预测物理变化。物理状态经积分后，解析模块推进任务记忆与指令，并计算观测、奖励和终止概率。

监督训练使用 8 步窗口，逐步自回归预测，以归一化坐标误差的均方作为目标。重置转移及窗口内随后无效的位置通过掩码排除；任务记忆和指令采用真实记录进行教师强制。输入、输出均使用运行统计归一化，接触概率对数几率的输出统计固定为零均值、单位方差。

经验回放缓存保存真实采集的带噪策略网络历史，使用 float16 存储。世界模型采样默认包含 25% 优先窗口；稀有事件缓存非空时再使用 25% 稀有事件缓存窗口，其余为常规样本。

实现：[WorldModel](../src/tasks/supermimic/world_model/world_model.py)、[StateSpec](../src/tasks/supermimic/world_model/state_spec.py)、[ReplayBuffer](../src/tasks/supermimic/data/replay_buffer.py)。

### 历史策略

历史编码器采用包含 8 个专家的混合专家结构。门控网络对专家投影加权，得到 L2 归一化的 32 维隐变量，与当前 47 维观测拼接后，输入 `512, 256, 128` 的策略网络 MLP。

策略网络输出确定性动作。真实环境采集通过额外探索噪声扩展数据：默认 25% 环境使用 OU 噪声，其余使用独立高斯噪声。价值网络使用带运行统计归一化的特权 MLP 估计价值。

实现：[策略与价值网络](../src/tasks/supermimic/policy/nets.py)、[真实采集循环](../src/tasks/supermimic/training/loop.py)。

### 可微想象与策略目标

从经验回放缓存采样状态与其真实观测历史，在世界模型中展开默认 32 步轨迹。每一步执行策略、预测动力学、计算解析奖励与软终止概率，再更新历史。

令 `S_0=1`、`S_(t+1)=S_t(1-d_(t+1))`，策略最大化生存概率加权的折扣回报：

$$
J=\mathbb{E}\left[\sum_{t=0}^{H-1}\gamma^t S_t r_t+\gamma^H S_H V(x_H)\right].
$$

更新策略网络时冻结世界模型和价值网络的参数，但保留经过模型状态和末端价值输入的梯度。优化器最小化 `-J`，同时裁剪梯度范数和梯度值。该更新直接对想象轨迹求导，不使用 PPO 概率比目标。

价值网络另用固定的生存概率加权 λ 回报监督，默认 `gamma=0.99`、`lambda=0.95`。训练目标在更新价值归一化统计之前计算。

实现：[想象轨迹](../src/tasks/supermimic/imagination/rollout.py)、[优化目标](../src/tasks/supermimic/imagination/objective.py)、[策略训练器](../src/tasks/supermimic/imagination/policy_trainer.py)。

### 默认训练日程

| 参数 | 默认值 |
| --- | --- |
| 总学习迭代 | 40000 |
| 每次迭代真实控制步 | 平均 2.5，交替收集 2 / 3 步 |
| 仅拟合世界模型的预热 | 300 次迭代 |
| 策略网络学习率爬升 | 1000 次迭代 |
| 世界模型更新 | 每迭代 10 批，每批 2048 个 8 步窗口 |
| 想象更新 | 每批 4096 个起点，展开 32 步 |
| 世界模型与策略基础学习率 | `2e-4` |
| 学习率余弦衰减 | 30000 次迭代后降至初值的 `0.03` |

训练首先填充经验回放缓存，再拟合模型，第 301 次迭代开始更新策略。非有限梯度会跳过更新；连续触发跌倒保护时恢复已保存的策略与价值网络、保留世界模型、减半学习率并重启策略网络爬升。当前训练循环支持单进程。

## 任务与运行

当前注册 `go2-supermimic-flat` 与 `go2-supermimic-rough`。从项目根目录执行：

```bash
python scripts/supermimic/train.py go2-supermimic-flat 

python scripts/supermimic/train.py go2-supermimic-rough

python scripts/supermimic/train.py go2-supermimic-flat --resume /absolute/path/ckpt_latest.pt
```

环境参数使用 `--env.*`，学习参数使用 `--agent.world-model.*`、`--agent.wm-trainer.*`、`--agent.policy-trainer.*` 和 `--agent.schedule.*`。

默认日志目录为 `logs/supermimic/<task_id>/<timestamp>/`；指定 `--run-name` 后，以自定义名称代替任务编号。目录内包含 `ckpt_latest.pt`、周期检查点、`tb/`、`train.log` 和 `params/`。TensorBoard 指标包括 `wm/*`、`policy/*`、`train/*`、`time/*` 和 `perf/*`。

恢复训练会读取权重、优化器与进度；经验回放缓存和物理环境重新初始化。使用非默认设置训练时，恢复命令需保留对应设置。

### 回放与导出

```bash
# 加载当前任务的最新策略
python scripts/supermimic/play.py go2-supermimic-flat

# 指定检查点，使用浏览器查看
python scripts/supermimic/play.py go2-supermimic-flat \
  --checkpoint-file /absolute/path/ckpt_latest.pt --viewer viser

# 仅导出策略
python scripts/supermimic/play.py go2-supermimic-flat \
  --checkpoint-file /absolute/path/ckpt_latest.pt \
  --export-only True --output-dir /absolute/path/deploy

# 在 CPU 上回放指定步数，关闭导出
python scripts/supermimic/play.py go2-supermimic-flat \
  --device cpu --steps 100 --export False
```

未指定检查点时，按运行目录名称从新到旧查找，优先使用 `ckpt_latest.pt`，否则选择该目录编号最大的检查点。自定义训练目录时，回放需同步传入 `--log-root` 和 `--run-name`。

训练策略回放默认先导出 `policy.pt`、`policy.onnx` 和 `policy.json`，输出位于检查点旁的 `exported/<checkpoint_stem>/`。当前默认导出输入为 `[N, 282]`，策略网络的两套运行统计归一化已包含在图内；调用方负责按 JSON 中的顺序准备历史，输出为 12 维未缩放动作。

## 代码入口

| 路径 | 内容 |
| --- | --- |
| [config/go2](../src/tasks/supermimic/config/go2) | 任务注册、环境与学习默认值 |
| [envs/mdp](../src/tasks/supermimic/envs/mdp) | 观测、指令、奖励、地形与终止 |
| [world_model](../src/tasks/supermimic/world_model) | 状态表示、动力学及监督更新 |
| [data](../src/tasks/supermimic/data) | 采集记录、经验回放缓存与稀有事件缓存 |
| [policy](../src/tasks/supermimic/policy) | 混合专家策略网络与特权价值网络 |
| [imagination](../src/tasks/supermimic/imagination) | 想象展开、策略目标与价值网络更新 |
| [training](../src/tasks/supermimic/training) | 训练日程、保护机制、日志与检查点 |
| [eval](../src/tasks/supermimic/eval) | 回放、评估与导出 |
