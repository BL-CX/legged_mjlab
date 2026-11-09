# Loco：本体运动控制

[项目首页](../README.md) · [PIE](pie.md) · [SuperMimic](supermimic.md)

Loco 使用本体观测历史估计速度和运动隐变量，以 PPO 训练四足机器人的速度跟踪策略。当前包含 HIM、DreamWaQ、CTS，以及各自的 AMP 变体。

## 观测与控制

默认物理步长为 0.005 s，每 4 个物理步执行一次策略，控制频率为 50 Hz。策略输出 12 维关节位置偏移，经动作缩放和默认关节位置转换为 PD 目标。

策略网络的单帧观测为 45 维，顺序如下：

| 观测项 | 维度 | 缩放 |
| --- | --- | --- |
| 速度指令 `vx, vy, yaw_rate` | 3 | `` |
| 机体角速度 | 3 | `0.25` |
| 投影重力 | 3 | `1` |
| 相对默认姿态的关节位置 | 12 | `1` |
| 相对默认速度的关节速度 | 12 | `0.05` |
| 上一步动作 | 12 | `1` |

原始观测依次经过噪声、裁剪和缩放。策略网络使用 6 帧历史，形状为 `[N, 6, 45]`，按时间从旧到新排列，训练器再展平为 270 维。

价值网络使用不加本体观测噪声的当前帧，并追加真实线速度、机体外力和地形高度扫描。CTS 另设 `privileged` 组供教师编码器使用；默认内容与其 `critic` 组一致。这些特权信息仅用于训练。

## 算法说明

### PPO

三类方法均通过真实环境采集的轨迹计算 GAE 和价值目标，再使用裁剪 PPO 目标更新随机策略与价值网络。策略损失采用最小化形式：

$$
L_\pi=-\mathbb{E}\left[\min\left(\rho_t A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\right)\right].
$$

其中 `rho` 为新旧策略的动作概率比。价值误差、熵奖励和各方法的辅助损失共同参与训练；默认使用自适应 KL 学习率。

### HIM

历史估计器接收全部 6 帧，输出 3 维速度估计和 16 维归一化隐变量。策略网络将当前 45 维观测与两类估计拼接，输出动作分布。

估计器使用独立优化器，目标为：

$$
L_{\mathrm{est}}=L_{\mathrm{velocity}}+L_{\mathrm{swap}}.
$$

速度项由下一步价值网络观测中的真实线速度监督。隐变量项将历史编码与下一步特权观测投影到共享原型，使用 Sinkhorn 分配和双向交换预测学习表征。默认原型数为 32。PPO 更新时，估计输出停止梯度传播，估计器由上述辅助目标更新。

实现：[HIMEstimator](../src/tasks/loco/rsl_rl/modules/him_estimator.py)、[HIMActorCritic](../src/tasks/loco/rsl_rl/modules/him_actor_critic.py)、[HIMPPO](../src/tasks/loco/rsl_rl/algorithms/him_ppo.py)。

### DreamWaQ

VAE 接收当前帧之前的 5 帧，输出 3 维显式速度和 16 维隐式状态。策略网络将估计结果与当前帧拼接。训练时重参数化采样，推理时使用均值。

VAE 的辅助目标为：

$$
L_{\mathrm{VAE}}=L_{\mathrm{velocity}}+L_{\mathrm{reconstruction}}+\beta L_{\mathrm{KL}}.
$$

速度监督和 45 维重建目标均来自采样轨迹保存的无噪声当前价值网络观测，KL 约束隐式分布接近标准正态；默认 `beta=1`。终止样本通过有效标记屏蔽。辅助更新使用独立的 VAE 优化器。

实现：[VAE](../src/tasks/loco/rsl_rl/modules/vae.py)、[DreamWaQActorCritic](../src/tasks/loco/rsl_rl/modules/dreamwaq_actor_critic.py)、[DreamWaQPPO](../src/tasks/loco/rsl_rl/algorithms/dreamwaq_ppo.py)。

### CTS

教师编码器从特权观测生成 32 维隐变量；学生编码器从当前帧之前的 5 帧生成同维隐变量。教师与学生共享策略网络，均将隐变量与当前帧拼接后产生动作。

默认 75% 环境运行教师策略，25% 环境运行学生策略。教师编码器与共享策略通过 PPO 更新；学生编码器使用独立优化器，拟合停止梯度的教师隐变量：

$$
L_{\mathrm{student}}=\mathbb{E}\left[\|z_s-\operatorname{stopgrad}(z_t)\|^2\right].
$$

部署和回放使用学生编码器，不需要真实线速度或地形扫描作为策略网络输入。

实现：[CTSActorCritic](../src/tasks/loco/rsl_rl/modules/cts_actor_critic.py)、[CTS](../src/tasks/loco/rsl_rl/algorithms/cts.py)。

### AMP

AMP 在上述方法上增加运动判别器，不替换历史估计器。单步 AMP 状态为 30 维：关节位置 12、机体线速度 3、机体角速度 3、关节速度 12。判别器接收相邻两帧共 60 维，采用专家标签 `+1`、策略标签 `-1` 的最小二乘损失，并加入专家输入梯度惩罚。

风格奖励与任务奖励按以下方式组合：

$$
r_{\mathrm{style}}=c\max\left(0,1-\tfrac14(D(s_t,s_{t+1})-1)^2\right),\qquad
r=\alpha r_{\mathrm{task}}+(1-\alpha)r_{\mathrm{style}}.
$$

`c` 和 `alpha` 由各训练器配置决定。参考动作默认读取 `src/assets/motions/mocap_motions_go2/*.txt`，归一化器和经验回放缓存用于判别器训练。

实现：[AMPDiscriminator](../src/tasks/loco/rsl_rl/algorithms/amp_discriminator.py)、[Go2 学习配置](../src/tasks/loco/config/go2/rl_cfg.py)。

## 任务与运行

任务名为 `go2-{method}-{terrain}`，其中 `method` 为 `him`、`dwaq`、`cts` 或对应的 `*-amp`；`terrain` 为 `flat`（平坦地形）或 `rough`（粗糙地形）。例如 `go2-cts-amp-rough`。

```bash
python scripts/loco/train.py go2-him-flat
python scripts/loco/train.py go2-dwaq-rough --env.scene.num-envs 4096
python scripts/loco/train.py go2-cts-amp-rough

python scripts/loco/play.py go2-him-flat
python scripts/loco/play.py go2-him-flat --checkpoint-file /absolute/path/model_1000.pt
```

默认每环境收集 24 步、执行 5 轮参数更新、划分 4 个小批次。HIM、DreamWaQ 及其 AMP 变体训练上限为 50000 次迭代，CTS 系列为 20000 次。三个基础方法的策略与价值网络隐层均为 `512, 256, 128`。完整参数通过相应入口的 `--help` 查看。

## 代码入口

| 路径 | 内容 |
| --- | --- |
| [config/go2](../src/tasks/loco/config/go2) | 任务注册、环境与学习参数 |
| [rsl_rl/modules](../src/tasks/loco/rsl_rl/modules) | 策略、估计器、VAE 与教师学生网络 |
| [rsl_rl/algorithms](../src/tasks/loco/rsl_rl/algorithms) | PPO、辅助目标与 AMP 更新 |
| [rsl_rl/runners](../src/tasks/loco/rsl_rl/runners) | 数据采集、训练循环和检查点 |
