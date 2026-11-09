# PIE：视觉运动控制

[项目首页](../README.md) · [Loco](loco.md) · [SuperMimic](supermimic.md)

PIE 将深度图和本体观测历史融合为带记忆的状态估计，通过 PPO 与辅助监督联合训练视觉运动策略。当前提供普通 PIE 和 PIE-AMP 两个 Go2 粗糙地形任务。

## 观测与控制

控制频率为 50 Hz，动作是 12 维关节位置偏移。策略网络使用以下输入：

| 输入 | 默认形状 | 说明 |
| --- | --- | --- |
| `actor` | `[N, 10, 45]` | 10 帧本体历史，旧帧在前，当前帧在后 |
| `camera` | `[N, 10320]` | 展平的两帧深度图，即 `[N, 2, 60, 86]` |
| GRU 隐状态 | `[1, N, 128]` | 跨控制步维护，回合重置时清零 |

本体单帧顺序为 `机体角速度 → 投影重力 → 速度指令 → 关节位置 → 关节速度 → 上一步动作`，维度依次为 `3, 3, 3, 12, 12, 12`。PIE 沿用自身的原始量与统计归一化流程，顺序和历史长度由 PIE 配置独立定义。

深度图由 `60 × 106` 裁去左右各 10 列，得到 `60 × 86`。无效深度视为最远距离，经高斯模糊、距离裁剪和归一化后存入两帧缓存。缓存每 5 个控制步更新一次，即 10 Hz；重置时用初始帧填满历史。

价值网络的输入由干净的当前本体帧、真实线速度、地形高度扫描和 4 个足端离地高度组成。它为价值网络和辅助损失提供监督，推理时策略网络不直接读取这些特权量。

## 网络与训练目标

```text
本体历史 → 历史编码器 ┐
                     ├→ 多模态融合 → 循环记忆 → 状态估计
深度图像 → 视觉编码器 ┘
状态估计 → 辅助解码 → 监督损失
当前本体观测 + 状态估计 → 策略网络 → 关节动作
```

历史 MLP 生成 64 维特征单元，深度 CNN 生成空间特征单元。两层 Transformer 融合不同模态；本体特征单元与视觉特征单元的均值拼接后进入 128 维 GRU。记忆输出产生以下估计：

| 输出 | 维度 | 作用 |
| --- | --- | --- |
| 线速度 | 3 | 显式运动状态 |
| 地形隐变量 | 16 | 重建地形高度扫描 |
| 足端离地高度 | 4 | 显式足端几何状态 |
| VAE 隐变量 | 16 | 补充未显式建模的信息 |

策略网络将 39 维估计与归一化的当前 45 维本体观测拼接，输入 `512, 256, 128` 的 MLP，输出高斯动作分布。训练时采样 VAE 隐变量，推理时使用均值；部署需要同时维护本体历史、深度缓存和 GRU 状态。

### 联合损失

PPO 的裁剪目标、价值损失与熵项之外，额外加入五类估计器损失：

$$
L=L_\pi+c_vL_V-c_H\mathcal{H}
 +c_{\mathrm{aux}}\left(L_{\mathrm{velocity}}+L_{\mathrm{foot}}+L_{\mathrm{height}}+L_{\mathrm{successor}}+4L_{\mathrm{KL}}\right).
$$

上式为默认权重，各项含义如下：

| 损失 | 监督目标 |
| --- | --- |
| `velocity` | 当前价值网络观测中的真实线速度 |
| `foot_clearance` | 当前价值网络观测中归一化的足端离地高度 |
| `height_reconstruction` | 地形隐变量解码得到当前价值网络观测中的高度扫描 |
| `successor` | 估计向量解码得到下一步价值网络观测中的前 45 维干净本体观测 |
| `kl` | VAE 隐变量分布与标准正态分布的 KL 距离 |

默认 `auxiliary_loss_coef=1`，前四项系数均为 1，`kl_loss_coef=4`。辅助目标和 PPO 联合反向传播到策略网络与估计器；序列采样轨迹的隐状态和有效掩码保持序列对应关系。

实现：[PIEActorModel](../src/tasks/pie/rl/pie_model.py)、[PIEPPO](../src/tasks/pie/rl/ppo.py)。

### PIE-AMP

PIE-AMP 保留上述网络与辅助目标，增加独立的运动判别器、参考动作数据和经验回放缓存。AMP 使用 30 维运动状态及相邻两帧输入，损失和奖励形式与 [Loco AMP](loco.md) 相同。

实现：[PIEAMPPPO](../src/tasks/pie/rl/amp_ppo.py)、[AMP 配置](../src/tasks/pie/rl/config.py)。

## 任务与运行

| 任务编号 | 算法 |
| --- | --- |
| `go2-pie-rough` | PIE |
| `go2-pie-amp-rough` | PIE + AMP |

```bash
python scripts/pie/train.py go2-pie-rough
python scripts/pie/train.py go2-pie-amp-rough --env.scene.num-envs 128

python scripts/pie/play.py go2-pie-rough
python scripts/pie/play.py go2-pie-rough --checkpoint-file /absolute/path/model_1000.pt
```

恢复训练可使用 `--agent.resume True --agent.load-run '.*' --agent.load-checkpoint 'model_1000.pt'`。回放默认按任务的 `experiment_name` 查找最新本地检查点。所有命令均可追加 `--help` 查看参数。

## 代码入口

| 路径 | 内容 |
| --- | --- |
| [config/go2/env_cfgs.py](../src/tasks/pie/config/go2/env_cfgs.py) | 本体观测、深度相机、地形与奖励 |
| [config/go2/rl_cfg.py](../src/tasks/pie/config/go2/rl_cfg.py) | 网络结构与 PPO、AMP 参数 |
| [mdp/observations.py](../src/tasks/pie/mdp/observations.py) | 深度预处理、历史缓存与足端监督 |
| [rl/pie_model.py](../src/tasks/pie/rl/pie_model.py) | 多模态估计器、GRU 与策略网络 |
| [rl/runner.py](../src/tasks/pie/rl/runner.py) | 训练循环与 ONNX 导出 |
