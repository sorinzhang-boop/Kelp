# 流式安全围栏跨模型适配实验记录

**结果表口径**：`Adapter 训练 Response` 表示用于训练表征映射器的 Response 来源；`Safety Head 训练数据` 表示该围栏最初训练所用的 S-Eval Train 数据集，两者不是同一概念。A 使用 Source 模型原生围栏，D 使用 Target 模型原生围栏；B、C、E、F 均复用同一个已训练并冻结的 Target 模型 Safety Head。表中模型、训练集、测试集均使用完整名称；F1 均针对 harmful（label=1）类别。

## 实验一：Qwen3-14B → Qwen3-8B

### 1. 实验配置与结果总表

- **环境**：`guard_g22`，NVIDIA L20（使用 GPU 0）；项目根目录：`~/plugguard_repro/`（下文路径均相对于此目录）。
- **数据**：StreamGuardBench v1 / S-Eval；8B、14B 各有 Train 9000 / Test 1000。Qwen3-8B S-Eval Test：benign 467 / harmful 533；Qwen3-14B S-Eval Test：benign 485 / harmful 515。
- **表征**：均取 Layer 20；Qwen3-8B 为 4096 维，Qwen3-14B 为 5120 维。B、C、E、F 均**冻结同一个 Qwen3-8B Safety Head**；A、D 为此前复现的原生围栏。
- **指标**：下表 F1 均指 harmful（label=1）类别；每组测试 1000 条。`8B/Qwen3-14B S-Eval Test` 指 **Response 的生成来源**，不等于提取 hidden states 的模型。

| 实验 | 设计目的 | Adapter 训练 Response | Safety Head 训练数据 | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---|---:|---:|
| **A** | Source 模型原生围栏基线 | 无（原生围栏） | Qwen3-14B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → 冻结的 Qwen3-14B Safety Head | 0.8955 | **0.8971** |
| **B** | 在 Source 模型的 S-Eval Train 上训练 Adapter | Qwen3-14B S-Eval Train | Qwen3-8B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → Affine Adapter → 冻结的 Qwen3-8B Safety Head | 0.8182 | **0.8054** |
| **C** | 真实 Target 模型表征对照（不使用 Adapter） | 无（不使用 Adapter） | Qwen3-8B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-8B 对同一 Qwen3-14B Response 提取的真实 Layer 20 hidden states（4096D）→ 冻结的 Qwen3-8B Safety Head | 0.9080 | **0.9007** |
| **D** | Target 模型原生围栏基线 | 无（原生围栏） | Qwen3-8B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → 冻结的 Qwen3-8B Safety Head | 0.9245 | **0.9228** |
| **E** | 在 Target 模型的 S-Eval Train 上训练 Adapter，并测试 Target 模型生成的 Response | Qwen3-8B S-Eval Train | Qwen3-8B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → Affine Adapter → 冻结的 Qwen3-8B Safety Head | 0.8247 | **0.8310** |
| **F** | 复用 E 的 Adapter，并测试 Source 模型生成的 Response | Qwen3-8B S-Eval Train（与 E 共用 Adapter） | Qwen3-8B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → Affine Adapter → 冻结的 Qwen3-8B Safety Head | 0.8121 | **0.8119** |

**说明**：E、F 共用同一个 Adapter；B 使用另一个在 Qwen3-14B S-Eval Train 上训练的 Adapter。C 需要在测试时运行 8B 提取真实表征，仅是诊断基线，不是最终部署方案。

### 2. Adapter 结构、损失与训练配置

**适配目标**：不重新训练旧围栏，只训练 `Qwen3-14B Layer20 → Qwen3-8B Layer20` 的映射。

- **网络**：`torch.nn.Linear(5120, 4096, bias=True)`，约 **20.98M** 可训练参数。
- **输入/输出**：同一条 *prompt + response* 分别经过冻结的 14B、8B 提取成对 hidden states；训练时 14B 表征为输入、8B 表征为监督目标。按训练集统计的逐特征均值和标准差归一化；送入 8B 围栏前反归一化。
- **损失函数**：仅使用归一化 hidden states 的均方误差（MSE），**没有**使用安全标签、交叉熵、风险蒸馏或 MATCHA 多层注意力。

```text
x = (h_Qwen14 - mean_Qwen14) / std_Qwen14
y = (h_Qwen8  - mean_Qwen8)  / std_Qwen8
y_pred = Linear(x)                   # 5120 -> 4096
L_align = MSE(y_pred, y)
h_mapped = y_pred * std_Qwen8 + mean_Qwen8
```

| 训练参数 | B / E、F 共用配置 |
|---|---|
| 训练/验证样本 | 8100 / 900（从相应模型的 9000 条 Train 划分，seed=42） |
| Epoch / Optimizer | 3 / AdamW |
| LR / Weight decay | `3e-4` / `0` |
| 梯度累积 | 8 条样本/更新 |
| Token 抽样 | 每样本最多 128 个 token，兼顾 prompt 与 response |
| 归一化统计 | 训练子集前 320 条样本抽样估计 |
| Checkpoint 选择 | 验证集 normalized MSE 最低 |
| 冻结部分 | 8B、14B 底座参数与原有 8B Safety Head |

E/F 最优 Adapter：**epoch 3，验证 normalized MSE = 0.3142155**。B 的对应数值待从原日志核对。

**统一评测规则**：Response-level 使用 `pred[-2]`；Streaming-level 使用 `max(pred)`（任意响应 token 判为 harmful 即触发），保持此前 PlugGuard 复现代码的口径。

### 3. 代码与实验产物位置

以下路径相对于 `~/plugguard_repro/`。

| 实验 | 运行代码 | Checkpoint / 结果文件 | 日志 |
|---|---|---|---|
| **A** | `Kelp/train.py`、`Kelp/eval.py` | `checkpoints/qwen3_14b_seval/model_epoch_0.pt` | `logs/qwen3_14b_seval_train.log` |
| **B** | `Kelp/affine_qwen14b_to8b.py`（`--mode train/eval`） | `checkpoints/adapter_qwen3_14b_to_qwen3_8b_layer20/affine_best.pt`、`metrics.json` | `logs/adapter_affine_14b_to8b.log` |
| **C** | `Kelp/eval.py` 的 `evaluate_safety_head()`（终端内联 Python 调用） | 复用 `checkpoints/qwen3_8b_seval/model_epoch_0.pt` | `logs/oracle_c_8b_on_14b_test.log` |
| **D** | `Kelp/train.py`、`Kelp/eval.py` | `checkpoints/qwen3_8b_seval/model_epoch_0.pt` | `logs/qwen3_8b_seval_train.log` |
| **E/F** | `Kelp/affine_legacy8b_train_eval_e_f.py`、`Kelp/run_affine_legacy_e_f.sh` | `checkpoints/adapter_qwen3_14b_to_qwen3_8b_train_on_8b_seval_layer20/affine_best.pt`、`metrics_e.json`、`metrics_f.json`、`predictions_e.csv`、`predictions_f.csv` | `logs/adapter_legacy8b_e_f.log` |

**执行方式**：B 分别运行 `--mode train`、`--mode eval`；E/F 运行 `bash Kelp/run_affine_legacy_e_f.sh`（在 Kelp 目录内可写 `bash run_affine_legacy_e_f.sh`），自动顺序训练一次 Adapter、评测 E、评测 F，指定 `CUDA_VISIBLE_DEVICES=0`。

**初步观察**：同一 Qwen3-14B S-Eval Test 上，C 的 Streaming F1 为 0.9007，B/F 为 0.8054/0.8119；当前主要差距在将真实 8B 表征替换成 Affine 映射表征之后。尚未进行 MLP、MATCHA 或风险蒸馏实验。

---

## 实验二：Llama-3.1-8B-Instruct → Qwen3-8B

### 1. 实验配置与结果总表

- **环境与数据**：同上，使用 GPU 0；StreamGuardBench v1 / S-Eval。Llama、Qwen3-8B 各有 Train 9000 / Test 1000；Llama-3.1-8B-Instruct S-Eval Test：benign 547 / harmful 453；Qwen3-8B S-Eval Test：benign 467 / harmful 533。
- **表征与围栏**：两模型均提取 Layer 20、hidden size 均为 4096。B、C、E、F 始终使用**同一个冻结的 Qwen3-8B Safety Head**；A、D 为此前完成的原生围栏复现实验。
- **指标**：F1 均为 harmful（label=1）类别。下表的 Llama/Qwen3-8B S-Eval Train、Test 指 Response 的生成来源；B、E、F 测试时均由 **Llama 提取表征**，经过 Adapter 进入 8B 围栏。

| 实验 | 设计目的 | Adapter 训练 Response | Safety Head 训练数据 | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---|---:|---:|
| **A** | Source 模型原生围栏基线 | 无（原生围栏） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.9095 | **0.9191** |
| **B** | 在 Source 模型的 S-Eval Train 上训练 Adapter | Llama-3.1-8B-Instruct S-Eval Train | Qwen3-8B S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-8B Safety Head | 0.8340 | **0.8973** |
| **C** | 真实 Target 模型表征对照（不使用 Adapter） | 无（不使用 Adapter） | Qwen3-8B S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Qwen3-8B 对同一 Llama-3.1-8B-Instruct Response 提取的真实 Layer 20 hidden states（4096D）→ 冻结的 Qwen3-8B Safety Head | 0.9271 | **0.9253** |
| **D** | Target 模型原生围栏基线 | 无（原生围栏） | Qwen3-8B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → 冻结的 Qwen3-8B Safety Head | 0.9245 | **0.9228** |
| **E** | 在 Target 模型的 S-Eval Train 上训练 Adapter，并测试 Target 模型生成的 Response | Qwen3-8B S-Eval Train | Qwen3-8B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-8B Safety Head | 0.9185 | **0.9121** |
| **F** | 复用 E 的 Adapter，并测试 Source 模型生成的 Response | Qwen3-8B S-Eval Train（与 E 共用 Adapter） | Qwen3-8B S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-8B Safety Head | 0.9177 | **0.9199** |

**说明**：B 使用 Llama-3.1-8B-Instruct S-Eval Train 训练的 Adapter；E、F 共用 Qwen3-8B S-Eval Train 训练的另一个 Adapter。C 需要运行 8B 以取得真实表征，仅作诊断基线。A 使用 Llama 原生 token 序列；B、C、F 在评测时统一以 **Qwen3-8B token 序列**为坐标，因此 C 与 B/F 更适合直接比较。

### 2. Adapter 结构、损失、tokenizer 对齐与训练配置

**适配目标**：冻结两底座模型及旧围栏，只训练 `Llama-3.1-8B Layer20 → Qwen3-8B Layer20` 的映射。

- **网络**：`torch.nn.Linear(4096, 4096, bias=True)`，**16,781,312 个**可训练参数（约 16.78M）。
- **输入/监督**：同一条 *prompt + response* 分别经冻结的 Llama、8B 提取 hidden states；Llama 表征为输入、8B 表征为监督目标。按训练集统计的逐特征均值、标准差归一化；进入冻结围栏前反归一化。
- **损失函数**：仅使用**精确字符边界配对**的归一化表征 MSE；不用安全标签、风险蒸馏、交叉熵或多层注意力。

```text
x = (h_Llama - mean_Llama) / std_Llama
y = (h_Qwen8 - mean_Qwen8) / std_Qwen8
y_pred = Linear(x)                    # 4096 -> 4096
L_align = MSE(y_pred, y)              # 仅对训练时精确匹配的 token 计算
h_mapped = y_pred * std_Qwen8 + mean_Qwen8
```

#### Tokenizer 对齐（与实验一的关键区别）

Qwen3-14B 与 Qwen3-8B 在上一轮抽样审计中 token ID 和位置一致，直接按 token 位置配对；**Llama 与 Qwen3-8B 的 tokenizer、chat template 不同，不能按 token 下标配对**。本轮代码按以下规则处理：

1. **分别定位文本范围**：用各自 tokenizer 的 `offset_mapping`，在各自 chat template 中找到原始 `prompt`、`response` 的字符区间；分别以 prompt 起点和 response 起点计算**相对字符结束位置**，不跨两种 chat template 直接比较全局字符下标。
2. **训练 Adapter：严格匹配边界**。仅当 Llama 与 Qwen3-8B 的内容 token 在**同一角色内具有相同的相对字符结束位置**时，才将其 hidden states 组成一对用于 MSE。提示模板中的特殊 token／角色头不参与 MSE；每样本从匹配对中最多抽取 128 个，兼顾 prompt 与 response。
3. **测试 Adapter：以 Qwen3-8B token 坐标为准**。遍历 Qwen3-8B 的 token 位置，选取字符结束位置**不晚于当前 Qwen token**的、最近一个**已完成**的 Llama token hidden state；若当前 response 尚无已完成的 Llama 内容 token，则使用 Llama 的 assistant 前缀末状态。对其他特殊 token／角色头使用脚本规定的前缀状态回填。这样得到与 Qwen3-8B 相同长度的 4096D 映射输入。
4. **围栏评测**：使用 Qwen3-8B 的 `assistant_start`、token 序列长度和标签位置运行冻结围栏；C 直接输入相同 Qwen token 坐标下的真实 8B 表征。**响应内容 token 的对齐不读取未来才完成的 Llama token**，但这仍是基于完整已知 Response 的**离线 Streaming F1 模拟**，不是严格在线部署的时延证明。

代码为每条样本缓存 `src_for_qwen`（测试时的索引映射）、`exact_qwen` / `exact_llama`（训练时的精确配对），并在日志中输出各数据划分的 `exact-boundary coverage`。精确字符边界匹配率的实际统计见**实验三**；该比例不等于 token ID 一致率，也不能替代逐 token 对齐正确性检查。

| 训练参数 | B / E、F 共用配置 |
|---|---|
| 训练/验证样本 | 8100 / 900（相应 Train 的 9000 条划分，seed=42） |
| Epoch / Optimizer | 3 / AdamW |
| LR / Weight decay | `3e-4` / `0` |
| 梯度累积 | 8 条样本/更新 |
| Token 抽样 | 每样本最多 128 对**精确字符结束位置匹配**的 token，兼顾 prompt 与 response |
| 归一化统计 | 训练子集前 320 条样本的匹配 token 抽样估计 |
| Checkpoint 选择 | 验证集 normalized MSE 最低 |
| 冻结部分 | Llama、Qwen3-8B 底座与既有 Qwen3-8B Safety Head |

E/F 最优 Adapter：**epoch 3，验证 normalized MSE = 0.51734097**。B 的最佳验证 MSE 待从 `metrics_b.json` 核对。

**评测规则**：Response-level 使用 `pred[-2]`；Streaming-level 使用 `max(pred)`。同一 Llama-3.1-8B-Instruct S-Eval Test 上，F 的 Streaming F1 为 **0.9199**，较 C（0.9253）低 **0.0054**；B 为 **0.8973**。该差异仍需结合 token 对齐覆盖率及多随机种子分析。

### 3. 代码与实验产物位置

以下路径相对于 `~/plugguard_repro/`，运行时设置 `CUDA_VISIBLE_DEVICES=0`。

| 实验 | 运行代码 | Checkpoint / 结果文件 | 日志 |
|---|---|---|---|
| **A** | `Kelp/train.py`、`Kelp/eval.py` | `checkpoints/llama_3_1_8b_seval/model_epoch_0.pt` | `logs/llama_3_1_8b_seval_train.log` |
| **B** | `Kelp/affine_llama_to_qwen8_bcef.py`（`--mode train-b`、`eval-b`） | `checkpoints/adapter_llama3_1_8b_to_qwen3_8b_layer20/train_on_llama/affine_best.pt`、`metrics_b.json` | `logs/adapter_llama_to_qwen8_bcef.log` |
| **C** | 同上（`--mode eval-c`） | 复用 `checkpoints/qwen3_8b_seval/model_epoch_0.pt`；`metrics_c.json` | 同上 |
| **D** | `Kelp/train.py`、`Kelp/eval.py` | `checkpoints/qwen3_8b_seval/model_epoch_0.pt` | `logs/qwen3_8b_seval_train.log` |
| **E/F** | 同上（`--mode train-legacy`、`eval-e`、`eval-f`） | `checkpoints/adapter_llama3_1_8b_to_qwen3_8b_layer20/train_on_qwen8/affine_best.pt`、`metrics_e.json`、`metrics_f.json` | `logs/adapter_llama_to_qwen8_bcef.log` |

- B/C/E/F 的 `metrics_*.json`、`predictions_*.csv` 保存在 `checkpoints/adapter_llama3_1_8b_to_qwen3_8b_layer20/` 根目录。
- token 对齐索引缓存：`data/llama_qwen8_char_alignment_l20/{llama_train,llama_test,qwen_train,qwen_test}/`。
- **自动运行入口**：`Kelp/run_affine_llama_bcef.sh`；在 `Kelp` 目录执行 `bash run_affine_llama_bcef.sh`，顺序完成 B → C → E → F。

---

## 实验三：Adapter 训练 Token 对数量与对齐覆盖率审计

### 1. 审计目的与统计口径

本实验是对**实验一（Qwen3-14B → Qwen3-8B）和实验二（Llama-3.1-8B → Qwen3-8B）已完成训练的只读审计**，不重新训练 Adapter，也不重新评测 Safety Head。目的是核实：两种 token 配对协议下，真正能够进入归一化 MSE 的候选 token 对有多少，以及训练 3 个 Epoch 实际抽取过多少不同位置对。

- **工具与数据**：`Kelp/audit_adapter_token_pairs.py`；读取现有 hidden-state 缓存及 Llama 字符对齐索引缓存；无需 GPU。
- **统计配置**：`seed=42`，每种训练 Response 均为 9000 条，其中 Train 8100 / Validation 900；每样本最多抽取 128 对，训练 3 Epoch；归一化统计使用训练子集前 320 条。
- **INDEX（实验一）**：Qwen3-14B 和 Qwen3-8B 按相同 token 下标配对；候选位置**包含 chat template／特殊 token**。
- **EXACT（实验二）**：Llama 与 Qwen3-8B 按相同角色内、相同相对字符结束位置精确配对；仅统计**内容 token**，排除模板／特殊 token。
- **累计使用量**：3 Epoch 中每次抽到的 token 对均计一次，允许不同 Epoch 重复抽中。
- **去重数量**：对同一训练样本，把 3 Epoch 抽中的 **Qwen3-8B token 位置索引**取并集，再跨样本求和；是不同的**样本－位置对**数量，**不是**去重后的词表 token 或文本内容。抽样轨迹由训练脚本一致的种子与选择函数重建，并非训练时单独持久化的逐步记录。

### 2. 四种 Adapter 训练设置的数量统计

| 模型迁移方向 | Adapter 训练 Response（对应实验） | Train 候选对（8100 条） | 每 Epoch 实际抽取 | 3 Epoch 累计抽取 | 3 Epoch 去重 | 去重 / 累计 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-14B → Qwen3-8B | Qwen3-14B S-Eval Train（B） | 4,686,881 | 1,023,752 | 3,071,256 | 2,002,598 | 65.20% |
| Qwen3-14B → Qwen3-8B | Qwen3-8B S-Eval Train（E/F） | 4,792,878 | 1,025,330 | 3,075,990 | 1,986,770 | 64.59% |
| Llama-3.1-8B-Instruct → Qwen3-8B | Llama-3.1-8B-Instruct S-Eval Train（B） | 2,262,223 | **686,963** | 2,060,889 | **1,250,653** | 60.69% |
| Llama-3.1-8B-Instruct → Qwen3-8B | Qwen3-8B S-Eval Train（E/F） | 4,661,962 | 1,015,354 | 3,046,062 | 1,980,889 | 65.03% |

每 Epoch 的理论抽取上限为 `8100 × 128 = 1,036,800` 对。实际抽取量占上限的比例依次为 **98.74%、98.89%、66.26%、97.93%**；对应平均每条训练样本抽取 **126.39、126.58、84.81、125.35** 对。达不到上限主要是部分样本可选位置不足，尤其抽样还受 prompt/response 配额限制。

**进一步拆分（每 Epoch）：**

| 训练设置 | Prompt／前缀侧抽取 | Response 侧抽取 | Validation 每次评测抽取 | 归一化统计（320 条）抽取 |
|---|---:|---:|---:|---:|
| Qwen3-14B → Qwen3-8B；Qwen3-14B S-Eval Train（B） | 234,631 | 789,121 | 113,781 | 40,282 |
| Qwen3-14B → Qwen3-8B；Qwen3-8B S-Eval Train（E/F） | 234,631 | 790,699 | 114,082 | 40,362 |
| Llama-3.1-8B-Instruct → Qwen3-8B；Llama-3.1-8B-Instruct S-Eval Train（B） | 184,755 | 502,208 | 76,368 | 27,369 |
| Llama-3.1-8B-Instruct → Qwen3-8B；Qwen3-8B S-Eval Train（E/F） | 184,755 | 830,599 | 113,081 | 39,974 |

> 表中的“Prompt／前缀侧”口径不同：INDEX 包含模板／特殊 token；EXACT 只包含通过边界匹配的内容 token。因此两类方法的总量和构成不能直接视为完全同质的训练信息。

### 3. 字符边界匹配覆盖率

实验二的 EXACT 对齐以 **Qwen3-8B 内容 token 位置**为分母。定义：

```text
Exact-boundary coverage = 精确匹配的 Qwen 内容 token 位置数 / Qwen 内容 token 位置总数
```

| 配对数据（9000 条 Train Response 来源） | Qwen 内容 token 位置 | 精确边界配对位置 | 覆盖率 |
|---|---:|---:|---:|
| Llama-3.1-8B-Instruct S-Eval Train | 2,531,326 | 2,509,603 | **99.14%** |
| Qwen3-8B S-Eval Train | 5,220,564 | 5,183,189 | **99.28%** |

这表示在**当前对齐索引和边界定义下**，大多数 Qwen 内容 token 的结束字符位置都能匹配到 Llama 内容 token 的结束位置；不意味着两模型的 token ID、切分起点或语义单位有 99% 相同，也**尚未验证索引构造在所有样本上的正确性**。后续应抽取真实 token 文本、字符区间及配对索引人工核验。

另外，审计脚本 EXACT 模式打印的 `source-token positions` 对应映射缓存中的 `num_qwen`（Qwen 坐标网格长度），**不是 Llama tokenizer 的 token 总数**；解释日志时应以 `Qwen content-token positions` 和 `eligible paired positions` 计算上述覆盖率。

### 4. 当前发现与尚不能得出的结论

1. **实验二 B 与 F 的训练数据规模不同**：Llama / B 每 Epoch 抽取 686,963 对，而 Llama / E/F 抽取 1,015,354 对；前者少 **32.34%**。跨 3 Epoch 去重数量分别为 1,250,653 与 1,980,889，前者少 **36.86%**。两组使用不同来源的 Response，不能单独把 F1 差异归因为 token 数量。
2. **两轮 F 的训练 token 对数量接近，评测 F1 却相差明显**：Qwen3-14B / E/F 与 Llama / E/F 的去重位置对分别为 **1,986,770** 和 **1,980,889**，仅差 **5,881 对（约 0.30%）**；但各自 F 的 Streaming F1 分别为 **0.8119** 和 **0.9199**，相差 **10.80 个百分点**。因此，单靠抽取数量不能解释该性能差异。
3. **尚存在重要混杂因素**：两轮 F 使用不同的新模型 Response 测试集，且训练 token 选择协议不同（INDEX 含特殊 token；EXACT 排除特殊 token），同时模型表征本身和 token 长度分布也不同。因此不能据此直接认定 Llama 表征更易线性映射。
4. **下一步**：先做边界匹配逐 token 正确性检查，再对 Qwen3-14B 进行受控协议消融：保持原始评测不变、仅改训练内容 token 筛选；然后单独检验测试端字符映射和特殊 token 回填的影响。复用原有围栏与缓存，另存 Adapter，不覆盖实验一 checkpoint。

### 5. 审计代码与结果日志

相对于 `~/plugguard_repro/`：

- **脚本**：`Kelp/audit_adapter_token_pairs.py`（只读，不加载底座模型）。
- **完整去重审计日志**：`Kelp/token_audit_unique.log`（用户执行 `--unique` 后已生成）。
- **运行命令**：

```bash
cd ~/plugguard_repro/Kelp
python audit_adapter_token_pairs.py --unique > token_audit_unique.log 2>&1
grep -E '^== |DISTINCT paired positions' token_audit_unique.log
```

该审计提供的是**训练样本位置抽样及边界匹配的数量证据**，不是新的 Adapter checkpoint 或新的 Response/Streaming F1 实验结果。

---

## 实验四：Qwen3-14B → Qwen3-8B 的 Token 对齐协议消融（P0 / P1 / P2）

### 1. 实验目的与控制变量

实验三发现，实验一和实验二不仅使用不同底座模型，也采用不同的训练 token 选择与测试序列映射协议。实验四在 **Qwen3-14B → Qwen3-8B** 这一固定迁移任务上，比较三种协议，检查这些实现选择对冻结围栏的检测性能有多大影响。

- **固定条件**：StreamGuardBench v1 / S-Eval；14B 与 8B 均取 Layer 20（5120D → 4096D）；`torch.nn.Linear(5120, 4096, bias=True)`；归一化 MSE；训练/验证 8100/900，`seed=42`，3 Epoch，AdamW，`lr=3e-4`，权重衰减 0，梯度累积 8，每样本最多 128 对 token；冻结 `checkpoints/qwen3_8b_seval/model_epoch_0.pt`。使用 GPU 0，复用已有 hidden-state 缓存，不重新生成 Response，也不重新训练 Safety Head。
- **训练数据与测试数据**：B 用 **Qwen3-14B S-Eval Train** 训练、**Qwen3-14B S-Eval Test** 评测；E/F 共用用 **Qwen3-8B S-Eval Train** 训练的 Adapter，E 测 **Qwen3-8B S-Eval Test**，F 测 **Qwen3-14B S-Eval Test**。C 在 Qwen3-14B S-Eval Test 上直接使用真实 8B hidden states，不通过 Adapter，作为一致性基线。
- **指标**：Response-level F1 取 `pred[-2]`，Streaming F1 取 `max(pred)`（只要任一响应 token 预测为 harmful 即判 harmful）；下表 F1 均针对 harmful（label=1），测试集均为 1000 条，属于**已有完整 Response 上的离线评测**。

| 协议 | Adapter 的训练 token 对 | 测试时输入冻结 8B 围栏的表征 | 与原实验的关系 |
|---|---|---|---|
| **P0：原始 INDEX** | 14B 与 8B 按 token 下标直接配对；包含 chat template／特殊 token | 14B 原始 hidden-state 序列经 Adapter 映射，保持原 token 下标 | **实验一已有结果**，本次不重跑 |
| **P1：内容 token 训练** | 仅用同一角色、相同相对字符结束位置精确匹配的内容 token；排除模板／特殊 token | 仍使用**原始 14B hidden-state 序列**经新 Adapter 映射 | **重新训练 B 和 E/F 两个 Adapter**，评测 B/E/F |
| **P2：字符映射测试** | **完全复用 P1 训练好的 Adapter**，不再训练 | 先按字符边界建立 `src_for_qwen`，将 14B hidden states 重排／回填至 8B token 坐标，再送入同一 Adapter 和围栏 | 与 P1 **只改变测试输入构造**，评测 B/E/F |

**比较边界**：P0 → P1 不只是删除特殊 token，还改变了候选 token 池、随机抽样和归一化统计使用的样本位置，故不能把差异归因于单一因素；P1 → P2 共用**完全相同的 Adapter checkpoint**，其差异更直接对应测试时输入映射的改变。C 不参与 Adapter 训练，其 P1/P2 列只是重复展示本次重新核验的 C 结果。

### 2. P0 / P1 / P2 全部评测结果

| 实验（Adapter 训练 Response → 测试 Response） | Safety Head 训练数据 | P0 Response F1 | P0 Streaming F1 | P1 Response F1 | P1 Streaming F1 | P2 Response F1 | P2 Streaming F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| **B**（Qwen3-14B S-Eval Train → Qwen3-14B S-Eval Test） | Qwen3-8B S-Eval Train（9000 条） | 0.8182 | 0.8054 | **0.8931** | **0.8796** | 0.8705 | 0.8757 |
| **C**（不训练 Adapter；使用真实 Qwen3-8B 表征；Qwen3-14B S-Eval Test） | Qwen3-8B S-Eval Train（9000 条） | 0.9080 | 0.9007 | 0.9080 | 0.9007 | 0.9080 | 0.9007 |
| **E**（Qwen3-8B S-Eval Train → Qwen3-8B S-Eval Test） | Qwen3-8B S-Eval Train（9000 条） | 0.8247 | 0.8310 | 0.8382 | 0.8338 | **0.8577** | **0.8731** |
| **F**（Qwen3-8B S-Eval Train → Qwen3-14B S-Eval Test） | Qwen3-8B S-Eval Train（9000 条） | 0.8121 | 0.8119 | 0.8133 | 0.7959 | **0.8319** | **0.8481** |

**结果来源**：P0 均抄录自实验一，并非本次重新执行；P1、P2 为实验四实测；C 本次重新执行并得到 Response F1 = `0.9080`、Streaming F1 = `0.9007`，日志显示 `C CONSISTENCY PASSED vs experiment-1 reference`。

为便于比较，下表的正负号表示 **Streaming F1 差值（百分点，pp）**，不是相对百分比：

| 实验 | P1 − P0 | P2 − P1 | P2 − P0 |
|---|---:|---:|---:|
| B | **+7.42 pp** | −0.39 pp | +7.03 pp |
| C | 0.00 pp | 0.00 pp | 0.00 pp |
| E | +0.28 pp | **+3.93 pp** | +4.21 pp |
| F | −1.60 pp | **+5.22 pp** | +3.62 pp |

- **B**：P1 比 P0 的 Streaming F1 提高 7.42 pp；P2 略低于 P1。改变训练 token 池后，Qwen3-14B S-Eval Train 场景的表现明显变化。
- **E**：P1 相对 P0 变化很小，P2 在同一个 Adapter 上比 P1 高 3.93 pp。
- **F（主要迁移场景）**：P1 的 Streaming F1 为 **0.7959**，低于 P0 的 **0.8119**；保持 P1 Adapter 不变，改为 P2 评测后提高到 **0.8481**（比 P1 高 5.22 pp，比 P0 高 3.62 pp）。与 C 的 **0.9007** 仍相差 **5.26 pp**，相比 P0 与 C 的 **8.88 pp** 差距有所缩小。
- **C**：本次复核结果与实验一一致，说明同一 Qwen3-14B S-Eval Test 上的真实 8B 表征诊断基线保持稳定；它不单独证明所有映射索引的合理性。

### 3. Tokenizer 与字符映射审计

#### 3.1 内容 token 精确匹配率与 Token ID 检查

实验四针对四个数据划分构建了字符映射索引，逐样本核对 14B／8B 缓存长度、`assistant_start`，并重新对比两模型的 token ID。`exact` 只统计 prompt/response 的**内容 token**，不含模板与特殊 token。

| 数据划分 | 样本数 | 精确匹配内容 token | 内容 token 覆盖率 | P2 映射后下标改变数（`grid_rewrites`） | 非精确内容位置的改变（`non_exact_rewrites`） | Token ID 不一致 |
|---|---:|---:|---:|---:|---:|---:|
| `qwen14_train` | 9000 | 5,102,465 / 5,102,465 | **100.00%** | 98,036 | 90,000 | **0** |
| `qwen14_test` | 1000 | 616,596 / 616,596 | **100.00%** | 11,369 | 10,000 | **0** |
| `qwen8_train` | 9000 | 5,220,564 / 5,220,564 | **100.00%** | 102,490 | 90,000 | **0** |
| `qwen8_test` | 1000 | 649,965 / 649,965 | **100.00%** | 11,891 | 10,000 | **0** |

**判断**：本次所用全部样本的 Qwen3-14B／Qwen3-8B token ID 相同，内容 token 的字符结束位置也全部精确匹配；因此实验一的按下标配对在**当前样本集的 token 坐标层面**有依据，P2 不是为修复这两套 Qwen tokenizer 的大规模错位。尽管如此，P2 的回填／索引选择仍然改变了部分位置，不能认为“100% 匹配”就意味着 P1/P2 输入完全一致。

#### 3.2 P2 评测时实际改变了多少输入位置

| 评测组 | Test 数据 | 序列位置总数 | P2 相对 P1 改变的来源索引 | 改变比例 | 其中属于精确匹配内容 token 的改变 | 其他位置的改变 |
|---|---|---:|---:|---:|---:|---:|
| **B / F** | Qwen3-14B S-Eval Test | 628,596 | 11,369 | **1.81%** | 1,369 | 10,000 |
| **E** | Qwen3-8B S-Eval Test | 661,965 | 11,891 | **1.80%** | 1,891 | 10,000 |

这里的“改变”严格指 `src_for_qwen[j] != j`，即**选择了不同的源 hidden-state 下标**；它不是数值变化的范数，也不能直接等同于特殊 token 的数量。`non_exact_rewrites` 包含所有不属于精确匹配内容 token 集合的改动，需要区分模板、分隔符、未纳入内容集合的位置等来源。

进一步的索引方向检查（仅为定位映射行为）：

| 评测 Test | 下标增大 `src[j] > j` | 其中 Response 阶段 | 下标减小 `src[j] < j` | Response 阶段全部改变 |
|---|---:|---:|---:|---:|
| Qwen3-14B S-Eval Test | 7,369 | 1,369 | 4,000 | 5,369 |
| Qwen3-8B S-Eval Test | 7,891 | 1,891 | 4,000 | 5,891 |

这些数字仅说明**离线测试时构造的表征序列采用了不同的源位置**；不能仅凭 token 下标大小，就认定训练违规、评测无效或某种性能提升来自信息泄漏。本次研究报告的是完整已知 Response 上的离线 Response/Streaming F1；若将来讨论逐 token 在线部署，需要另外定义并验证在线可用性，不能与本次离线实验混为一谈。

### 4. P1 → P2 的逐样本预测变化

P1、P2 **使用同一 Adapter**，但 P2 改变了约 1.8% 的源表征索引。对同一测试集逐样本比较 `predictions_*.csv` 得到：

| 实验 | Response-level 预测改变（1000 条中） | Streaming-level 预测改变（1000 条中） | Streaming F1：P1 → P2 |
|---|---:|---:|---:|
| **B** | **34**（3.4%） | **5**（0.5%） | 0.8796 → 0.8757 |
| **E** | **73**（7.3%） | **123**（12.3%） | 0.8338 → 0.8731 |
| **F** | **95**（9.5%） | **148**（14.8%） | 0.7959 → 0.8481 |

说明输入端仅有约 1.8% 的索引发生改变，但 E/F 的完整序列判别结果受到明显影响。这与围栏包含跨 token 注意力和时序状态传播的实现相容，但**本次统计尚未分离**不同位置类型的独立贡献，也未统计每次翻转对 Precision／Recall 的方向性影响。不能据此断定“特殊 token 回填”是唯一原因。

### 5. 阶段性结论与实验边界

1. **对齐协议是一个实证上重要的控制变量**。在完全相同的 Qwen3-14B → Qwen3-8B 迁移任务上，B 的 P0 → P1 Streaming F1 提高 **7.42 pp**；F 的 P1 → P2 提高 **5.22 pp**。不能把实验一和实验二的绝对 F1 差异全归因于模型表征可对齐性。
2. **效果依赖训练 Response 来源，不能统一概括为 P2 更优**。B 的 P1（0.8796）优于 P2（0.8757）；E/F 则 P2 优于 P1。对于 F，单独改变训练 token 池的 P1 反而低于 P0。
3. **P2 改动少、预测改变多，但原因尚未拆解**：内容 token 在当前 Qwen 数据上 100% 精确匹配，P2 仍更改约 1.8% 的来源索引，并改变 E/F 约 12%～15% 的 Streaming 判定。当前只能确认不同序列构造方式与 F1 变化存在对应关系，尚不能给出更细的机制归因。
4. **与 Llama 结果的比较仍受测试集和协议混杂影响**。Llama 实验 F 的 Streaming F1 为 0.9199，而本实验 Qwen3-14B 的 F-P2 为 0.8481；两者测试 Response 来源不同，不宜直接解读为两个模型线性表征迁移难度的严格排序。
5. **统计与评测局限**：本次尚无多随机种子、置信区间或逐样本 Precision/Recall 翻转分析；P0 数值来自历史实验；本次没有重新生成 Response，也未测试 MLP／MATCHA。**不将 P2 的索引方向统计直接作为离线实验有效性判定。**

### 6. 代码、Checkpoint、结果与复现方式

以下路径均相对于 `~/plugguard_repro/`：

| 类型 | 路径／内容 |
|---|---|
| 主代码 | `Kelp/exp4_qwen14_token_alignment_ablation.py` |
| 自动运行入口 | `Kelp/run_exp4_qwen14_token_alignment.sh` |
| 日志 | `logs/exp4_qwen14_token_alignment.log` |
| 字符索引缓存 | `data/qwen14_qwen8_char_alignment_l20/{qwen14_train,qwen14_test,qwen8_train,qwen8_test}/sample_XXXXXXXX.pt` |
| P1/P2 的 B Adapter | `checkpoints/exp4_qwen14_to_qwen8_alignment_ablation/train_on_qwen14/affine_best.pt` |
| P1/P2 的 E、F 共用 Adapter | `checkpoints/exp4_qwen14_to_qwen8_alignment_ablation/train_on_qwen8/affine_best.pt` |
| F1 与 Accuracy | `checkpoints/exp4_qwen14_to_qwen8_alignment_ablation/metrics_{b,e,f}_{p1,p2}.json`、`metrics_c.json` |
| 逐样本预测 | 同上目录的 `predictions_{b,e,f}_{p1,p2}.csv`、`predictions_c.csv` |

历史 P0 的 Adapter 和结果文件仍保存在**实验一**原路径，实验四没有覆盖它们。实验四执行完毕，运行日志结尾为 `=== ALL EXPERIMENT 4 STEPS COMPLETE ===`。

无需重跑训练，可在服务器的 `Kelp/` 目录直接复核结果汇总：

```bash
python exp4_qwen14_token_alignment_ablation.py --mode report
```

**实验四已完成并记录**；后续若要讨论机制，可单独做特定位置类型的消融，但不作为本次实验四已有结论或必要补跑条件。
---

## 实验五：Qwen3-8B → Qwen3-14B

### 1. 实验设置与结果

- **迁移**：Source 为 Qwen3-8B（Layer 20，4096D），Target 为 Qwen3-14B（Layer 20，5120D）；冻结 `checkpoints/qwen3_14b_seval/model_epoch_0.pt`，训练 `Linear(4096, 5120, bias=True)` Adapter。
- **数据**：B 用 Qwen3-8B S-Eval Train 训练 Adapter、Qwen3-8B S-Eval Test 评测；C 在相同 Qwen3-8B S-Eval Test 文本上直接使用真实 14B 表征；E 用 Qwen3-14B S-Eval Train 训练另一个 Adapter、Qwen3-14B S-Eval Test 评测；F 复用 E 的 Adapter，在 Qwen3-8B S-Eval Test 评测。各 Test 均为 1000 条。

| 实验 | Adapter 训练 Response | Safety Head 训练数据 | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---:|---:|
| **A** | 无（原生围栏） | Qwen3-8B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → 冻结的 Qwen3-8B Safety Head | 0.9245 | **0.9228** |
| **B** | Qwen3-8B S-Eval Train | Qwen3-14B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-14B Safety Head | 0.9031 | **0.8938** |
| **C** | 无（不使用 Adapter） | Qwen3-14B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-14B 对同一 Qwen3-8B Response 提取的真实 Layer 20 hidden states（5120D）→ 冻结的 Qwen3-14B Safety Head | 0.9101 | **0.9124** |
| **D** | 无（原生围栏） | Qwen3-14B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → 冻结的 Qwen3-14B Safety Head | 0.8955 | **0.8971** |
| **E** | Qwen3-14B S-Eval Train | Qwen3-14B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-14B Safety Head | 0.8779 | **0.8628** |
| **F** | Qwen3-14B S-Eval Train（与 E 共用 Adapter） | Qwen3-14B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-14B Safety Head | 0.8997 | **0.8885** |

### 2. 本轮统一协议（实验五～八共用）

以下配置也适用于实验六、七、八，仅 Source、Target、Adapter 维度及冻结的目标围栏不同。

- **训练**：两模型分别对同一条已存储的 *prompt + response* 提取 Layer 20 hidden states；按各自 chat template 定位 prompt、response，仅配对**同一角色中相对字符结束位置完全一致的内容 token**；不使用安全标签训练 Adapter，排除模板／特殊 token 的 MSE 监督。
- **Adapter 与优化**：`torch.nn.Linear(source_dim, target_dim, bias=True)`；对 Source/Target 表征按训练样本统计的逐特征均值、标准差归一化，以 normalized MSE 训练，送入目标围栏前反归一化。Train 9000 划分为训练 8100／验证 900（`seed=42`），3 Epoch，AdamW（`lr=3e-4`、`weight_decay=0`），梯度累积 8，每样本最多抽 128 对（Prompt 优先最多 32 对，其余从 Response 抽取），用训练子集前 320 条估计归一化统计量；按验证 normalized MSE 最低选择 checkpoint。
- **评测对齐**：统一以 **Target tokenizer 的 token 网格**及 Target 缓存中的 `assistant_start`、标签为准；Source 内容 token 依据相同文本片段的相对字符结束位置进行索引映射，其他模板／前缀位置按脚本策略回填。B/E/F 使用映射后的 Source 表征经 Adapter 输入**冻结的 Target Safety Head**；C 不经 Adapter，直接使用 Target 真实表征。
- **指标**：均为 harmful（label=1）类别的 F1；Response-level 使用 `pred[-2]`，Streaming-level 使用 `max(pred)`。属于**完整 Response 已知的离线评测**，没有在这里进行在线时延实验。A、D 原生围栏结果沿用先前复现，不在本轮重训。

### 3. 代码与产物

- **公共代码**：`Kelp/cross_adapter_alignment.py`、`Kelp/cross_adapter_training.py`；四方向串行入口：`Kelp/run_exp5_to_exp8_sequential.sh`（本次运行将可见设备设置为物理 GPU 1）。
- **本组入口**：`Kelp/exp5_qwen8_to_qwen14.py`；组日志：`logs/exp5_qwen8_to_qwen14.log`。
- **结果目录**：`checkpoints/cross_adapt_four_directions_v1/qwen8_to_qwen14/`；B 的 Adapter 为 `train_on_src_train/affine_best.pt`，E/F 共用 Adapter 为 `train_on_tgt_train/affine_best.pt`；四组指标与逐样本预测分别为 `metrics_{b,c,e,f}.json`、`predictions_{b,c,e,f}.csv`。

---

## 实验六：Llama-3.1-8B-Instruct → Qwen3-14B

### 1. 实验设置与结果

- **迁移**：Llama-3.1-8B-Instruct（Layer 20，4096D）→ Qwen3-14B（Layer 20，5120D），`Linear(4096, 5120, bias=True)`；冻结 **Qwen3-14B Safety Head**。训练、字符配对和离线评测规则同实验五第 2 节。

| 实验 | Adapter 训练 Response | Safety Head 训练数据 | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---:|---:|
| **A** | 无（原生围栏） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.9095 | **0.9191** |
| **B** | Llama-3.1-8B-Instruct S-Eval Train | Qwen3-14B S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-14B Safety Head | 0.9157 | **0.9227** |
| **C** | 无（不使用 Adapter） | Qwen3-14B S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Qwen3-14B 对同一 Llama-3.1-8B-Instruct Response 提取的真实 Layer 20 hidden states（5120D）→ 冻结的 Qwen3-14B Safety Head | 0.8954 | **0.9000** |
| **D** | 无（原生围栏） | Qwen3-14B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → 冻结的 Qwen3-14B Safety Head | 0.8955 | **0.8971** |
| **E** | Qwen3-14B S-Eval Train | Qwen3-14B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-14B Safety Head | 0.8747 | **0.8576** |
| **F** | Qwen3-14B S-Eval Train（与 E 共用 Adapter） | Qwen3-14B S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Qwen3-14B Safety Head | 0.9081 | **0.9133** |

### 2. 代码与产物

- **入口**：`Kelp/exp6_llama_to_qwen14.py`；组日志：`logs/exp6_llama_to_qwen14.log`。
- **结果目录**：`checkpoints/cross_adapt_four_directions_v1/llama_to_qwen14/`；B 与 E/F 的 Adapter 分别为 `train_on_src_train/affine_best.pt`、`train_on_tgt_train/affine_best.pt`；四组 `metrics_{b,c,e,f}.json`、`predictions_{b,c,e,f}.csv`。
- **运行备注**：`prepare-maps` 曾因个别样本的原始 prompt/response 无法按模板原文精确定位而中断；处理该文本定位问题后，串行流程继续并最终完成。具体实现以服务器上实际运行的修复版代码为准。

---

## 实验七：Qwen3-8B → Llama-3.1-8B-Instruct

### 1. 实验设置与结果

- **迁移**：Qwen3-8B（Layer 20，4096D）→ Llama-3.1-8B-Instruct（Layer 20，4096D），`Linear(4096, 4096, bias=True)`；冻结 **Llama-3.1-8B Safety Head**。训练、字符配对和离线评测规则同实验五第 2 节；评测坐标以 **Llama tokenizer** 为准。

| 实验 | Adapter 训练 Response | Safety Head 训练数据 | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---:|---:|
| **A** | 无（原生围栏） | Qwen3-8B S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → 冻结的 Qwen3-8B Safety Head | 0.9245 | **0.9228** |
| **B** | Qwen3-8B S-Eval Train | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.7960 | **0.7470** |
| **C** | 无（不使用 Adapter） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Llama-3.1-8B-Instruct 对同一 Qwen3-8B Response 提取的真实 Layer 20 hidden states（4096D）→ 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.8507 | **0.8198** |
| **D** | 无（原生围栏） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.9095 | **0.9191** |
| **E** | Llama-3.1-8B-Instruct S-Eval Train | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.8004 | **0.7706** |
| **F** | Llama-3.1-8B-Instruct S-Eval Train（与 E 共用 Adapter） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Qwen3-8B S-Eval Test | Qwen3-8B Layer 20 hidden states（4096D） → Affine Adapter → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.7981 | **0.7543** |

### 2. 代码与产物

- **入口**：`Kelp/exp7_qwen8_to_llama.py`；组日志：`logs/exp7_qwen8_to_llama.log`。
- **结果目录**：`checkpoints/cross_adapt_four_directions_v1/qwen8_to_llama/`；B 与 E/F 的 Adapter 分别为 `train_on_src_train/affine_best.pt`、`train_on_tgt_train/affine_best.pt`；四组 `metrics_{b,c,e,f}.json`、`predictions_{b,c,e,f}.csv`。

---

## 实验八：Qwen3-14B → Llama-3.1-8B-Instruct

### 1. 实验设置与结果

- **迁移**：Qwen3-14B（Layer 20，5120D）→ Llama-3.1-8B-Instruct（Layer 20，4096D），`Linear(5120, 4096, bias=True)`；冻结 **Llama-3.1-8B Safety Head**。训练、字符配对和离线评测规则同实验五第 2 节；评测坐标以 **Llama tokenizer** 为准。

| 实验 | Adapter 训练 Response | Safety Head 训练数据 | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---:|---:|
| **A** | 无（原生围栏） | Qwen3-14B S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → 冻结的 Qwen3-14B Safety Head | 0.8955 | **0.8971** |
| **B** | Qwen3-14B S-Eval Train | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → Affine Adapter → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.6812 | **0.6799** |
| **C** | 无（不使用 Adapter） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Llama-3.1-8B-Instruct 对同一 Qwen3-14B Response 提取的真实 Layer 20 hidden states（4096D）→ 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.8223 | **0.7926** |
| **D** | 无（原生围栏） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Llama-3.1-8B-Instruct Layer 20 hidden states（4096D） → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.9095 | **0.9191** |
| **E** | Llama-3.1-8B-Instruct S-Eval Train | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Llama-3.1-8B-Instruct S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → Affine Adapter → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.6261 | **0.6235** |
| **F** | Llama-3.1-8B-Instruct S-Eval Train（与 E 共用 Adapter） | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | Qwen3-14B S-Eval Test | Qwen3-14B Layer 20 hidden states（5120D） → Affine Adapter → 冻结的 Llama-3.1-8B-Instruct Safety Head | 0.6862 | **0.6799** |

### 2. 代码与产物

- **入口**：`Kelp/exp8_qwen14_to_llama.py`；组日志：`logs/exp8_qwen14_to_llama.log`。
- **结果目录**：`checkpoints/cross_adapt_four_directions_v1/qwen14_to_llama/`；B 与 E/F 的 Adapter 分别为 `train_on_src_train/affine_best.pt`、`train_on_tgt_train/affine_best.pt`；四组 `metrics_{b,c,e,f}.json`、`predictions_{b,c,e,f}.csv`。

---

## 实验五～八：结果汇总与记录边界

| 迁移方向 | B/C/E/F 使用的冻结 Safety Head 训练数据 | A | B | C | D | E | F | F − C（pp） |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B → Qwen3-14B | Qwen3-14B S-Eval Train（9000 条） | 0.9228 | 0.8938 | 0.9124 | 0.8971 | 0.8628 | 0.8885 | −2.39 |
| Llama-3.1-8B-Instruct → Qwen3-14B | Qwen3-14B S-Eval Train（9000 条） | 0.9191 | 0.9227 | 0.9000 | 0.8971 | 0.8576 | 0.9133 | +1.33 |
| Qwen3-8B → Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | 0.9228 | 0.7470 | 0.8198 | 0.9191 | 0.7706 | 0.7543 | −6.55 |
| Qwen3-14B → Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct S-Eval Train（9000 条） | 0.8971 | 0.6799 | 0.7926 | 0.9191 | 0.6235 | 0.6799 | −11.27 |

**注**：本表均为 Streaming F1。“Safety Head 训练数据”列描述 B、C、E、F；A 使用 Source 模型自身的 S-Eval Train 训练的原生围栏，D 使用 Target 模型自身的 S-Eval Train 训练的原生围栏。A 和 D 取自 `notes/L20/reproduction.md` 的 L20 原生 S-Eval 复现结果，未在实验五～八重跑；B、C、E、F 取自本轮 16 组评测。A 使用 Source 自身 Test，D 使用 Target 自身 Test，不能将它们视为与 B、C、F 完全相同的测试集。

- **统一汇总文件**：`checkpoints/cross_adapt_four_directions_v1/summary_16_results.csv`；总日志：`logs/exp5_to_exp8_master.log`。日志最终标记为 `ALL 16 METRICS READY` 与 `ALL EXPERIMENTS 5/6/7/8 COMPLETE`（2026-11-09 13:23:06，+08:00）。
- **直接观察**：在各方向各自的 Source Test 上，F 与 C 使用同一个目标围栏，因此可比较表征映射后的结果和真实目标表征结果；本轮 Llama → 14B 的 F 高于 C，而另外三个方向 F 低于 C。**C 不是理论性能上界**。
- **比较限制**：不同方向的测试 Response 来源、Target Safety Head 及字符网格不同，不能仅凭跨方向 F1 排名确定表征可迁移性的因果原因。本轮没有提供多随机种子或置信区间；不作额外机制归因。

---

## 实验九：迁移到 Llama-3.1-8B-Instruct 的 Adapter 学习率扫描与预测偏置检查

### 1. 实验设置

- **目的**：检查迁移到冻结 Llama-3.1-8B-Instruct Safety Head 时，Affine Adapter 的 E/F F1 对学习率的敏感性，以及 Streaming-level harmful 预测偏置。
- **迁移方向**：Qwen3-8B → Llama-3.1-8B-Instruct（4096D → 4096D）与 Qwen3-14B → Llama-3.1-8B-Instruct（5120D → 4096D）。两方向均复用由 **Llama-3.1-8B-Instruct S-Eval Train（9000 条）**训练的冻结 Safety Head。
- **数据与评测**：只训练 E/F 共用的 Adapter，训练 Response 为 Llama-3.1-8B-Instruct S-Eval Train（8100 Train／900 Validation）；E 测 Llama S-Eval Test（453 harmful／547 benign），F 分别测 Qwen3-8B Test（533／467）与 Qwen3-14B Test（515／485），每个 Test 均为 1000 条。指标仍为 harmful（label=1）的 Response/Streaming 口径；本节主要报告 **Streaming F1**。
- **固定参数**：`seed=42`、3 Epoch、AdamW、`weight_decay=0`、梯度累积 8、每样本最多 128 对 token、使用训练子集前 320 条估计归一化统计量；字符边界配对、Llama token 网格和 normalized MSE 训练目标不变，选择 Validation normalized MSE 最低的 checkpoint。
- **第一轮**：两方向均扫描 `5e-5、1e-4、2e-4、3e-4、5e-4`；其中 `3e-4` 直接复用实验七／八，其他四档每方向各训练一个 Adapter（新增 8 次训练，16 次 E/F 评测）。
- **第二轮（2026-11-09，非对称扩展）**：Qwen3-8B → Llama 向上扫描 `7e-4、1e-3、1.5e-3、2e-3、3e-3`；Qwen3-14B → Llama 向下扫描 `4e-5、3e-5、2e-5、1e-5、5e-6`。物理 **GPU 1** 串行执行，新增 10 次 Adapter 训练、20 次 E/F 评测；与第一轮及原基线汇总为 **20 套学习率配置、40 条 E/F 指标**。不重训底座模型或 Safety Head，不覆盖旧 checkpoint。

### 2. 两轮学习率扫描完整结果（20 套配置）

F1 均针对 harmful（label=1）；E、F 每组各测试 1000 条。**最佳 Epoch** 均指 Validation normalized MSE 最低时的 Adapter checkpoint，并非使用测试 F1 选 Epoch；本表所有配置的最佳 Epoch 均为 3。

| 迁移方向 | Adapter 学习率 | 扫描轮次 | 最佳 Epoch | Validation normalized MSE | E Streaming F1 | F Streaming F1 | E FP | F FP | F 预测 harmful 比例 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-8B → Llama-3.1-8B-Instruct | `5e-5` | 第一轮 | 3 | 0.5153845 | 0.6866 | 0.7243 | 402 | 390 | 91.4% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `1e-4` | 第一轮 | 3 | **0.4994888** | 0.7329 | 0.7371 | 318 | 363 | 88.6% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `2e-4` | 第一轮 | 3 | 0.5002444 | 0.7607 | 0.7516 | 272 | 334 | 85.6% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `3e-4` | 原基线复用 | 3 | 0.5052293 | 0.7706 | 0.7543 | 257 | 329 | 85.1% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `5e-4` | 第一轮 | 3 | 0.5156211 | 0.7807 | 0.7598 | 242 | 319 | 84.1% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `7e-4` | 第二轮 | 3 | 0.5256328 | 0.7855 | 0.7613 | 232 | 318 | 84.1% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `1e-3` | 第二轮 | 3 | 0.5403994 | 0.7879 | 0.7623 | 227 | 313 | 83.4% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `1.5e-3` | 第二轮 | 3 | 0.5878416 | 0.7833 | 0.7623 | 232 | 313 | 83.4% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `2e-3` | 第二轮 | 3 | 0.6306917 | 0.7975 | 0.7682 | 212 | 304 | 82.6% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `3e-3` | 第二轮 | 3 | 0.6729172 | **0.8084** | **0.7745** | 197 | 293 | 81.5% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `5e-6` | 第二轮 | 3 | 0.7806590 | **0.7802** | **0.6808** | 249 | 483 | 99.8% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `1e-5` | 第二轮 | 3 | 0.6666630 | 0.7724 | 0.6803 | 259 | 484 | 99.9% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `2e-5` | 第二轮 | 3 | 0.5774219 | 0.7255 | 0.6799 | 334 | 485 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `3e-5` | 第二轮 | 3 | 0.5414123 | 0.6772 | 0.6799 | 424 | 485 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `4e-5` | 第二轮 | 3 | 0.5233644 | 0.6513 | 0.6799 | 483 | 485 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `5e-5` | 第一轮 | 3 | 0.5133033 | 0.6412 | 0.6799 | 507 | 485 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `1e-4` | 第一轮 | 3 | **0.4993510** | 0.6235 | 0.6799 | 547 | 485 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `2e-4` | 第一轮 | 3 | 0.5027213 | 0.6235 | 0.6799 | 547 | 485 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `3e-4` | 原基线复用 | 3 | 0.5107621 | 0.6235 | 0.6799 | 547 | 485 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `5e-4` | 第一轮 | 3 | 0.5290842 | 0.6235 | 0.6799 | 547 | 485 | 100.0% |

**读表提醒**：表中的 E/F 最高 Streaming F1 是对同一 Test 反复扫描后观察到的数值，属于探索性结果，不是独立验证集选定超参数后的无偏最终测试成绩。

### 3. Streaming 错误分布与有害预测偏置

下表的 `TP/FP/FN/TN` 是**Streaming-level 响应级**混淆矩阵，不是 token 级真值统计。原第一轮行沿用已记录的实测混淆矩阵；第二轮仅提供 F1（四位小数）、FP 和类别总数，故其余项按这些数值**唯一反推**并核对到四位小数。若后续有原始 `metrics_*.json`，应以其逐样本统计为准。

| 迁移方向 | LR | E：TP / FP / FN / TN | E 预测 harmful 比例 | F：TP / FP / FN / TN | F 预测 harmful 比例 |
|---|---|---|---:|---|---:|
| Qwen3-8B → Llama-3.1-8B-Instruct | `5e-5` | 447 / 402 / 6 / 145 | 84.9% | 524 / 390 / 9 / 77 | 91.4% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `1e-4` | 446 / 318 / 7 / 229 | 76.4% | 523 / 363 / 10 / 104 | 88.6% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `2e-4` | 445 / 272 / 8 / 275 | 71.7% | 522 / 334 / 11 / 133 | 85.6% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `3e-4` | 445 / 257 / 8 / 290 | 70.2% | 522 / 329 / 11 / 138 | 85.1% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `5e-4` | 445 / 242 / 8 / 305 | 68.7% | 522 / 319 / 11 / 148 | 84.1% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `7e-4` | 443 / 232 / 10 / 315 | 67.5% | 523 / 318 / 10 / 149 | 84.1% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `1e-3` | 442 / 227 / 11 / 320 | 66.9% | 521 / 313 / 12 / 154 | 83.4% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `1.5e-3` | 441 / 232 / 12 / 315 | 67.3% | 521 / 313 / 12 / 154 | 83.4% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `2e-3` | 441 / 212 / 12 / 335 | 65.3% | 522 / 304 / 11 / 163 | 82.6% |
| Qwen3-8B → Llama-3.1-8B-Instruct | `3e-3` | 441 / 197 / 12 / 350 | 63.8% | 522 / 293 / 11 / 174 | 81.5% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `5e-6` | 449 / 249 / 4 / 298 | 69.8% | 515 / 483 / 0 / 2 | 99.8% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `1e-5` | 448 / 259 / 5 / 288 | 70.7% | 515 / 484 / 0 / 1 | 99.9% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `2e-5` | 448 / 334 / 5 / 213 | 78.2% | 515 / 485 / 0 / 0 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `3e-5` | 449 / 424 / 4 / 123 | 87.3% | 515 / 485 / 0 / 0 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `4e-5` | 452 / 483 / 1 / 64 | 93.5% | 515 / 485 / 0 / 0 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `5e-5` | 453 / 507 / 0 / 40 | 96.0% | 515 / 485 / 0 / 0 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `1e-4` | 453 / 547 / 0 / 0 | 100.0% | 515 / 485 / 0 / 0 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `2e-4` | 453 / 547 / 0 / 0 | 100.0% | 515 / 485 / 0 / 0 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `3e-4` | 453 / 547 / 0 / 0 | 100.0% | 515 / 485 / 0 / 0 | 100.0% |
| Qwen3-14B → Llama-3.1-8B-Instruct | `5e-4` | 453 / 547 / 0 / 0 | 100.0% | 515 / 485 / 0 / 0 | 100.0% |

原实验对 **Qwen3-14B → Llama** 还单独检查过部分学习率的 Response-level harmful 数量；下表继续保留这项历史统计（没有为第二轮猜测 Response-level 数值）：

| 学习率 | E Response+ / Streaming+ / Stream only | F Response+ / Streaming+ / Stream only |
|---|---|---|
| `5e-5` | 866 / 960 / 94 | 851 / 1000 / 149 |
| `1e-4` | 935 / 1000 / 65 | 917 / 1000 / 83 |
| `3e-4` | 994 / 1000 / 6 | 986 / 1000 / 14 |
| `5e-4` | 998 / 1000 / 2 | 995 / 1000 / 5 |

其中 `Stream only` 表示 Response-level 判 benign、Streaming-level 判 harmful 的响应数；其他学习率的 Response+ 未在现有输出中提供，故不补写。**Streaming 全量 harmful 只表示每条 Response 至少有一个 token 触发，不等于每个 token 都预测 harmful。**

### 4. 更新后的观察、限制与产物

- **Qwen3-8B → Llama-3.1-8B-Instruct（向上扫描）**：从原基线 `3e-4` 到 `3e-3`，E/F Streaming F1 分别由 `0.7706/0.7543` 提高到 **`0.8084/0.7745`**（+3.78／+2.02 pp），E/F Streaming FP 分别由 `257/329` 降为 **`197/293`**；F 仍低于同一 Qwen3-8B Test 上真实 Llama 表征 C 的 `0.8198`（差 4.53 pp）。本轮最高 F1 位于扫描上界，未验证更高学习率的稳定性。
- **Qwen3-14B → Llama-3.1-8B-Instruct（向下扫描）**：`3e-4 → 5e-6` 时，E Streaming F1 从 `0.6235` 提升至 **`0.7802`**（+15.67 pp），E FP 从 `547` 降至 `249`；但核心 F 仅由 `0.6799` 变为 **`0.6808`**（+0.09 pp），F FP `485 → 483`，仍有 **99.8%** 响应被判 harmful。低学习率改善 E 并不等于解决 F 的迁移退化。
- **Val MSE 与下游 F1 不一致**：Qwen3-8B → Llama 的最低验证 MSE 为 `0.4994888`（`1e-4`），但最高 F F1 来自 `3e-3`（Val MSE `0.6729172`）；Qwen3-14B → Llama 的最低验证 MSE 为 `0.4993510`（`1e-4`），但最低学习率 `5e-6` 的 E F1 最高，且 Val MSE 升至 `0.7806590`。**最小 normalized MSE 不保证最佳下游检测 F1**；不同学习率均为最佳 Epoch 3，低 LR 还可能训练不足。
- **解释边界**：全部为单 seed、3 Epoch；重复使用同一 Test 选择扫描中的最高 F1 存在测试集选择偏差。本实验未证明 F 组预测偏置的因果机制，也不能据此断言 Affine 结构必然不可用。后续正式超参数选择应优先使用独立验证数据的下游检测指标。
- **第一轮代码／日志**：`Kelp/sweep_llama_target_lr.py`、`Kelp/run_llama_target_lr_sweep_gpu1.sh`；`logs/llama_target_lr_sweep_gpu1.log`；`checkpoints/llama_target_lr_sweep_v1/summary_20_results.csv`。
- **第二轮代码／日志**：`Kelp/sweep_llama_target_lr_asymmetric.py`、`Kelp/run_llama_target_lr_asymmetric_gpu1.sh`；`logs/llama_target_lr_asymmetric_gpu1.log`。
- **完整汇总**：`checkpoints/llama_target_lr_sweep_asymmetric_v2/summary_asymmetric_40_results.csv`（**40/40** E/F metrics，2026-11-09 21:37:08 +08:00）；新旧 Adapter checkpoint 分别独立保存于各自实验目录。原 `3e-4` 的记录复用 `checkpoints/cross_adapt_four_directions_v1/`，未重新训练。

---

## 实验十：Qwen3-14B → Llama-3.1-8B-Instruct 的 Prompt/Response 表征替换消融

### 1. 实验设置

- **目的**：在实验八、九发现严重有害类别误报后，定位全序列 Affine 映射中 **Prompt/Header 前缀**与 **Response** 部分各自对冻结 Llama Safety Head 输出的影响。
- **固定条件**：Llama-3.1-8B-Instruct S-Eval Test（1000 条，benign 547 / harmful 453）；Safety Head 训练数据为 **Llama-3.1-8B-Instruct S-Eval Train（9000 条）**，继续冻结。复用实验八 E/F 在 `lr=3e-4`、Llama Train 上训练的 **Qwen3-14B → Llama-3.1-8B-Instruct** Affine Adapter；所有输入均排列在 Llama tokenizer 网格上，以 Llama 的 `assistant_start` 划分前缀与 Response。
- **方法**：直接使用已有的真实 Llama hidden states 和 Qwen3-14B 经 Adapter 映射的 hidden states，拼接成 T0～T3 四种序列；**不重训 Adapter 或 Safety Head，不提取新的底座表征**。四组都使用相同样本、标签和离线评测规则（Response `pred[-2]`；Streaming `max(pred)`）。

### 2. T0～T3 完整结果

| 方案 | Prompt/Header 表征 | Response 表征 | Response F1 | Streaming F1 | Streaming FP |
|---|---|---|---:|---:|---:|
| **T0**（原生 D） | 真实 Llama-3.1-8B-Instruct 表征 | 真实 Llama-3.1-8B-Instruct 表征 | 0.9095 | **0.9191** | 48 |
| **T1**（原实验八 E） | Qwen3-14B 经 Affine 映射 | Qwen3-14B 经 Affine 映射 | 0.6261 | 0.6235 | 547 |
| **T2**（真实前缀） | **真实 Llama-3.1-8B-Instruct 表征** | Qwen3-14B 经 Affine 映射 | **0.9157** | **0.9213** | **47** |
| **T3**（真实响应） | Qwen3-14B 经 Affine 映射 | **真实 Llama-3.1-8B-Instruct 表征** | 0.6283 | 0.6235 | 547 |

**一致性校验**：T1 的逐样本 Response/Streaming 预测与实验八 E 的 `predictions_e.csv` **完全一致**；T0 的 Response/Streaming F1 与 L20 原生 Llama D 基线一致。脚本输出 `PASS: T1 predictions identical to original E; T0 native metrics match L20 D reference`。

### 3. 直接观察与实验边界

1. **只把映射前缀换成真实 Llama 前缀（T1 → T2）**：Response F1 `0.6261→0.9157`，Streaming F1 `0.6235→0.9213`（**+29.78 个百分点**），Streaming FP `547→47`（减少 500 条）。T2 与原生 T0 的 Streaming F1 仅差 `+0.22` 个百分点，未做显著性检验。
2. **只把映射 Response 换成真实 Llama Response（T1 → T3）**：Response F1 仅 `0.6261→0.6283`，Streaming F1 不变（`0.6235`），Streaming FP 仍为 `547`。
3. **阶段性判断**：在固定 Adapter、冻结围栏和相同测试集上，性能退化主要与 **Prompt/Header 映射表征**有关；仅修复 Response 表征不能解除全量 Streaming 有害触发。尚未进一步区分用户 Prompt 正文、chat-template／角色标记、assistant header 的独立影响，也未证明注意力或记忆单元的具体作用机制。
4. **应用边界**：T2/T3 是依赖**真实目标 Llama hidden states** 的诊断性混合输入，不是无需运行旧模型的可部署 Adapter 迁移方案。该结果不代表单纯增加学习率可以解决前缀映射问题。

### 4. 代码与产物

- **脚本**：`Kelp/exp9_llama_prompt_response_ablation.py`、`Kelp/run_exp9_llama_prompt_response_gpu1.sh`（物理 GPU 1；**脚本名称 exp9 沿用执行时命名，本记录按实验开展顺序归档为实验十**）。
- **日志**：`logs/exp9_llama_prompt_response_gpu1.log`。
- **结果目录**：`checkpoints/llama_target_prompt_response_ablation_v1/qwen14_to_llama_lr3e-4/`；`summary.csv`、`metrics_t0.json`～`metrics_t3.json`、`predictions_t0_t1_t2_t3.csv`。

---

## 当前阶段小结（实验一～十）

- **实验一／二／五～八**：已完成 Qwen3-8B、Qwen3-14B、Llama-3.1-8B-Instruct 之间全部六个有向迁移方向的 Affine Adapter 基线。主要迁移指标应在**相同 Source Test、同一冻结 Target Safety Head** 下比较 C 与 F，不能把不同方向的 F1 视为同质测试集上的直接排名。
- **实验三／四**：Token 对数量、内容 token 筛选和测试时字符网格构造都会影响评测结果，是跨模型比较时需要明确记录的协议变量。
- **实验九／十**：迁移到 Llama Safety Head 时，尤其是 Qwen3-14B → Llama-3.1-8B-Instruct，存在严重的有害预测偏置；学习率扫描未消除此现象。Prompt/Response 混合表征消融表明，**使用真实 Llama Prompt/Header 前缀可显著恢复离线检测性能**。这为后续研究前缀表征映射提供方向，但尚不是可部署的修复方案。
---

## 实验十一：Qwen3-14B → Llama-3.1-8B-Instruct 的 F 组 Prompt/Response 表征替换消融

- **目的与实验十的区别**：实验十只针对 **E 组（Llama-3.1-8B-Instruct S-Eval Test）**；本次改为核心迁移场景 **F 组（Qwen3-14B S-Eval Test，1000 条：benign 485 / harmful 515）**，检验真实 Prompt/Header 替换能否同样恢复检测能力。
- **固定条件**：复用实验八 E/F 共用的 `lr=3e-4` Affine Adapter、冻结的 Llama Safety Head、现有双侧 Layer 20 hidden-state 缓存及字符映射；统一使用 Llama token 网格和 `assistant_start`。**不重新训练或提取底座表征**。T0/T1 分别与历史 C/F 逐样本预测完全一致。

| 方案 | Prompt/Header 表征 | Response 表征 | Response F1 | Streaming F1 | Streaming FP |
|---|---|---|---:|---:|---:|
| **T0**（原 C） | 真实 Llama | 真实 Llama | 0.8223 | **0.7926** | 239 |
| **T1**（原 F） | Qwen3-14B → Affine | Qwen3-14B → Affine | 0.6862 | 0.6799 | 485 |
| **T2** | **真实 Llama** | Qwen3-14B → Affine | 0.7397 | **0.7076** | 422 |
| **T3** | Qwen3-14B → Affine | **真实 Llama** | 0.6908 | 0.6799 | 485 |

**观察**：T1 → T2 的 Streaming F1 仅提高 **2.77 pp**，FP 从 485 降至 422；T2 仍比 T0 低 **8.50 pp**。T1 → T3 的 Streaming F1 不变，仍对全部 1000 条响应触发 harmful。与实验十 E 组 T2 基本恢复原生表现不同，**F 组的退化不能只用 Prompt/Header 映射解释**；在真实前缀条件下，Response 映射差异仍显著影响最终预测。混合输入是诊断手段，并非可部署方案；不能直接将 E/F 差异归因于 Response 生成模型，因为测试集组成也不同。

- **代码**：`Kelp/exp11_llama_f_prompt_response_ablation.py`、`Kelp/run_exp11_llama_f_prompt_response_gpu1.sh`；物理 GPU 1。
- **产物**：`checkpoints/llama_target_prompt_response_ablation_v1/qwen14_to_llama_f_src_test_lr3e-4/`（`summary.csv`、`metrics_t0.json`～`metrics_t3.json`、逐样本预测）；日志 `logs/exp11_llama_f_prompt_response_gpu1.log`。

---

## 实验十二：F 组逐 Token 风险触发诊断

- **方法**：在相同 Qwen3-14B S-Eval Test 上，复用实验十一 T0～T3 的输入构造，重新运行**冻结的 Llama Safety Head 前向**并记录逐 token 二分类判定；不重训 Adapter/Safety Head、不提取底座表征。四组 Response/Streaming 逐样本预测与实验十一**完全一致**。下述 token 比例指 Safety Head *预测*为 harmful 的比例，**没有 token 级真实标签**。

| 方案 | Streaming TP / FP / FN / TN | Streaming F1 | Benign 响应的平均 harmful token 占比 | 首次触发相对位置中位数¹ |
|---|---|---:|---:|---:|
| T0（真实/真实） | 495 / 239 / 20 / 246 | 0.7926 | 23.78% | 29.28% |
| T1（映射/映射） | 515 / 485 / 0 / 0 | 0.6799 | **98.61%** | 0% |
| T2（真实/映射） | 513 / 422 / 2 / 63 | 0.7076 | **48.57%** | 17.13% |
| T3（映射/真实） | 515 / 485 / 0 / 0 | 0.6799 | 96.94% | 0% |

¹ 仅统计至少发生一次 harmful 触发的 benign Response，位置为 Safety Head 输出序列中的相对比例，**不是原始文本字符位置**。

**同样本配对（485 条 benign，T0 → T2）**：两者均预测 benign **63** 条；T0 benign → T2 harmful **183** 条；T0 harmful → T2 benign **0** 条；两者均 harmful **239** 条。T1/T3 的几乎全 token harmful 说明退化并非仅由孤立 token 的 `max(pred)` 放大；T2 虽有改善，仍存在大量持续的 harmful 预测。

- **代码**：`Kelp/exp12_f_token_triggers.py`、`Kelp/run_exp12_f_token_triggers_gpu1.sh`；物理 GPU 1。
- **产物**：`checkpoints/llama_target_token_trigger_diagnostics_v1/qwen14_to_llama_f_src_test_lr3e-4/`，主要包括 `summary_by_label.csv`、`first_trigger_histogram.csv`、`paired_transitions.csv`、`sample_token_stats.csv`、`positive_token_runs.jsonl.gz`。

---

## 实验十三：F 组 183 条新增安全响应误报的离线分组分析

- **方法**：仅用实验十二保存的 CSV 与 token 触发区间进行 **CPU 只读统计**。将 485 条 benign Response 按 T0/T2 的 Streaming 判定划分为：`new_fp_183`（T0 正确、T2 误报）、`stable_tn_63`（均正确）和 `persistent_fp_239`（均误报）。无新模型前向。

| T2 统计量 | 新增误报 183 | 始终正确 63 | 持续误报 239 |
|---|---:|---:|---:|
| Response 输出 token 长度中位数 | 158 | 170 | **413** |
| Harmful token 占比均值 / 中位数 | 30.21% / 25.20% | 0% / 0% | **75.44% / 79.40%** |
| 首次触发相对位置中位数 | 31.58% | —（未触发） | 11.42% |
| Harmful 连续区间数中位数 | 4 | 0 | 5 |
| 最长连续 harmful 区间占响应比例中位数 | 11.16% | 0% | **55.84%** |

**新增误报首次触发分布**：0～10%：15 条；10～25%：54 条；25～50%：74 条；50～75%：26 条；75～100%：14 条。即 **143/183（78.1%）** 在前半段首次触发，但并非都在开头。对 T0 原本判 benign 的 246 条按长度分四组，T2 翻转率依次为 **40/61（65.6%）、52/62（83.9%）、48/61（78.7%）、43/62（69.3%）**，未观察到随长度单调升高。以上仅为描述性统计，不能排除长度或文本分布的混杂影响。

- **代码**：`Kelp/exp13_f_false_positive_analysis.py`、`Kelp/run_exp13_f_false_positive_analysis.sh`；无需 GPU。
- **产物**：`checkpoints/llama_target_false_positive_analysis_v1/qwen14_to_llama_f_src_test_lr3e-4/`，主要包括 `analysis_summary.txt`、`group_summary.csv`、`candidate_samples_for_token_inspection.csv`、`length_stratified_flip_rates.csv`。

---

## 实验十四：F 组典型误触发位置与原始 Response 文本对照

- **方法**：CPU 只读检查实验十三筛出的 benign 新增误报样本 `597、624、697、36`。根据 **`target_cache_index = assistant_start + head_output_position`**，结合 Llama tokenizer 的字符偏移和已有 Qwen→Llama 来源索引，将首次触发位置、连续触发区间对应到 Response 文本。**不重新运行底座模型或 Safety Head**。

| 样本 index | 首次触发输出位置 / 长度 | T2 harmful token 占比 | 观察 |
|---|---:|---:|---|
| 597 | 34 / 843（4.03%） | **81.38%** | 首次触发在普通解释文本；最长连续 harmful 区间为输出位置 **489～789（301 token）** |
| 624 | 45 / 473（9.51%） | **84.78%** | 较早触发且持续占比较高 |
| 697 | 50 / 243（20.58%） | **69.55%** | 中前段开始触发，存在长连续 harmful 区间 |
| 36 | 71 / 77（92.21%） | **7.79%** | 临近结尾才触发，连续 6 个位置 harmful |

**样本 597 的定位核查**：首次触发为 Llama Response 正文 token `' understood'`（Head 输出位置 34；目标缓存位置 100），映射 Qwen 来源缓存位置 70，`exact-boundary candidate=True`。这只表示当前 token 的**字符结束边界存在精确匹配候选**，不代表两种 hidden states 对齐充分或分类错误由该词触发。样本 597 的长连续 harmful 区间与描述性医疗费用文本对应，说明**至少部分误报是持续预测偏置，而非单个 token 的偶发触发**；四条案例不能代表全部 183 条的内容主题或因果机制。

- **代码**：`Kelp/exp14_f_token_text_inspection.py`、`Kelp/run_exp14_f_token_text_inspection_cpu.sh`；无需 GPU。
- **产物**：`checkpoints/llama_target_trigger_text_inspection_v1/qwen14_to_llama_f_src_test_lr3e-4/`，包括 `overview.csv`、`sample_0597_review.md`、`sample_0597_positions.csv` 等。

---

## 更新后的阶段小结（实验一～十四）

1. **基线与协议**：六个有向迁移方向已覆盖（实验一、二、五～八）；字符对齐与 token 选择协议本身是重要变量（实验三、四）。在核心迁移 F 场景中，应优先在同一 Source Test 与同一 Target Safety Head 下比较真实目标表征 C 和映射表征 F。
2. **Qwen3-14B → Llama 的预测偏置**：学习率扫描（实验九）未解决 F 的全量 Streaming harmful 触发。实验十在 **E（Llama Test）** 中发现真实 Prompt/Header 可将 Streaming F1 从 `0.6235` 恢复至 `0.9213`；但实验十一在 **F（Qwen3-14B Test）** 中只能从 `0.6799` 提高到 `0.7076`，仍低于 C 的 `0.7926`。**因此实验十的“前缀恢复即可恢复性能”不应推广为 F 组的结论**。
3. **F 组误报的具体表现**：实验十二～十四表明，T2 在 485 条 benign 中新增 **183 条**误报；新增误报的 harmful token 占比均值 **30.21%**、首次触发位置中位数 **31.58%**，部分样本出现很长的持续 harmful 区间。问题不只在 Streaming 聚合或单一词语，Response 映射与前缀表征之间可能存在非加性交互。
4. **尚未证实的机制与后续方向**：目前未直接测量映射 hidden states 的误差、余弦相似度或 Safety Head 对不同误差方向的敏感性，**不能将偏置确认为 Affine MSE 失真、Tokenizer 错位或具体敏感词导致**。可优先比较真实与映射 Response 表征的逐 token 误差及其与新增误报的关联；继续使用离线评测口径，混合真实 Llama hidden states 的消融不等于可部署方案。
