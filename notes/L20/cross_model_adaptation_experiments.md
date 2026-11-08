# 流式安全围栏跨模型适配实验记录

## 实验一：Qwen3-14B → Qwen3-8B

### 1. 实验配置与结果总表

- **环境**：`guard_g22`，NVIDIA L20（使用 GPU 0）；项目根目录：`~/plugguard_repro/`（下文路径均相对于此目录）。
- **数据**：StreamGuardBench v1 / S-Eval；8B、14B 各有 Train 9000 / Test 1000。8B Test：benign 467 / harmful 533；14B Test：benign 485 / harmful 515。
- **表征**：均取 Layer 20；Qwen3-8B 为 4096 维，Qwen3-14B 为 5120 维。B、C、E、F 均**冻结同一个 Qwen3-8B Safety Head**；A、D 为此前复现的原生围栏。
- **指标**：下表 F1 均指 harmful（label=1）类别；每组测试 1000 条。`8B/14B Test` 指 **Response 的生成来源**，不等于提取 hidden states 的模型。

| 实验 | 设计目的 | Adapter 训练 Response | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---:|---:|
| **A** | 新模型原生基线 | 无（14B 围栏原生训练） | 14B Test | 14B 5120D → 14B Head | 0.8955 | **0.8971** |
| **B** | 新数据训练的 Affine 基线 | 14B Train | 14B Test | 14B 5120D → Affine → 8B Head | 0.8182 | **0.8054** |
| **C** | 真实 8B 表征对照，分离映射损失 | 无 | 14B Test | **8B 对同一 14B 文本提取的 4096D** → 8B Head | 0.9080 | **0.9007** |
| **D** | 旧模型原生基线 | 无（8B 围栏原生训练） | 8B Test | 8B 4096D → 8B Head | 0.9245 | **0.9228** |
| **E** | 旧数据训练，在旧模型生成文本上测试 | 8B Train | 8B Test | 14B 5120D → Affine → 8B Head | 0.8247 | **0.8310** |
| **F** | 旧数据训练，在新模型生成文本上测试（主要适配场景） | 8B Train | 14B Test | 14B 5120D → **同 E 的 Affine** → 8B Head | 0.8121 | **0.8119** |

**说明**：E、F 共用同一个 Adapter；B 使用另一个在 14B Train 上训练的 Adapter。C 需要在测试时运行 8B 提取真实表征，仅是诊断基线，不是最终部署方案。

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

**初步观察**：同一 14B Test 上，C 的 Streaming F1 为 0.9007，B/F 为 0.8054/0.8119；当前主要差距在将真实 8B 表征替换成 Affine 映射表征之后。尚未进行 MLP、MATCHA 或风险蒸馏实验。

---

## 实验二：Llama-3.1-8B → Qwen3-8B

### 1. 实验配置与结果总表

- **环境与数据**：同上，使用 GPU 0；StreamGuardBench v1 / S-Eval。Llama、Qwen3-8B 各有 Train 9000 / Test 1000；Llama Test：benign 547 / harmful 453；8B Test：benign 467 / harmful 533。
- **表征与围栏**：两模型均提取 Layer 20、hidden size 均为 4096。B、C、E、F 始终使用**同一个冻结的 Qwen3-8B Safety Head**；A、D 为此前完成的原生围栏复现实验。
- **指标**：F1 均为 harmful（label=1）类别。下表的 Llama/8B Train、Test 指 Response 的生成来源；B、E、F 测试时均由 **Llama 提取表征**，经过 Adapter 进入 8B 围栏。

| 实验 | 设计目的 | Adapter 训练 Response | 测试 Response | 测试表征 → Safety Head | Response F1 | Streaming F1 |
|---|---|---|---|---|---:|---:|
| **A** | 新模型原生基线 | 无（Llama 围栏原生训练） | Llama Test | Llama 4096D → Llama Head | 0.9095 | **0.9191** |
| **B** | 新数据训练的 Affine 基线 | Llama Train | Llama Test | Llama 4096D → Affine → 8B Head | 0.8340 | **0.8973** |
| **C** | 真实 8B 表征对照，检验映射造成的损失 | 无 | Llama Test | **8B 对同一 Llama 文本提取的 4096D** → 8B Head | 0.9271 | **0.9253** |
| **D** | 旧模型原生基线 | 无（8B 围栏原生训练） | 8B Test | 8B 4096D → 8B Head | 0.9245 | **0.9228** |
| **E** | 旧数据训练，在旧模型生成文本上测试 | 8B Train | 8B Test | Llama 4096D → Affine → 8B Head | 0.9185 | **0.9121** |
| **F** | 旧数据训练，在新模型生成文本上测试（主要适配场景） | 8B Train | Llama Test | Llama 4096D → **同 E 的 Affine** → 8B Head | 0.9177 | **0.9199** |

**说明**：B 使用 Llama Train 训练的 Adapter；E、F 共用 8B Train 训练的另一个 Adapter。C 需要运行 8B 以取得真实表征，仅作诊断基线。A 使用 Llama 原生 token 序列；B、C、F 在评测时统一以 **Qwen3-8B token 序列**为坐标，因此 C 与 B/F 更适合直接比较。

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

代码为每条样本缓存 `src_for_qwen`（测试时的索引映射）、`exact_qwen` / `exact_llama`（训练时的精确配对），并在日志中输出各数据划分的 `exact-boundary coverage`。精确匹配率的具体数值需从运行日志补录，不能假定为 100%。

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

**评测规则**：Response-level 使用 `pred[-2]`；Streaming-level 使用 `max(pred)`。同一 Llama Test 上，F 的 Streaming F1 为 **0.9199**，较 C（0.9253）低 **0.0054**；B 为 **0.8973**。该差异仍需结合 token 对齐覆盖率及多随机种子分析。

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
