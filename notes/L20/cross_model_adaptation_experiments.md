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

| 模型迁移方向 | 训练 Response（对应实验） | Train 候选对（8100 条） | 每 Epoch 实际抽取 | 3 Epoch 累计抽取 | 3 Epoch 去重 | 去重 / 累计 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-14B → 8B | 14B Train（B） | 4,686,881 | 1,023,752 | 3,071,256 | 2,002,598 | 65.20% |
| Qwen3-14B → 8B | 8B Train（E/F） | 4,792,878 | 1,025,330 | 3,075,990 | 1,986,770 | 64.59% |
| Llama → 8B | Llama Train（B） | 2,262,223 | **686,963** | 2,060,889 | **1,250,653** | 60.69% |
| Llama → 8B | 8B Train（E/F） | 4,661,962 | 1,015,354 | 3,046,062 | 1,980,889 | 65.03% |

每 Epoch 的理论抽取上限为 `8100 × 128 = 1,036,800` 对。实际抽取量占上限的比例依次为 **98.74%、98.89%、66.26%、97.93%**；对应平均每条训练样本抽取 **126.39、126.58、84.81、125.35** 对。达不到上限主要是部分样本可选位置不足，尤其抽样还受 prompt/response 配额限制。

**进一步拆分（每 Epoch）：**

| 训练设置 | Prompt／前缀侧抽取 | Response 侧抽取 | Validation 每次评测抽取 | 归一化统计（320 条）抽取 |
|---|---:|---:|---:|---:|
| Qwen3-14B / B | 234,631 | 789,121 | 113,781 | 40,282 |
| Qwen3-14B / E/F | 234,631 | 790,699 | 114,082 | 40,362 |
| Llama / B | 184,755 | 502,208 | 76,368 | 27,369 |
| Llama / E/F | 184,755 | 830,599 | 113,081 | 39,974 |

> 表中的“Prompt／前缀侧”口径不同：INDEX 包含模板／特殊 token；EXACT 只包含通过边界匹配的内容 token。因此两类方法的总量和构成不能直接视为完全同质的训练信息。

### 3. 字符边界匹配覆盖率

实验二的 EXACT 对齐以 **Qwen3-8B 内容 token 位置**为分母。定义：

```text
Exact-boundary coverage = 精确匹配的 Qwen 内容 token 位置数 / Qwen 内容 token 位置总数
```

| 配对数据（9000 条 Train Response 来源） | Qwen 内容 token 位置 | 精确边界配对位置 | 覆盖率 |
|---|---:|---:|---:|
| Llama Train | 2,531,326 | 2,509,603 | **99.14%** |
| Qwen3-8B Train | 5,220,564 | 5,183,189 | **99.28%** |

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
