# PlugGuard Reproduction

Official repository:
Alibaba-AAIG/Kelp

Base commit:
de7abe9bb1af1186e12b3fe5bf7fab65c2fdc550

## Qwen3-8B + S-Eval 复现结果

### 实验配置

- Base model: Qwen3-8B
- Dataset: S-Eval
- Train / Test: 9000 / 1000
- Hidden layer: 20
- Epoch: 1
- Learning rate: 5e-5
- Weight decay: 0
- Batch size: 1
- Gradient accumulation: 32
- Seed: 42

### 复现结果

| Metric | Reproduction | Paper |
|---|---:|---:|
| Response-level F1 (label=1) | 0.9087 | 0.9282 |
| Streaming F1 (label=1) | 0.9069 | 0.9246 |

补充：

- Response accuracy: 0.9030
- Streaming accuracy: 0.9000
- Test set: benign 467，harmful 533
- `label=1` 对应 harmful。
- 论文未明确说明 Table 1 的 F1 averaging 方式，目前暂按 harmful 类（label=1）的 F1 与论文结果比较。

原始输出：
-------------Response level-------- 
               precision    recall  f1-score   support

           0     0.8936    0.8994    0.8965       467
           1     0.9113    0.9062    0.9087       533

    accuracy                         0.9030      1000
   macro avg     0.9025    0.9028    0.9026      1000
weighted avg     0.9031    0.9030    0.9030      1000


-----------Streaming level-----------
               precision    recall  f1-score   support

           0     0.8998    0.8844    0.8920       467
           1     0.9002    0.9137    0.9069       533

    accuracy                         0.9000      1000
   macro avg     0.9000    0.8990    0.8994      1000
weighted avg     0.9000    0.9000    0.8999      1000

Checkpoint:

```text
/data1/plugguard_repro/checkpoints/qwen3_8b_seval/model_epoch_0.pt
```

### 已知问题

- 训练日志显示总计划为 282 个 optimizer update，但实际只执行到 281/282。原因是 9000 不能被 gradient accumulation 32 整除，最后 8 个样本完成了 backward，但未触发 `optimizer.step()`。
- Response 尾部实现审计：Qwen3 chat template 在 response 结束后还会附加额外 assistant header，因此官方 `eval.py` 的 `pred[-2]` 并非严格意义上的最后正文 token。对现有 checkpoint 进行 readout sweep 后，`pred[-5]` 到 `pred[-1]` 的 Response harmful F1 均为 `0.9087`；Streaming 的不同尾部截断范围 harmful F1 也均为 `0.9069`。因此尾部 token 选择在本次实验中未影响最终指标，后续继续保留作者默认实现。

## Qwen3-8B + WildGuard 复现结果

### 实验配置

同 **Qwen3-8B + S-Eval 复现配置**，仅将数据集替换为 WildGuard。

- Base model: Qwen3-8B
- Dataset: WildGuard
- Train / Test: 37934 / 1725

### 复现结果

| Metric | Reproduction | Paper |
|---|---:|---:|
| Response-level F1 (label=1) | 0.8495 | 0.8462 |
| Streaming F1 (label=1) | 0.8199 | 0.8333 |

补充：

- Response accuracy: 0.9513
- Streaming accuracy: 0.9351
- Test set: benign 1441，harmful 284
- `label=1` 对应 harmful。
- Response-level F1 比论文高 `0.0033`，与论文结果基本一致。
- Streaming F1 比论文低 `0.0134`。
- 论文未明确说明 Table 1 的 F1 averaging 方式，目前暂按 harmful 类（label=1）的 F1 与论文结果比较。WildGuard 复现中的 harmful F1 `0.8495` 与论文 `0.8462` 非常接近，而 macro F1 为 `0.9102`，进一步支持当前比较方式。

原始输出：

```text
-------------Response level--------

               precision    recall  f1-score   support

           0     0.9676    0.9743    0.9710      1441
           1     0.8650    0.8345    0.8495       284

    accuracy                         0.9513      1725
   macro avg     0.9163    0.9044    0.9102      1725
weighted avg     0.9507    0.9513    0.9510      1725


-----------Streaming level-----------

               precision    recall  f1-score   support

           0     0.9791    0.9424    0.9604      1441
           1     0.7544    0.8979    0.8199       284

    accuracy                         0.9351      1725
   macro avg     0.8668    0.9201    0.8902      1725
weighted avg     0.9421    0.9351    0.9373      1725

Checkpoint:
/data1/plugguard_repro/checkpoints/qwen3_8b_wildguard/model_epoch_0.pt

已知问题
与 S-Eval 相同，当前训练代码仅在累计满 32 个样本时触发 optimizer.step()。WildGuard 训练集有 37934 条样本，37934 = 1185 × 32 + 14，因此计划 optimizer update 数为 1186，但最后 14 个样本完成 backward 后不会触发最后一次 optimizer.step()。

## Qwen3-14B + S-Eval 复现结果

### 实验配置

同 **Qwen3-8B + S-Eval 复现配置**，仅将 Base model 替换为 Qwen3-14B。

- Base model: Qwen3-14B
- Dataset: S-Eval
- Train / Test: 9000 / 1000
- Hidden layer: 20

### 复现结果

| Metric | Reproduction | Paper |
|---|---:|---:|
| Response-level F1 (label=1) | 0.8668 | — |
| Streaming F1 (label=1) | 0.8748 | 0.9041 |

补充：

- Response accuracy: 0.8660
- Streaming accuracy: 0.8700
- Test set: benign 485，harmful 515
- `label=1` 对应 harmful。
- 论文 Table 5 对 Qwen3-14B + S-Eval 报告 Streaming F1 `0.9041`，未单独报告 Response-level F1。
- Streaming F1 比论文低 `0.0293`。
- Qwen3-14B 的测试集 harmful 数量为 515，与论文 StreamGuardBench 数据统计一致。
- 当前继续按 harmful 类（label=1）的 F1 与论文 Streaming F1 进行比较。

原始输出：

```text
-------------Response level--------

               precision    recall  f1-score   support

           0     0.8448    0.8866    0.8652       485
           1     0.8880    0.8466    0.8668       515

    accuracy                         0.8660      1000
   macro avg     0.8664    0.8666    0.8660      1000
weighted avg     0.8670    0.8660    0.8660      1000


-----------Streaming level-----------

               precision    recall  f1-score   support

           0     0.8721    0.8577    0.8649       485
           1     0.8681    0.8816    0.8748       515

    accuracy                         0.8700      1000
   macro avg     0.8701    0.8696    0.8698      1000
weighted avg     0.8700    0.8700    0.8700      1000

Checkpoint:
/data1/plugguard_repro/checkpoints/qwen3_14b_seval/model_epoch_0.pt
已知问题
与 Qwen3-8B + S-Eval 相同，训练集有 9000 条样本，无法被 gradient accumulation 32 整除，因此最后 8 个样本完成 backward 后不会触发最后一次 optimizer.step()。

## Qwen3-14B + WildGuard 复现结果

### 实验配置

同 **Qwen3-14B + S-Eval 复现配置**，仅将数据集替换为 WildGuard。

- Base model: Qwen3-14B
- Dataset: WildGuard
- Train / Test: 37934 / 1725
- Hidden layer: 20

### 复现结果

| Metric | Reproduction | Paper |
|---|---:|---:|
| Response-level F1 (label=1) | 0.7945 | — |
| Streaming F1 (label=1) | 0.7770 | 0.7845 |

补充：

- Response accuracy: 0.9391
- Streaming accuracy: 0.9281
- Test set: benign 1466，harmful 259
- `label=1` 对应 harmful。
- 论文 Table 5 对 Qwen3-14B + WildGuard 报告 Streaming F1 `0.7845`，未单独报告 Response-level F1。
- Streaming F1 比论文低 `0.0075`。
- Qwen3-14B WildGuard test 中 harmful 数量为 259，与论文数据统计一致。
- Streaming 相比 Response-level 将 harmful recall 从 `0.7838` 提高到 `0.8340`，同时 precision 从 `0.8056` 降至 `0.7273`。

原始输出：

```text
-------------Response level--------

               precision    recall  f1-score   support

           0     0.9620    0.9666    0.9643      1466
           1     0.8056    0.7838    0.7945       259

    accuracy                         0.9391      1725
   macro avg     0.8838    0.8752    0.8794      1725
weighted avg     0.9385    0.9391    0.9388      1725


-----------Streaming level-----------

               precision    recall  f1-score   support

           0     0.9699    0.9447    0.9572      1466
           1     0.7273    0.8340    0.7770       259

    accuracy                         0.9281      1725
   macro avg     0.8486    0.8894    0.8671      1725
weighted avg     0.9335    0.9281    0.9301      1725

Checkpoint:
/data1/plugguard_repro/checkpoints/qwen3_14b_wildguard/model_epoch_0.pt
已知问题:
与 Qwen3-8B + WildGuard 相同，当前训练代码仅在累计满 32 个样本时触发 optimizer.step()。
WildGuard 训练集有 37934 条样本，37934 = 1185 × 32 + 14，因此最后 14 个样本完成 backward 后不会触发最后一次 optimizer.step()。
继续保留作者默认的 Response-level pred[-2] 和 Streaming-level max(pred) 评测实现。

## Llama-3.1-8B + S-Eval 复现结果

### 实验配置

同前述 S-Eval baseline 配置，仅将 Base model 替换为 Llama-3.1-8B-Instruct，并针对 Llama chat template 适配 assistant boundary。

- Base model: Llama-3.1-8B-Instruct
- Dataset: S-Eval
- Train / Test: 9000 / 1000
- Hidden layer: 20
- Hidden size: 4096
- Epoch: 1
- Learning rate: 5e-5
- Weight decay: 0
- Batch size: 1
- Gradient accumulation: 32
- Seed: 42

### 复现结果

| Metric | Reproduction | Paper |
|---|---:|---:|
| Response-level F1 (label=1) | 0.9095 | — |
| Streaming F1 (label=1) | 0.9214 | 0.9590 |

补充：

- Response accuracy: 0.9180
- Streaming accuracy: 0.9270
- Test set: benign 547，harmful 453
- `label=1` 对应 harmful。
- 论文 Table 5 对 Llama-3.1-8B + S-Eval 报告 Streaming F1 `0.9590`，未单独报告 Response-level F1。
- Streaming F1 比论文低 `0.0376`。
- Llama-3.1-8B 的 S-Eval test harmful 数量为 453，与论文 StreamGuardBench 数据统计一致。
- Streaming 相比 Response-level 将 harmful recall 从 `0.9095` 提高到 `0.9448`，同时 precision 从 `0.9095` 降至 `0.8992`。
- 在正式实验前已验证 Llama chat template、assistant boundary、Layer 20 hidden-state extraction、hidden size 4096 以及 embedding / label 长度均正常。

原始输出：

```text
-------------Response level--------

               precision    recall  f1-score   support

           0     0.9250    0.9250    0.9250       547
           1     0.9095    0.9095    0.9095       453

    accuracy                         0.9180      1000
   macro avg     0.9173    0.9173    0.9173      1000
weighted avg     0.9180    0.9180    0.9180      1000


-----------Streaming level-----------

               precision    recall  f1-score   support

           0     0.9523    0.9122    0.9318       547
           1     0.8992    0.9448    0.9214       453

    accuracy                         0.9270      1000
   macro avg     0.9257    0.9285    0.9266      1000
weighted avg     0.9282    0.9270    0.9271      1000

Checkpoint:
/data1/plugguard_repro/checkpoints/llama_3_1_8b_seval/model_epoch_0.pt
已知问题:
与 Qwen3 S-Eval 实验相同，训练集有 9000 条样本，无法被 gradient accumulation 32 整除，因此最后 8 个样本完成 backward 后不会触发最后一次 optimizer.step()。
Llama-3.1-8B 使用其自身 chat template。由于 tokenizer.encode() 默认会附加 BOS token，当前仅对 meta-llama/Llama-3.1-8B-Instruct 的 assistant marker 编码使用 add_special_tokens=False，Qwen 路径保持作者原始行为。
与 Qwen baseline 一致，继续保留作者默认的 add_generation_prompt=True、assistant_end=-1、Response-level pred[-2] 和 Streaming-level max(pred) 评测逻辑。

## Llama-3.1-8B + WildGuard 复现结果

### 实验配置

同前述 WildGuard baseline 配置，仅将 Base model 替换为 Llama-3.1-8B-Instruct，并对 Llama chat template 的 assistant boundary 做最小适配。

- Base model: Llama-3.1-8B-Instruct
- Dataset: WildGuard
- Train / Test: 37934 / 1725
- Hidden layer: 20
- Hidden size: 4096
- Epoch: 1
- Learning rate: 5e-5
- Weight decay: 0
- Batch size: 1
- Gradient accumulation: 32
- Seed: 42

论文 Table 6 中，Llama-3.1-8B 对应 WildGuard 的 harmful response 数为：

- Train harmful: 6111
- Test harmful: 206

本次测试集统计为 benign 1519 / harmful 206，与论文一致。

### 复现结果

| Metric | Reproduction | Paper |
|---|---:|---:|
| Response-level F1 (label=1) | 0.7404 | — |
| Streaming F1 (label=1) | 0.6969 | 0.8115 |

补充：

- Response accuracy: 0.9374
- Streaming accuracy: 0.9148
- `label=1` 对应 harmful。
- 论文 Table 5 对 Llama-3.1-8B + WildGuard 报告 Streaming F1 `0.8115`，未单独报告 Response-level F1。
- 本次 Streaming F1 比论文低 `0.1146`，是当前各组复现实验中差距最大的一组。

原始输出：

```text
-------------Response level--------

               precision    recall  f1-score   support

           0     0.9657    0.9631    0.9644      1519
           1     0.7333    0.7476    0.7404       206

    accuracy                         0.9374      1725
   macro avg     0.8495    0.8554    0.8524      1725
weighted avg     0.9379    0.9374    0.9377      1725


-----------Streaming level-----------

               precision    recall  f1-score   support

           0     0.9744    0.9276    0.9504      1519
           1     0.6057    0.8204    0.6969       206

    accuracy                         0.9148      1725
   macro avg     0.7901    0.8740    0.8237      1725
weighted avg     0.9304    0.9148    0.9201      1725

Checkpoint:
/data1/plugguard_repro/checkpoints/llama_3_1_8b_wildguard/model_epoch_0.pt
实验过程中的兼容问题

首次正式评测时，WildGuard test set 中存在一条极短 response，其 assistant sequence 长度仅为 6，小于默认 num_supervised_token=10，导致原始标签构造逻辑发生 tensor shape mismatch：

RuntimeError: The expanded size of the tensor (6) must match the existing size (10)

检查后确认：

Train cache 已完整构建 37934 条；
PlugGuard 训练已正常完成；
checkpoint 已成功保存；
错误仅发生在训练完成后的 test cache 构建阶段；
test set 中仅有 1 条 seq_len < 10 的样本，index 为 951，label 为 benign。

因此仅对 Llama 且 seq_len < num_supervised_token 的情况使用：

effective_n = seq_len

其余 Llama 样本和 Qwen 路径均保持原始行为不变。

修复后重新构建完整 test cache，逐条检查结果：

raw samples      : 1725
cached samples   : 1725
label mismatches : 0
short samples    : 1

说明 test cache 完整，且 cache label 与原始数据逐条一致。

Streaming False Positive 诊断

为定位 Streaming F1 明显偏低的原因，对 1519 条 benign test samples 的 token-level prediction 进行了诊断。

结果：

Response FP  : 56
Streaming FP : 110

Streaming 相比 Response-level 额外产生了较多 false positive。

对 110 条 Streaming false positive 的首次 harmful trigger 位置统计：

Last 3 header tokens : 0
<|eot_id|>           : 0
Last body token      : 0
Earlier body         : 110

因此，本次 Streaming false positive 并非由 response 尾部的 special token 或额外 assistant header 触发，而全部发生在 response 正文较早位置。

按 assistant trajectory 长度统计 benign 样本：

<=50 tokens:
benign = 489
FP = 0
FP rate = 0.0000

51-100 tokens:
benign = 42
FP = 0
FP rate = 0.0000

101-200 tokens:
benign = 36
FP = 2
FP rate = 0.0556

>200 tokens:
benign = 952
FP = 108
FP rate = 0.1134

其中：

110 个 Streaming false positive 中有 108 个来自长度超过 200 token 的 benign response；
全部 benign response 平均 assistant trajectory 长度约为 370.9；
Streaming false positive 样本平均长度约为 649.5。

当前诊断表明：

Llama-3.1-8B + WildGuard 的 Streaming F1 明显低于论文，主要表现为长 benign response 正文中的中途 harmful 误触发；该现象不是由尾部 special token、test cache 不完整或 label 错位造成。

当前结论
Llama-3.1-8B 模型加载、Layer 20 hidden-state extraction、hidden size 4096、assistant boundary 均已验证正常。
WildGuard train/test 数据规模及 harmful 数与论文一致。
Train cache 完整，训练正常结束，checkpoint 正常保存。
Test cache 完整，1725 条 cache label 与原始数据逐条一致。
短 response 兼容修复只影响 1 条 test sample，不会解释整体 F1 大幅下降。
当前较大的复现差异集中在 Llama-3.1-8B + WildGuard 的长序列 Streaming false positive。
暂不进一步修改 baseline 实现，保留该结果及诊断作为复现差异记录。

## Figure 3 Cross-model Transfer：Qwen3-14B → Qwen3-8B

### 实验设置

复现 Figure 3 中：

- Source Model: Qwen3-14B
- Target Model: Qwen3-8B
- Dataset: S-Eval
- Hidden layer: 20
- Target backbone: Qwen3-8B
- PlugGuard head: 重新训练

论文 Figure 3 的定义是：使用 source model 生成的 query-response pairs 训练 PlugGuard，并在 target model 上进行风险检测。:contentReference[oaicite:0]{index=0}

因此本实验采用：

```text
Train responses : Qwen3-14B / S-Eval trainset
Train features  : Qwen3-8B hidden states
Test responses  : Qwen3-8B / S-Eval testset
Test features   : Qwen3-8B hidden states
为支持该实验，对 dataset.py 做了最小修改：assistant marker 不再依赖全局 ACTIVE_MODEL，而是根据实际传入的 model_name 选择对应配置。其余 PlugGuard 训练、损失函数、评测逻辑均保持不变。

复现结果
Metric	                   Reproduction	   Paper
Response-level harmful F1	0.9130	         —
Streaming harmful F1	    0.9139	       0.9161
原始输出：
-------------Response level-------- 
               precision    recall  f1-score   support

           0     0.9030    0.8972    0.9001       467
           1     0.9104    0.9156    0.9130       533

    accuracy                         0.9070      1000
   macro avg     0.9067    0.9064    0.9066      1000
weighted avg     0.9070    0.9070    0.9070      1000


-----------Streaming level-----------
               precision    recall  f1-score   support

           0     0.9229    0.8715    0.8965       467
           1     0.8927    0.9362    0.9139       533

    accuracy                         0.9060      1000
   macro avg     0.9078    0.9039    0.9052      1000
weighted avg     0.9068    0.9060    0.9058      1000

## Figure 3 Cross-model Transfer：Llama-3.1-8B → Qwen3-8B

### 实验设置

复现 Figure 3 中：

- Source Model: Llama-3.1-8B-Instruct
- Target Model: Qwen3-8B
- Dataset: S-Eval
- Hidden layer: 20
- Target backbone: Qwen3-8B
- PlugGuard head: 重新训练

实验数据流：

```text
Train responses : Llama-3.1-8B / S-Eval trainset
Train features  : Qwen3-8B hidden states
Test responses  : Qwen3-8B / S-Eval testset
Test features   : Qwen3-8B hidden states

复现结果
Metric	                    Reproduction	Paper
Response-level harmful F1	0.8952	          —
Streaming harmful F1	    0.8993	        0.8783

原始输出：
-------------Response level-------- 
               precision    recall  f1-score   support

           0     0.9037    0.8437    0.8726       467
           1     0.8706    0.9212    0.8952       533

    accuracy                         0.8850      1000
   macro avg     0.8871    0.8824    0.8839      1000
weighted avg     0.8860    0.8850    0.8847      1000


-----------Streaming level-----------
               precision    recall  f1-score   support

           0     0.9401    0.8073    0.8687       467
           1     0.8497    0.9550    0.8993       533

    accuracy                         0.8860      1000
   macro avg     0.8949    0.8811    0.8840      1000
weighted avg     0.8920    0.8860    0.8850      1000

## Figure 3 Cross-model Transfer：Qwen3-8B → Qwen3-14B

### 实验设置

复现 Figure 3 中：

- Source Model: Qwen3-8B
- Target Model: Qwen3-14B
- Dataset: S-Eval
- Hidden layer: 20
- Target backbone: Qwen3-14B
- PlugGuard head: 重新训练

实验数据流：

```text
Train responses : Qwen3-8B / S-Eval trainset
Train features  : Qwen3-14B hidden states
Test responses  : Qwen3-14B / S-Eval testset
Test features   : Qwen3-14B hidden states

复现结果
Metric	                    Reproduction	Paper
Response-level harmful F1	0.8859	         —
Streaming harmful F1	    0.8795	       0.8827
原始输出：
-------------Response level-------- 
               precision    recall  f1-score   support

           0     0.8815    0.8742    0.8778       485
           1     0.8825    0.8893    0.8859       515

    accuracy                         0.8820      1000
   macro avg     0.8820    0.8818    0.8819      1000
weighted avg     0.8820    0.8820    0.8820      1000


-----------Streaming level-----------
               precision    recall  f1-score   support

           0     0.8827    0.8536    0.8679       485
           1     0.8663    0.8932    0.8795       515

    accuracy                         0.8740      1000
   macro avg     0.8745    0.8734    0.8737      1000
weighted avg     0.8743    0.8740    0.8739      1000
