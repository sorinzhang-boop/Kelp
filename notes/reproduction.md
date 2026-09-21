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