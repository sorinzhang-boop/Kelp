# NVIDIA L20 复现实验记录

## Qwen3-8B + S-Eval

### 1. 实验配置

| 参数 | 配置 |
|---|---|
| GPU | NVIDIA L20 |
| Base Model | Qwen3-8B |
| Dataset | StreamGuardBench v1 / S-Eval |
| Train / Test | 9000 / 1000 |
| Hidden Layer | 20 |
| Hidden Size | 4096 |
| Epoch | 1 |
| Learning Rate | 5e-5 |
| Weight Decay | 0 |
| Batch Size | 1 |
| Gradient Accumulation | 32 |
| Warmup Ratio | 0.05 |
| Max Length | 4096 |
| Supervised Tokens | 10 |
| Seed | 42 |

冻结底座 LLM，仅训练 PlugGuard Safety Head。

### 2. 复现结果

以下 F1 均为 harmful 类（label=1）。

| Metric | 本次复现 | 论文 | 差值 |
|---|---:|---:|---:|
| Response-level F1 | **0.9245** | 0.9282 | -0.0037 |
| Streaming F1 | **0.9228** | 0.9246 | -0.0018 |

测试集标签分布：benign 467，harmful 533。

### 3. 实验执行情况

- 训练缓存：9000/9000，构建成功。
- 测试缓存：1000/1000，构建成功。
- 实际 optimizer update：281/282。
- 训练、评测均正常完成。
- Checkpoint：`checkpoints/qwen3_8b_seval/model_epoch_0.pt`
- Log：`logs/qwen3_8b_seval_train.log`

路径相对于 `plugguard_repro/` 工作目录。

### 4. 已知问题

1. **梯度累积：** 9000 = 281 × 32 + 8，最后 8 条样本没有触发 `optimizer.step()`，保留原实现。
2. **评测规则：** Response-level 使用 `pred[-2]`，Streaming-level 使用 `max(pred)`，未修改原代码逻辑。
3. **指标口径：** 按 harmful 类 F1 与论文比较；论文未明确说明对应表格的 F1 averaging 方式。

### 5. 原始评测输出

```text
-------------Response level--------
              precision    recall  f1-score   support

           0     0.9196    0.9058    0.9126       467
           1     0.9185    0.9306    0.9245       533

    accuracy                         0.9190      1000
   macro avg     0.9190    0.9182    0.9186      1000
weighted avg     0.9190    0.9190    0.9190      1000


-----------Streaming level-----------
              precision    recall  f1-score   support

           0     0.9547    0.8565    0.9029       467
           1     0.8847    0.9644    0.9228       533

    accuracy                         0.9140      1000
   macro avg     0.9197    0.9104    0.9129      1000
weighted avg     0.9174    0.9140    0.9135      1000
```

本次 Response-level F1 和 Streaming F1 均接近论文报告值。

---

## Qwen3-8B + WildGuard

### 实验配置

与前述 Qwen3-8B + S-Eval 实验配置一致，仅替换数据集。

- Dataset：StreamGuardBench v1 / WildGuard
- Train / Test：37934 / 1725
- Hidden Layer：20
- Test：benign 1441，harmful 284

### 复现结果

以下 F1 均为 harmful 类（label=1）。

| Metric | 本次复现 | 论文 | 差值 |
|---|---:|---:|---:|
| Response-level F1 | **0.8520** | 0.8462 | +0.0058 |
| Streaming F1 | **0.8176** | 0.8333 | -0.0157 |

- Response accuracy：0.9525
- Streaming accuracy：0.9351

### 原始评测输出

```text
-------------Response level--------
              precision    recall  f1-score   support

           0     0.9670    0.9764    0.9717      1441
           1     0.8741    0.8310    0.8520       284

    accuracy                         0.9525      1725
   macro avg     0.9205    0.9037    0.9118      1725
weighted avg     0.9517    0.9525    0.9520      1725


-----------Streaming level-----------
              precision    recall  f1-score   support

           0     0.9763    0.9452    0.9605      1441
           1     0.7606    0.8838    0.8176       284

    accuracy                         0.9351      1725
   macro avg     0.8685    0.9145    0.8890      1725
weighted avg     0.9408    0.9351    0.9370      1725
```

### 实验记录

- 训练及测试缓存构建完成。
- Checkpoint：`checkpoints/qwen3_8b_wildguard/model_epoch_0.pt`
- Log：`logs/qwen3_8b_wildguard_train.log`
- 完整流程日志：`logs/qwen3_8b_wildguard_pipeline.log`
- 梯度累积尾部：37934 = 1185 × 32 + 14，最后 14 条样本未触发 `optimizer.step()`，保持原实现。

本次 Response-level F1 略高于论文结果，Streaming F1 低于论文结果 0.0157。

---

## Qwen3-14B + S-Eval

### 实验配置

与前述 Qwen3-8B + S-Eval 实验配置一致，仅替换底座模型及对应数据。

- Base Model：Qwen3-14B
- Dataset：StreamGuardBench v1 / S-Eval
- Train / Test：9000 / 1000
- Hidden Layer：20
- Hidden Size：5120
- Test：benign 485，harmful 515

### 复现结果

以下 F1 均为 harmful 类（label=1）。

| Metric | 本次复现 | 论文 | 差值 |
|---|---:|---:|---:|
| Response-level F1 | **0.8955** | — | — |
| Streaming F1 | **0.8971** | 0.9041 | -0.0070 |

- Response accuracy：0.8910
- Streaming accuracy：0.8910
- 论文未单独报告该组合的 Response-level F1。

### 原始评测输出

```text
-------------Response level--------
              precision    recall  f1-score   support

           0     0.8983    0.8742    0.8861       485
           1     0.8845    0.9068    0.8955       515

    accuracy                         0.8910      1000
   macro avg     0.8914    0.8905    0.8908      1000
weighted avg     0.8912    0.8910    0.8909      1000


-----------Streaming level-----------
              precision    recall  f1-score   support

           0     0.9123    0.8577    0.8842       485
           1     0.8732    0.9223    0.8971       515

    accuracy                         0.8910      1000
   macro avg     0.8927    0.8900    0.8906      1000
weighted avg     0.8921    0.8910    0.8908      1000
```

### 实验记录

- 训练及测试缓存构建完成。
- 训练与评测正常完成。
- Checkpoint：`checkpoints/qwen3_14b_seval/model_epoch_0.pt`
- Log：`logs/qwen3_14b_seval_train.log`
- 完整流程日志：`logs/qwen3_14b_seval_pipeline.log`
- 梯度累积尾部：9000 = 281 × 32 + 8，沿用原始实现。

本次 Streaming F1 比论文低 0.0070，相比旧服务器复现结果 0.8748 提高 0.0223。

---

## Llama-3.1-8B + S-Eval

### 实验配置

与前述 Qwen3-8B + S-Eval 实验配置一致，仅替换底座模型及对应数据。

- Base Model：Llama-3.1-8B-Instruct
- Dataset：StreamGuardBench v1 / S-Eval
- Train / Test：9000 / 1000
- Hidden Layer：20
- Hidden Size：4096
- Test：benign 547，harmful 453
- 适配 Llama assistant marker，保留短序列标签处理逻辑。

### 复现结果

以下 F1 均为 harmful 类（label=1）。

| Metric | 本次复现 | 论文 | 差值 |
|---|---:|---:|---:|
| Response-level F1 | **0.9095** | — | — |
| Streaming F1 | **0.9191** | 0.9590 | -0.0399 |

- Response accuracy：0.9180
- Streaming accuracy：0.9250
- 论文未单独报告该组合的 Response-level F1。

### 原始评测输出

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

           0     0.9487    0.9122    0.9301       547
           1     0.8987    0.9404    0.9191       453

    accuracy                         0.9250      1000
   macro avg     0.9237    0.9263    0.9246      1000
weighted avg     0.9260    0.9250    0.9251      1000
```

### 实验记录

- 训练及测试缓存构建完成。
- 训练与评测正常完成。
- Checkpoint：`checkpoints/llama_3_1_8b_seval/model_epoch_0.pt`
- Log：`logs/llama_3_1_8b_seval_train.log`
- 完整流程日志：`logs/llama_3_1_8b_seval_pipeline.log`
- 梯度累积尾部：9000 = 281 × 32 + 8，沿用原始实现。

---

## Figure 3 Cross-model Transfer：Qwen3-14B → Qwen3-8B

### 实验设置

复现 Figure 3 第一行第二列：

- Source Model：Qwen3-14B
- Target Model：Qwen3-8B
- Dataset：S-Eval
- Train / Test：9000 / 1000
- Hidden Layer：20
- 其余训练参数与 Qwen3-8B + S-Eval baseline 一致。

具体流程：

```text
Train responses : Qwen3-14B / S-Eval trainset
Train features  : Qwen3-8B hidden states
Test responses  : Qwen3-8B / S-Eval testset
Test features   : Qwen3-8B hidden states
PlugGuard head  : 重新训练
```

### 复现结果

以下 F1 均为 harmful 类（label=1）。

| Metric | Reproduction | Paper |
|---|---:|---:|
| Response-level F1 | **0.9101** | — |
| Streaming F1 | **0.9175** | 0.9161 |

- Response accuracy：0.9050
- Streaming accuracy：0.9110
- Streaming F1 比论文高 0.0014。

### 原始评测输出

```text
-------------Response level--------
              precision    recall  f1-score   support

           0     0.8908    0.9079    0.8993       467
           1     0.9179    0.9024    0.9101       533

    accuracy                         0.9050      1000
   macro avg     0.9043    0.9052    0.9047      1000
weighted avg     0.9052    0.9050    0.9050      1000


-----------Streaming level-----------
              precision    recall  f1-score   support

           0     0.9163    0.8908    0.9034       467
           1     0.9066    0.9287    0.9175       533

    accuracy                         0.9110      1000
   macro avg     0.9114    0.9097    0.9104      1000
weighted avg     0.9111    0.9110    0.9109      1000
```

### 实验记录

- Source train cache：9000/9000，构建完成。
- Target test cache：1000/1000，复用已有缓存。
- 训练与评测正常完成。
- Checkpoint：`checkpoints/fig3_target_qwen3_8b_source_qwen3_14b/model_epoch_0.pt`
- Log：`logs/fig3_target_qwen3_8b_source_qwen3_14b_train.log`
- Pipeline：`logs/fig3_target_qwen3_8b_source_qwen3_14b_pipeline.log`