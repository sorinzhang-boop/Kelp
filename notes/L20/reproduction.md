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