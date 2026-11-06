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