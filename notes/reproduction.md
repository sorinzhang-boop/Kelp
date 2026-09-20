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