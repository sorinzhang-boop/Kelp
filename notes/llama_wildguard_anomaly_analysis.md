# Llama-3.1-8B + WildGuard Streaming F1 异常诊断

## 1. 问题背景

在 PlugGuard/Kelp 的单模型复现中，`Llama-3.1-8B-Instruct + WildGuard` 是目前最明显的异常组合。

论文中该组合的 **Streaming F1 = 0.8115**，而当前复现结果为：

- Response-level harmful F1：`0.7404`
- Streaming harmful F1：`0.6969`

两者差值：

`0.8115 - 0.6969 = 0.1146`

即约 **11.46 个百分点**。

相比之下，其他组合与论文的差距明显更小：

| Target model | Dataset | Paper Streaming F1 | Reproduction | Delta |
|---|---|---:|---:|---:|
| Qwen3-8B | S-Eval | 0.9246 | 0.9069 | -0.0177 |
| Qwen3-8B | WildGuard | 0.8333 | 0.8199 | -0.0134 |
| Qwen3-14B | S-Eval | 0.9041 | 0.8748 | -0.0293 |
| Qwen3-14B | WildGuard | 0.7845 | 0.7770 | -0.0075 |
| Llama-3.1-8B | S-Eval | 0.9590 | 0.9214 | -0.0376 |
| **Llama-3.1-8B** | **WildGuard** | **0.8115** | **0.6969** | **-0.1146** |

因此，本轮诊断重点回答以下问题：

1. 是否是随机种子（seed）导致的偶然低值？
2. 是否是 optimizer step / gradient accumulation 设置导致的？
3. 是否是 1 epoch 训练没有收敛？
4. 如果以上因素都不能解释，异常具体表现在哪里？

---

## 2. Baseline 的错误模式

Llama + WildGuard 测试集共有：

- benign：`1519`
- harmful：`206`
- total：`1725`

seed=42 baseline 的 Streaming 分类结果：

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0（benign） | 0.9744 | 0.9276 | 0.9504 | 1519 |
| 1（harmful） | **0.6057** | **0.8204** | **0.6969** | 206 |

对应混淆矩阵：

| | Pred benign | Pred harmful |
|---|---:|---:|
| True benign | TN = 1409 | FP = 110 |
| True harmful | FN = 37 | TP = 169 |

可以看到：

> harmful 类 Recall 尚可，但 Precision 明显偏低。主要问题是较多 benign response 在 Streaming 判定中被误报为 harmful。

Streaming 使用 any-token 规则：只要 response 中任意 token 被预测为 harmful，整条 response 就判为 harmful。因此，一个 benign response 中只要出现一次错误触发，就会形成 Streaming FP。

---

## 3. 之前的 FP 位置与长度诊断

### 3.1 FP 不是由尾部 special token 触发

对 seed=42 baseline 的 110 个 Streaming FP 统计首次 harmful trigger：

| 首次 harmful trigger 位置 | 数量 |
|---|---:|
| 最后 3 个 token | 0 |
| <code>&lt;&#124;eot_id&#124;&gt;</code> | 0 |
| response 最后一个正文 token | 0 |
| 更早的 response 正文 token | 110 |

结论：

> 误报来自 response 正文内部，而不是尾部 special token。

因此，“Llama 尾部 special token 被误判”不是主要解释。

### 3.2 FP 与 response token 长度明显相关

这里的 response 长度按 **token 数**统计。

| Response token 长度 | Benign 数量 | Streaming FP | FP rate |
|---|---:|---:|---:|
| ≤50 | 489 | 0 | 0% |
| 51–100 | 42 | 0 | 0% |
| 101–200 | 36 | 2 | 5.56% |
| >200 | 952 | 108 | 11.34% |

另外：

- benign response 平均长度约 `370.9` tokens
- FP response 平均长度约 `649.5` tokens
- 110 个 FP 中，108 个来自长度 `>200` 的 benign response

当前可以确认：

> 长 benign response 与 Streaming FP 显著相关。

合理解释是：any-token 判定会随着序列变长，积累更多“至少出现一次错误 harmful trigger”的机会。

但当前证据只能说明相关性，不能直接证明“response 长度就是根因”。

---

## 4. 本轮训练诊断设计

为了避免反复修改参数并耗时重跑，本轮直接复用已有 Llama hidden-state cache，只重新训练 PlugGuard head。

Cache 检查结果：

| 项目 | 数值 |
|---|---:|
| Train cache | 37934 |
| Test cache | 1725 |
| Hidden dim | 4096 |
| Layer | 20 |
| Max length | 4096 |

本轮没有重新运行 Llama-3.1-8B 提取 hidden states，也没有覆盖原 cache。

保持当前复现配置不变：

- learning rate：`5e-5`
- weight decay：`0`
- warmup ratio：`0.05`
- epoch：`1`
- micro batch size：`1`
- gradient accumulation：`32`
- effective global batch：`32`
- supervised tokens：`10`
- PlugGuard architecture：不变

本轮及后续追加诊断包括：

1. Seed sweep
2. 最后一个 gradient accumulation remainder 对照
3. 1 epoch 内 Streaming F1 收敛曲线
4. inference-time `dt` 论文设置与公开代码设置对照
5. 实际 cache sequence length / max length 检查
6. ATC supervised token 数 `N` 的多卡 sweep 与 seed robustness 对照

---

## 5. Seed sweep 结果

共测试 5 个 seed：

| Seed | Streaming Precision | Streaming Recall | Streaming F1 | FP | FN | Response F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6159 | 0.8252 | 0.7054 | 106 | 36 | 0.7670 |
| 21 | 0.6653 | 0.7718 | **0.7146** | 80 | 47 | 0.7321 |
| 42 | 0.6057 | 0.8204 | 0.6969 | 110 | 37 | 0.7404 |
| 123 | 0.5825 | 0.8398 | 0.6879 | 124 | 33 | 0.7500 |
| 3407 | 0.5828 | 0.8204 | **0.6815** | 121 | 37 | 0.7506 |

统计结果：

| Statistic | Streaming F1 |
|---|---:|
| Mean | 0.6972 |
| Std | 0.0119 |
| Min | 0.6815 |
| Max | 0.7146 |
| Paper | 0.8115 |

即使选择本轮最好的 seed=21：

`0.8115 - 0.7146 = 0.0969`

仍低约 **9.69 个百分点**。

### 结论

> 随机种子会造成一定波动，但无法解释论文与复现之间约 11.46 个百分点的整体差距。

5 个 seed 的 Streaming F1 范围为：

`0.6815 ~ 0.7146`

跨度约 `0.0331`。

同时，5 个 seed 都表现出相似的错误模式：

- Recall 大致在 `0.77 ~ 0.84`
- Precision 大致在 `0.58 ~ 0.67`
- Streaming FP 持续偏多

因此，seed=42 不是一个偶然“特别差”的异常 seed。

---

## 6. Gradient accumulation 最后一组样本诊断

WildGuard train size：

`37934 = 1185 × 32 + 14`

当前原始训练逻辑只有在：

```python
if (step + 1) % gradient_acc_steps == 0:
    optimizer.step()
```

时执行 `optimizer.step()`。

因此原 baseline：

- 实际 optimizer steps：`1185`
- 最后 14 条样本虽然执行 backward，但没有产生最后一次 optimizer update

为排除该问题，额外运行 seed=42 的 remainder-flush 对照。

| Setting | Optimizer steps | Precision | Recall | Streaming F1 | FP | FN | Response F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 1185 | 0.6057 | 0.8204 | **0.6969** | 110 | 37 | 0.7404 |
| Flush final remainder | 1186 | 0.6057 | 0.8204 | **0.6969** | 110 | 37 | 0.7404 |

所有最终分类指标完全一致。

### 结论

> 最后一个 incomplete gradient accumulation update 对当前异常没有可观测影响，可以排除为主要原因。

---

## 7. 收敛诊断

seed=42 在同一个 1-epoch 训练过程中，于约 25%、50%、75%、100% 进度对同一 test set 做 Streaming 评测。

| 训练进度 | Step | Precision | Recall | Streaming F1 | FP | FN | Response F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25% | 296 | 0.5157 | 0.7961 | **0.6266** | 154 | 42 | 0.6810 |
| 50% | 593 | 0.6022 | 0.8155 | **0.6928** | 111 | 38 | 0.7252 |
| 75% | 889 | 0.6036 | 0.8204 | **0.6955** | 111 | 37 | 0.7393 |
| 100% | 1185 | 0.6057 | 0.8204 | **0.6969** | 110 | 37 | 0.7404 |

Streaming F1 的增量：

- 25% → 50%：`+0.0662`
- 50% → 75%：`+0.0027`
- 75% → 100%：`+0.0014`

即：

`0.6266 → 0.6928 → 0.6955 → 0.6969`

训练前半程仍明显提升，但从 50% 之后已经进入明显平台。

从 step 593 到最终：

- F1：`0.6928 → 0.6969`
- Recall：`0.8155 → 0.8204`
- FP：`111 → 110`
- FN：`38 → 37`

变化都很小。

### Training loss 趋势

诊断脚本额外记录了每个 optimizer update 对应的未被 gradient accumulation 重复缩放的：

- raw total loss
- raw CE loss
- raw smooth loss
- anchor accuracy
- learning rate

由于单个 optimizer step 只对应 32 条随机训练样本，不同 step 的样本长度、类别和监督位置均不同，因此单个 step 的 loss 波动较大，不适合直接用某几个瞬时 loss 判断是否收敛。

因此，这里将整个 1-epoch 训练过程按 optimizer step 分成四段，对每一段约 296 个 update 的 loss 取平均。

| 训练区间 | Step 范围 | Mean raw total loss | Mean raw CE loss | Mean raw smooth loss | Mean anchor accuracy |
|---|---:|---:|---:|---:|---:|
| 0–25% | 1–296 | 0.159832 | 0.159699 | 0.000134 | 0.9411 |
| 25–50% | 297–593 | 0.090333 | 0.090266 | 0.000067 | 0.9614 |
| 50–75% | 594–889 | 0.081711 | 0.081662 | 0.000049 | 0.9663 |
| 75–100% | 890–1185 | 0.079147 | 0.079099 | 0.000047 | 0.9662 |

分段平均 raw total loss 的变化为：

`0.159832 → 0.090333 → 0.081711 → 0.079147`

各阶段下降幅度分别为：

- 0–25% → 25–50%：`-0.069499`
- 25–50% → 50–75%：`-0.008622`
- 50–75% → 75–100%：`-0.002564`

可以看到，loss 在训练前半程快速下降，而后半程下降幅度明显减小。

同时，anchor accuracy：

`0.9411 → 0.9614 → 0.9663 → 0.9662`

在后半程也基本稳定。

raw smooth loss 始终只有约 `1e-4 ~ 1e-5`，远小于 CE loss，因此当前 total loss 的整体变化主要由 CE loss 决定。

另外，训练最后一段的局部平均值为：

| 区间 | Mean raw total loss | Mean raw CE loss | Mean anchor accuracy |
|---|---:|---:|---:|
| Last 100 steps | 0.082229 | 0.082182 | 0.9659 |
| Last 50 steps | 0.084214 | 0.084169 | 0.9661 |

最后 50/100 step 的平均 loss 没有继续单调下降，而是在约 `0.08` 附近波动。这与不同 accumulation block 的样本组成有关，但至少没有表现出训练结束前仍持续快速下降的趋势。

### Training metric 与 Streaming F1 的联合判断

仅看 training loss 不能直接证明模型已经达到最优，因此还需要结合固定 test set 上的 Streaming F1。

Streaming F1：

`0.6266 → 0.6928 → 0.6955 → 0.6969`

对应：

| 训练进度 | Streaming F1 |
|---|---:|
| 25% | 0.6266 |
| 50% | 0.6928 |
| 75% | 0.6955 |
| 100% | 0.6969 |

其增量为：

- 25% → 50%：`+0.0662`
- 50% → 75%：`+0.0027`
- 75% → 100%：`+0.0014`

因此可以观察到一致的两阶段现象：

1. 训练前半程：training loss 快速下降，同时 Streaming F1 快速提升；
2. 训练后半程：training loss 下降幅度显著减小，同时 Streaming F1 基本进入平台。

此外，cosine learning-rate schedule 在 1 epoch 内也已经完整执行，训练末期 learning rate 已衰减至接近 0。

需要说明的是，中间的 test-set 评测仅用于复现后的诊断，没有用于选择 checkpoint、early stopping 或调节超参数。

### 结论

> 当前结果不能证明模型在严格优化意义上已经达到全局最优，也不能证明增加 epoch 一定不会带来任何变化。

但 training loss 和 Streaming F1 给出了相互一致的证据：

> **在论文规定的 1-epoch training budget 内，模型已经从前半程的快速学习阶段进入后半程的平台阶段。没有证据表明论文结果与复现结果之间约 0.1146 的 Streaming F1 差距主要来自“训练结束时仍明显未收敛”。**

因此，“明显未收敛”可以基本排除为当前异常的主要原因。

---

## 8. 当前可以基本排除的因素

### 8.1 Seed=42 的偶然异常

5 个 seed：

- mean：`0.6972`
- std：`0.0119`
- range：`0.6815 ~ 0.7146`

全部显著低于论文结果 `0.8115`。

### 8.2 最后一个 optimizer step 缺失

- 1185 steps：`0.6969`
- 1186 steps：`0.6969`

结果完全一致。

### 8.3 明显的未收敛

- 50%：`0.6928`
- 75%：`0.6955`
- 100%：`0.6969`

后半个 epoch 已进入平台区。

### 8.4 Inference-time dt 设置

论文描述的 SLD 时间步长为：

- training：`dt = 1 / Ta`
- inference：`dt = 1 / 2048`

而当前公开代码在 inference 时仍根据 assistant trajectory 长度构造 `dt`，近似为：

`dt ≈ 1 / (Ta - 1)`

因此使用同一个 seed=42 checkpoint 和同一份 WildGuard test cache，仅修改 inference-time `dt` 做对照。

| Setting | Response F1 | Streaming Precision | Streaming Recall | Streaming F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| Public code：`dt ≈ 1/(Ta-1)` | 0.7404 | 0.6057 | 0.8204 | 0.6969 | 110 | 37 |
| Paper inference：`dt = 1/2048` | 0.7404 | 0.6057 | 0.8204 | 0.6969 | 110 | 37 |

两种设置下：

- Streaming prediction changed：`0 / 1725`
- Response-level prediction changed：`0 / 1725`

#### 结论

> 论文与公开代码的 inference-time `dt` 设置确实存在差异，但在当前 Llama + WildGuard checkpoint 上，将 `dt` 改为论文描述的 `1/2048` 后，没有任何测试样本改变最终分类结果。

因此：

> **Inference-time `dt` mismatch 不能解释当前 Streaming F1 从论文 0.8115 到复现 0.6969 的差距。**

该实验只排除了 inference-time `dt` 的影响，不代表训练阶段 `dt` 的所有实现差异均已被严格验证。

---

### 8.5 Max sequence length / truncation 检查

论文实验设置 maximum sequence length 为 `4096`。

当前公开代码在 `apply_chat_template(..., tokenize=False)` 时传入了 `max_length` 和 `truncation`，但随后真正执行 tokenizer 的调用没有再次显式传入这两个参数。因此检查现有 Llama + WildGuard cache 的实际 sequence length。

#### Train cache

| Statistic | Full sequence length |
|---|---:|
| Samples | 37934 |
| P50 | 467 |
| P90 | 906 |
| P95 | 1017.3 |
| P99 | 1388 |
| Max | 2650 |
| `>4096` | 0 |

#### Test cache

| Statistic | Full sequence length |
|---|---:|
| Samples | 1725 |
| P50 | 574 |
| P90 | 919 |
| P95 | 1009.8 |
| P99 | 1282.6 |
| Max | 2303 |
| `>4096` | 0 |

另外，对所有 train/test cache 均有：

`(seq_len - assistant_start) - assistant_len = 0`

说明当前 cache 中的 sequence length、assistant start 和 assistant trajectory length 在长度关系上完全一致。

#### 结论

> Train 和 test 中均不存在超过 4096 token 的 cached sequence。

因此：

> **虽然当前 tokenizer 调用存在潜在的 truncation 实现隐患，但该问题没有在现有 Llama + WildGuard 数据上实际触发，不能解释当前 Streaming F1 异常。**

---

### 8.6 ATC supervised token 数 N 的对照实验

论文主实验使用 `N=10`。为检查 supervised token 数是否导致 Llama + WildGuard 的 Streaming F1 异常，在保持其他训练配置不变的情况下，复用已有 hidden-state cache，对 `N=1~10` 进行 sweep。

固定：

- seed：`42`
- learning rate：`5e-5`
- weight decay：`0`
- epoch：`1`
- effective global batch：`32`
- idx layer：`20`
- 其余 PlugGuard 结构及训练逻辑不变

#### Seed=42 的 N sweep

| N | Precision | Recall | Streaming F1 | FP | FN | Response F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.2385 | 0.9854 | 0.3841 | 648 | 3 | 0.7447 |
| 2 | 0.2164 | 0.9854 | 0.3549 | 735 | 3 | 0.7593 |
| 3 | 0.2371 | 0.9854 | 0.3823 | 653 | 3 | 0.7677 |
| 4 | 0.2570 | 0.9806 | 0.4073 | 584 | 4 | 0.7554 |
| 5 | 0.4013 | 0.9272 | 0.5601 | 285 | 15 | 0.7512 |
| 6 | 0.5394 | 0.8641 | 0.6642 | 152 | 28 | 0.7458 |
| 7 | 0.5743 | 0.8447 | 0.6837 | 129 | 32 | 0.7391 |
| 8 | 0.5911 | 0.8350 | 0.6922 | 119 | 34 | 0.7373 |
| 9 | 0.6014 | 0.8204 | 0.6940 | 112 | 37 | 0.7373 |
| 10 | 0.6057 | 0.8204 | **0.6969** | 110 | 37 | 0.7404 |

`N=10, seed=42` 精确复现了之前的 baseline：

`Streaming F1 = 0.6969`

说明本次从现有 cache 重建不同 N 的监督标签并重新训练的实验流程与原 baseline 一致。

从结果看，较小 N 下 harmful Recall 很高，但 Precision 很低，主要原因是 Streaming FP 大量增加。

例如：

`FP: 648 (N=1) → 285 (N=5) → 152 (N=6) → 110 (N=10)`

随着 N 增大，FP 显著下降，Precision 上升；同时 Recall 有一定下降。最终在当前 seed=42 下，Streaming F1 在 `N=8~10` 附近逐渐进入平台。

#### Seed robustness

进一步对 `N=4/6/8` 分别使用 seed=21、42、123：

| N | Seed 21 F1 | Seed 42 F1 | Seed 123 F1 | Mean | Std |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.7306 | 0.4073 | 0.3725 | 0.5034 | 0.1612 |
| 6 | 0.7182 | 0.6642 | 0.6571 | 0.6798 | 0.0273 |
| 8 | 0.7149 | 0.6922 | 0.6874 | 0.6982 | 0.0120 |

结合之前 `N=10` 的 5-seed 实验：

`mean = 0.6972, std = 0.0119`

可以看到：

- 较小 N 的结果对 seed 非常敏感；
- 随 N 增大，seed 方差明显降低；
- `N=8~10` 已进入较稳定的平台区；
- 当前论文默认的 `N=10` 并不是导致异常低 F1 的明显错误设置。

本轮 16 个 N/seed 实验中最高 Streaming F1 为：

`N=4, seed=21：0.7306`

仍低于论文结果 `0.8115`：

`0.8115 - 0.7306 = 0.0809`

即仍有约 **8.09 个百分点**差距。

#### 结论

> **ATC supervised token 数 N 会显著影响 Streaming Precision/Recall trade-off，特别是较小 N 会产生大量 FP 和较强的 seed 敏感性。**

但：

> **在当前 Llama + WildGuard 上，N=8~10 已形成相对稳定的性能平台，默认 N=10 不能解释论文与复现之间的主要差距。**

另外，N sweep 中 Response-level F1 始终约为 `0.74~0.77`，而 Streaming F1 变化范围很大。这进一步说明当前异常主要体现在 token-level Streaming 判定，而不是最终 response-level 分类完全失效。


---

## 9. 当前阶段结论

经过目前的对照实验，已经依次检查：

- 随机种子
- optimizer update remainder
- 1-epoch 收敛情况
- inference-time `dt`
- maximum sequence length / truncation
- ATC supervised token 数 `N`

当前结果如下：

| Diagnostic | Result |
|---|---|
| Seed | 5-seed mean `0.6972 ± 0.0119`，无法接近论文 `0.8115` |
| Optimizer remainder | 1185 vs 1186 steps 均为 `0.6969` |
| Convergence | training loss 与 Streaming F1 在后半程均进入平台 |
| Inference `dt` | 改为论文 `1/2048` 后 1725 条 prediction 全部不变 |
| Max length | train/test 均无 sequence `>4096` |
| ATC `N` | `N=8~10` 进入稳定平台，默认 `N=10` 不是明显异常来源 |

因此目前可以较有把握地认为：

> **Llama-3.1-8B + WildGuard 的 Streaming F1 异常低不是由单一的 seed、最后一个 optimizer step、明显未收敛、inference-time dt、4096 truncation 或默认 N=10 所造成。**

当前最稳定的错误模式仍然是：

> **harmful Recall 尚可，但 harmful Precision 偏低，即 Streaming false positive 偏多。**

在 baseline 中：

- harmful Precision：`0.6057`
- harmful Recall：`0.8204`
- FP：`110`
- FN：`37`

此前的位置诊断还表明：

- 110 个 FP 的首次 harmful trigger 均出现在 response 正文内部；
- 不是尾部 special token 触发；
- 110 个 FP 中有 108 个来自长度 `>200 token` 的 benign response。

因此当前最可靠的实验性结论是：

> **Llama + WildGuard 的异常主要表现为长 benign response 的 token-level Streaming false positive，而不是最终 response-level 分类完全失效。**

目前已有实验能够排除若干简单的训练配置解释，但尚未定位论文结果 `0.8115` 与当前复现 `0.6969` 之间差距的唯一根因。

---

## 10. 当前尚不能下的结论

当前实验还不能证明：

1. 长 response 本身就是根因
2. Llama tokenizer 一定导致异常
3. Llama hidden state 天生比 Qwen 更难做 Streaming safety detection
4. 论文结果有误
5. 增加 epoch 数一定会解决问题
6. 修改阈值一定能恢复论文结果

这些都需要进一步对照实验。

---

## 11. 实验产物

本轮诊断日志目录：

```text
/data1/plugguard_repro/logs/llama_wildguard_diagnostics/20260926_164936/

```
Inference dt 对照

```text
/data1/plugguard_repro/diagnostics/llama_wildguard_dt_compare.json

```

主要实验：

```text
seed1_orig
seed21_orig
seed42_orig
seed123_orig
seed3407_orig
seed42_flush
```

其中：

```text
seed42_orig.log
```

包含 25% / 50% / 75% / 100% 的收敛诊断。

最终汇总文件位于对应 diagnostics 输出目录中的：

```text
summary.csv
```

---

## 12. 一句话总结

> **Llama + WildGuard 的 Streaming F1 异常低是稳定、可重复的现象：多 seed、optimizer step、收敛状态、inference dt、sequence truncation 和 ATC N 均不能解释论文 0.8115 与复现 0.6969 的主要差距；当前最稳定的异常表现是长 benign response 正文内部出现 token-level harmful false positive，导致 any-token Streaming 判定的 Precision 明显下降。**
