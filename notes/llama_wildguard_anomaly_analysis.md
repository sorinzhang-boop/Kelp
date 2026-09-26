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

只进行以下三个诊断：

1. Seed sweep
2. 最后一个 gradient accumulation remainder 对照
3. 1 epoch 内 Streaming F1 收敛曲线

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

---

## 9. 当前阶段结论

目前最可靠的总结是：

> **Llama-3.1-8B + WildGuard 的低 Streaming F1 是一个稳定、可重复的异常，而不是随机种子、最后一个 optimizer step 或明显未收敛造成的。**

该异常的主要表现为：

> **harmful Recall 尚可，但 harmful Precision 偏低，即 benign response 的 Streaming false positive 明显偏多。**

并且：

> **这些 false positive 几乎全部发生在较长的 benign response 中，首次 harmful trigger 来自 response 正文内部，而不是尾部 special token。**

因此，目前问题已经从：

> “训练是不是没调好？”

转变为：

> **“为什么 Llama hidden-state trajectory 在 WildGuard 的长 benign response 上更容易出现局部 harmful trigger？”**

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

## 11. 下一步建议

下一阶段不建议继续盲目扫 seed、epoch 或 optimizer step。

更有价值的是做 **Llama vs Qwen 的 WildGuard trajectory 对比**。

### 11.1 同长度条件下比较 FP rate

在相同长度区间内比较：

- Qwen3-8B + WildGuard
- Llama-3.1-8B + WildGuard

统计：

- benign 数量
- Streaming FP 数量
- FP rate

如果同长度下 Llama 仍显著更高，说明问题不只是“Llama response 更长”。

### 11.2 比较 benign response 的 token-level harmful trajectory

对长度匹配的 benign response，比较：

- 最大 harmful score / probability
- 首次 harmful trigger 的位置
- harmful spike 数量
- spike 是否集中在某类语义片段
- trajectory 随 token 位置的变化

目标是回答：

> Llama 是否更容易在长 benign response 中产生局部 harmful spike？

### 11.3 对比 Qwen 与 Llama 的 tokenization / trajectory 长度

统计并比较：

- response token 数
- tokenizer 分词差异
- hidden trajectory 长度
- 长度与 FP 的关系

用于区分：

- 只是 token 数更多，所以 any-token 更容易误触
- 即使控制 token 数，Llama 的表征仍更容易产生 harmful spike

### 11.4 检查 Llama 特有的输入模板与 hidden-state extraction

当前 Llama 适配使用 Llama assistant marker，并沿用：

```python
tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

以及当前 assistant trajectory 截取逻辑。

虽然已有诊断表明 FP 不是尾部 special token 直接触发，但仍应进一步检查：

- 完整 token sequence
- `assistant_start` 是否严格正确
- response 后是否存在额外 assistant header
- Llama 与 Qwen 的模板差异是否改变 hidden trajectory

这属于下一阶段较高优先级的实现检查。

---

## 12. 实验产物

本轮诊断日志目录：

```text
/data1/plugguard_repro/logs/llama_wildguard_diagnostics/20260926_164936/
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

## 13. 一句话总结

> **Llama + WildGuard 的 Streaming F1 异常低是稳定现象：5 个 seed 均无法接近论文结果，补齐最后一个 optimizer step 无影响，1 epoch 后半程 F1 已基本平台化；当前最主要的错误模式是长 benign response 中出现正文内部的局部 harmful trigger，导致 any-token Streaming 判定产生大量 false positive。**
