# PlugGuard 实现审计记录

## 1. `train.py` 缺少启动入口

原始 `train.py` 定义了 `main()`，但文件末尾没有：

```python
if __name__ == "__main__":
    main()
```

因此直接执行：

```bash
python train.py
```

不会真正进入训练流程。

本地已经补上该入口。

该修改只用于修复程序无法启动的问题，不改变模型结构、损失函数或训练算法。

---

## 2. 论文与代码中的训练配置存在差异

目前发现两个主要差异。

### weight decay

论文中使用：

```text
weight decay = 0
```

而 `train.py` 默认值为：

```text
weight decay = 0.1
```

因此正式复现时需要明确采用论文配置还是代码默认配置。

### 使用的隐藏层位置不统一

不同文件中的默认值不同：

```text
train.py       idx_layer = 32
dataset.py     idx_layer = 20
eval.py 默认   idx_layer = 20
eval.py 示例   idx_layer = 21
```

因此公开代码没有给出一个完全统一的 Qwen3-8B 层数设置。

正式训练前需要进一步确认。

---

## 3. Qwen3 assistant 起点定位正常

`dataset.py` 使用下面的特殊 token 序列寻找 assistant response 的起点：

```text
<|im_start|>assistant\n
```

使用 Qwen3-8B tokenizer 和一条官方 S-Eval 数据进行了实际测试。

测试结果表明：

- 能够正确找到该 marker
- `assistant_start` 确实位于 response 的开头

因此对于当前计划复现的 Qwen3-8B，这部分实现可以正常工作。

需要注意的是，这种写法是 Qwen 风格的 chat template，不能直接认为对 Llama、InternLM 等其他模型同样适用。

---

## 4. response 结束后存在额外的 assistant header

`dataset.py` 中构造的 messages 已经包含完整的：

```text
user prompt
assistant response
```

但调用 `apply_chat_template()` 时仍然设置了：

```python
add_generation_prompt=True
```

实际测试发现，Qwen3 最终得到的序列末尾类似：

```text
response 最后正文
<|im_end|>
\n
<|im_start|>
assistant
\n
```

也就是说，完整 response 结束后，又额外添加了一段：

```text
<|im_start|>assistant\n
```

这不是下一条数据的 prompt，而是 chat template 添加的“准备开始下一次 assistant 生成”的标记。

目前公开论文和代码中没有说明为什么保留这一段。

它有可能只是实现细节，也有可能被作者当作一种 response 结束后的 readout 位置。

初次复现暂时保持作者代码，不修改。

---

## 5. `eval.py` 中的 `pred[-2]` 不是 response 最后正文 token

`eval.py` 中 response-level 指标使用：

```python
pred[-2]
```

作为整条 response 的最终预测。

论文描述该指标时使用的是：

```text
last-token decision
final-token score
```

但实际检查 Qwen3 的 token 序列后发现：

```text
最后正文 token
<|im_end|>
\n
<|im_start|>
assistant
\n
```

同时 `dataset.py` 设置：

```python
assistant_end = -1
```

因此最后一个 `\n` 被去掉，而剩余序列的尾部大致为：

```text
最后正文 token
<|im_end|>
\n
<|im_start|>
assistant
```

在这种情况下：

```text
pred[-1]  -> assistant
pred[-2]  -> <|im_start|>
```

所以当前代码中的 `pred[-2]` 实际对应 response 结束后额外出现的 `<|im_start|>`，而不是：

```text
最后正文 token
```

也不是：

```text
<|im_end|>
```

论文没有进一步解释这种差异。

初次复现时暂时保持作者原始的 `pred[-2]`，不自行修改。

以后如果需要，可以单独比较：

```text
最后正文 token
<|im_end|>
post-response <|im_start|>
```

三种位置的结果。

---

## 6. ATC 最后 10 个监督位置可能包含特殊 token

`dataset.py` 中设置：

```python
num_supervised_token = 10
```

并将 assistant 序列最后 10 个位置的 label 设置为整条 response 的安全标签。

但是由于序列尾部包含：

```text
<|im_end|>
\n
<|im_start|>
assistant
```

因此所谓“最后 10 个 token”并不一定全部属于 response 正文。

其中可能包含 response 结束符和额外的 assistant header。

这意味着公开代码中的 ATC tail supervision 与“最后 10 个 response 正文 token”并不完全等价。

初次复现时仍然保持作者代码，不修改这些位置。

---

## 7. 当前复现原则

当前目标是在尽量保持作者公开实现核心行为的前提下，完成可重复、可扩展的 PlugGuard baseline 复现，并逐步覆盖不同数据集和 backbone。

采用以下原则：

1. **优先保持作者公开实现的算法行为。**  
   不擅自修改 PlugGuard 的模型结构、SLD 模块、ATC 监督方式、损失函数主体和 Streaming / Response 评测规则。

2. **允许进行不改变算法语义的工程性修改。**  
   包括修复程序无法启动的问题、集中管理配置、统一模型和训练参数、整理实验路径，以及增加复现实验所需的日志和记录。

3. **论文与代码默认配置冲突时，优先采用论文明确给出的实验配置。**  
   例如当前正式复现采用：
   - learning rate = `5e-5`
   - weight decay = `0`
   - warmup ratio = `0.05`
   - epoch = `1`
   - batch size = `1`
   - gradient accumulation = `32`
   - max length = `4096`
   - ATC supervised tokens = `10`
   - seed = `42`

4. **模型相关配置统一放入 `config.py`。**  
   当前通过 `ACTIVE_MODEL` 和 `MODEL_CONFIGS` 管理不同 backbone 的：
   - Hugging Face model name
   - hidden layer index

   已支持：
   - Qwen3-8B
   - Qwen3-14B

   当前两者均按照论文的 layer-selection heuristic 使用 `idx_layer = 20`。

5. **不同实验的数据、cache、checkpoint 和 log 必须相互隔离。**  
   不覆盖已经完成的 baseline 结果。不同 backbone / dataset 分别使用独立目录，以便后续比较和追溯。

6. **对论文与代码之间的不一致进行记录，但 baseline 默认不擅自修正。**  
   当前已发现并记录的例子包括：
   - Qwen3 chat template 在完整 response 后额外添加 assistant header
   - `eval.py` 的 `pred[-2]` 并非严格意义上的 response 最后正文 token
   - ATC 最后 10 个监督位置可能包含特殊 token
   - gradient accumulation 无法整除数据集时，最后不足一个 accumulation window 的梯度不会触发 `optimizer.step()`

7. **对于含义不明确的实现，优先通过实验验证其实际影响，而不是直接修改。**  
   例如已经对 Qwen3-8B + S-Eval 进行了 response readout sweep，确认 `pred[-5]` 到 `pred[-1]` 的 Response F1 完全一致，因此继续保留作者默认 `pred[-2]` 实现。

8. **新的 backbone 在正式实验前优先做最小 smoke test。**  
   用少量样本确认模型可以加载、目标 hidden layer 可以读取、hidden size 与 PlugGuard head 兼容、assistant boundary 和 labels 长度正常，再启动完整实验。

9. **baseline 与后续修正版 / 消融实验严格区分。**  
   如果后续需要修改 assistant boundary、训练尾部监督、gradient accumulation 行为、hidden layer 或其他算法相关实现，应作为单独实验，不直接覆盖当前 baseline。

### 当前已经实际修改的内容

相对于作者公开代码，目前主要进行了以下复现工程修改：

```text
1. train.py
   - 补充缺失的 main() 启动入口
   - 从统一配置读取模型和训练参数
   - 正式复现采用论文给出的 weight decay = 0
   - 评测阶段使用统一的 max_length / batch_size 等配置

2. config.py
   - 新增统一复现配置
   - 集中管理训练超参数
   - 集中管理 Qwen3-8B / Qwen3-14B 的 model_name 和 idx_layer

3. dataset.py
   - 从 config.py 读取 idx_layer、max_length 和 num_supervised_token
   - 保留作者原有的数据构造、assistant boundary 和 cache 逻辑

4. eval.py
   - 从 config.py 读取 model_name、idx_layer 和 max_length
   - 保留作者原有 Response-level `pred[-2]` 和 Streaming-level `max(pred)` 评测逻辑