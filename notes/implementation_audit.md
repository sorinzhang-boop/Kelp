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

当前目标是首先复现作者公开实现，因此采用以下原则：

1. 优先保持作者公开代码的实际行为。
2. 只修复导致程序无法运行的问题。
3. 不擅自修改模型结构、损失函数和数据处理逻辑。
4. 对论文和代码之间的不一致进行记录。
5. 对含义不明确的实现先保留原样。
6. 后续如果需要修改，应作为单独的消融实验，而不是直接改变 baseline。

目前已经实际修改的内容只有：

```text
train.py 增加缺失的 main() 启动入口
```

其余发现目前均只记录，不修改。