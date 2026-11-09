# 四方向跨模型 Adapter：实验五～八（B/C/E/F）

本包是沿用 `affine_llama_to_qwen8_bcef.py` 的实验协议的四方向泛化版本：在**旧模型／目标模型的原生 Safety Head 保持冻结**的条件下，训练 Source Layer 20 → Target Layer 20 的 Affine Adapter。

## 文件与方向

| 实验 | 入口文件 | Source（新模型） | Target（冻结围栏所属旧模型） | 维度 |
|---|---|---|---|---|
| 实验五 | `exp5_qwen8_to_qwen14.py` | Qwen3-8B | Qwen3-14B | 4096→5120 |
| 实验六 | `exp6_llama_to_qwen14.py` | Llama-3.1-8B | Qwen3-14B | 4096→5120 |
| 实验七 | `exp7_qwen8_to_llama.py` | Qwen3-8B | Llama-3.1-8B | 4096→4096 |
| 实验八 | `exp8_qwen14_to_llama.py` | Qwen3-14B | Llama-3.1-8B | 5120→4096 |

公共模块分工：

- `cross_adapter_alignment.py`：数据路径、缓存完整性、缺失 hidden-state 自动提取、tokenizer offset、字符对齐、训练精确配对及预检。
- `cross_adapter_training.py`：归一化与 Adapter、checkpoint/断点恢复、冻结围栏推理、指标 CSV/JSON、CLI。
- `run_exp5_to_exp8_sequential.sh`：四方向**严格串行**调度，每个方向分别执行审计后从缓存准备到 B/C/E/F；只用物理 GPU 0。

## 实验定义

每一方向均固定一个 Target 原生 Safety Head（**不重新训练**）。对同一份已存储的 S-Eval prompt+response，Source/Target LLM 分别提取隐藏状态（缓存）以构造 Adapter 对齐数据。无安全标签参与 Adapter 的 MSE 训练。

- **B**：在 Source Train Response 上训练一个 Adapter，在 Source Test Response 上评测。
- **C**：Target 模型对 Source Test 文本直接提取真实表征，输入冻结 Target Safety Head（无 Adapter，仅诊断）。
- **E**：在 Target Train Response 上训练另一 Adapter，在 Target Test Response 上评测。
- **F**：复用 E 的**同一个 Adapter**，在 Source Test Response 上评测。

## 与 Llama→Qwen3-8B 保持一致的对齐协议

1. 两个模型各自用原生 tokenizer 和 chat template 处理同一个 prompt+response。定位各自原始 prompt/response 子串，记录 token 的字符 offset；**Prompt 与 Response 各自独立计算相对字符结束位置**。
2. **训练时**：只从两边相对字符结束位置**完全相同**、且同为 Prompt 或同为 Response 的**内容 token**中配对；header、chat-template 和特殊 token 不参与 MSE。每条样本最多 128 对，先抽不超过 32 个 Prompt token，再从 Response 抽剩余。归一化统计用训练子集前 320 条；训练/验证 8100/900，seed42，3 Epoch，AdamW，LR `3e-4`，weight decay 0，梯度累积 8。
3. **评测时**：以 **Target（冻结 Safety Head 所属模型）** 的 tokenizer 网格及 `assistant_start`、标签为准；对每个 Target 内容 token，选择同段 raw text 中字符结束位置不晚于该位置的最近已完成 Source token hidden state。Target 的非内容 token 使用脚本中的来源前缀／最近状态回填。Source hidden states 经 Adapter 后输入 Target Head；C 直接喂 Target 的真实 hidden states。
4. F1 口径：Response = `pred[-2]`，Streaming = `max(pred)`，均统计 harmful label 1。**这是离线、完整 Response 已知条件下的 token-triggered F1，不宣称已经证明严格在线因果性或时延。**

特别说明：Qwen3-8B 与 Qwen3-14B 即使 token 序列完全相同，也统一应用上述「内容 token 精确配对 + Target 网格特殊 token 处理」协议；这与最初按索引全 token 训练的实验一 P0 **不是同一个协议**。比较结果时应优先参照实验四的 P1/P2，而不是默认只看 P0。

## 上传和运行

所有 `.py` 和 `.sh` 文件都应直接位于服务器：

```text
~/plugguard_repro/Kelp/
```

进入 VS Code Remote-SSH 终端，先单独预检：

```bash
cd ~/plugguard_repro/Kelp
source ../.venv/bin/activate
python -m py_compile cross_adapter_alignment.py cross_adapter_training.py exp{5,6,7,8}_*.py
bash -n run_exp5_to_exp8_sequential.sh
for f in exp5_qwen8_to_qwen14.py exp6_llama_to_qwen14.py exp7_qwen8_to_llama.py exp8_qwen14_to_llama.py; do
    python "$f" --mode audit || break
done
```

预检会检查三个模型目录、四类数据集 9000/1000 条、目标 Safety Head checkpoint 的维度、已有 cache 文件数量、部分 tokenizer/cache 一致性、训练/测试 Prompt 重叠、磁盘剩余空间。若预检失败，**不要直接启动**四方向总控；按实际错误修正路径或缺失文件。

再查看 GPU 0 是否空闲（避免同时运行别的实验）：

```bash
nvidia-smi -i 0
```

新建一个 **普通（非嵌套）tmux 会话**：

```bash
tmux new -s four_directions
cd ~/plugguard_repro/Kelp
bash run_exp5_to_exp8_sequential.sh
```

退出但保持服务器实验运行：`Ctrl+B`、松开、再按**小写 d**。之后 `tmux attach -t four_directions` 恢复。

## 运行顺序、日志与失败处理

总控预检四组后，按以下顺序严格串行：

1. 实验五：Qwen3-8B → Qwen3-14B；
2. 实验六：Llama → Qwen3-14B；
3. 实验七：Qwen3-8B → Llama；
4. 实验八：Qwen3-14B → Llama。

每个方向内部：`prepare-caches → prepare-maps → smoke → eval-c → train-b → eval-b → train-legacy → eval-e → eval-f → report`。

缓存缺失时只加载一个底座模型；如果缓存已齐，不会加载底座；仅 Adapter 训练需要优化器。**如果任一步失败，则立即停止整个序列，不会在错误状态下继续下一个方向。**

```bash
# 另开普通 SSH 终端看全局进度
tail -f ~/plugguard_repro/logs/exp5_to_exp8_master.log

# 查看最终结果（全部完成后）
cat ~/plugguard_repro/checkpoints/cross_adapt_four_directions_v1/summary_16_results.csv
```

单组日志：`~/plugguard_repro/logs/exp5_qwen8_to_qwen14.log` 等。全部完成标志：

```text
=== ALL EXPERIMENTS 5/6/7/8 COMPLETE ... ===
```

如果服务器任务失败并修复了问题，可以**重新执行同一个总控脚本**。缺失缓存和已完成映射可复用，已完整训练的 Adapter 不重训，已写出的完整评测结果不重评；Adapter 训练若在 epoch 中断，可从最后完整 epoch 的 `last_epoch.pt` 恢复。不要删除已有 checkpoint。

注意：计算异常、GPU OOM、磁盘空间不足或服务器重启不由 tmux 自动修复；总控只保证顺序执行、错误停止和可重启的部分流程。

## 结果位置

所有实验新增文件保存在独立目录，不覆盖之前两轮结果：

```text
~/plugguard_repro/checkpoints/cross_adapt_four_directions_v1/
  qwen8_to_qwen14/
  llama_to_qwen14/
  qwen8_to_llama/
  qwen14_to_llama/
  summary_16_results.csv
```

每个方向分别有 `train_on_src_train/affine_best.pt`（B）与 `train_on_tgt_train/affine_best.pt`（E/F），以及 `metrics_{b,c,e,f}.json`、`predictions_{b,c,e,f}.csv`。

字符映射索引保存在：`~/plugguard_repro/data/cross_adapt_char_maps_v1/<方向>/{src_train,src_test,tgt_train,tgt_test}/`。

**本地测试情况**：Python/Shell 语法检查、合成 token offset 映射和 Adapter 维度测试已通过；尚未在你的 L20 服务器执行，不保证远程模型缓存和 checkpoint 名称一定正确，运行前务必先通过 `--mode audit`。
