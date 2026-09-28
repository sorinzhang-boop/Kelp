# PlugGuard Reproduction：环境迁移与快速启动

本文件用于在更换 GPU 服务器后，快速恢复当前 PlugGuard / Kelp 复现环境并跑通代码。

- 原仓库：`Alibaba-AAIG/Kelp`
- 当前复现仓库：`sorinzhang-boop/Kelp`
- 主要实验结果：`notes/reproduction.md`
- 环境版本：`notes/environment.md`
- 完整依赖：`notes/requirements_repro.txt`

---

## 1. 推荐目录

```text
<WORKSPACE>/
├── Kelp/
├── .venv/
├── hf_cache/
├── data/
├── checkpoints/
├── logs/
└── diagnostics/
```

旧服务器工作目录为：

```text
/data1/plugguard_repro/
```

新服务器可以使用其他路径，但需要检查代码中的旧绝对路径。

---

## 2. 克隆仓库并创建环境

```bash
mkdir -p <WORKSPACE>
cd <WORKSPACE>

git clone https://github.com/sorinzhang-boop/Kelp.git
cd Kelp

python3.12 -m venv ../.venv
source ../.venv/bin/activate

python -m pip install --upgrade pip
pip install -r notes/requirements_repro.txt
```

当前主要环境：

| Package | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.7.1+cu128 |
| Transformers | 4.56.2 |
| datasets | 5.0.1 |
| NumPy | 2.2.6 |
| scikit-learn | 1.9.1 |
| accelerate | 1.15.0 |

验证：

```bash
python - <<'PY'
import torch
import transformers
import datasets

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

再执行：

```bash
nvidia-smi
```

---

## 3. Hugging Face 模型

建议统一设置：

```bash
export HF_HOME=<WORKSPACE>/hf_cache
```

当前主要模型配置在 `config.py`：

```text
Qwen/Qwen3-8B
Qwen/Qwen3-14B
meta-llama/Llama-3.1-8B-Instruct
```

如需离线使用：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

---

## 4. 数据集

当前 `dataset.py` 使用：

```python
datasets.load_from_disk(dataset_dir)
```

因此：

```text
--train_dataset_dir
--test_dataset_dir
```

必须指向已经通过 Hugging Face Datasets `save_to_disk()` 保存的本地目录，而不是普通 JSON / CSV。

训练命令基本形式：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --train_dataset_dir <TRAIN_DATASET_DIR> \
  --test_dataset_dir <TEST_DATASET_DIR> \
  --save_dir <CHECKPOINT_DIR>
```

当前 Git 仓库不包含完整训练数据，因此服务器释放前应单独备份 `data/`，或补充可重复的数据构建脚本。

---

## 5. 当前训练配置

主要配置位于 `config.py`：

```python
TRAIN_CONFIG = {
    "lr": 5e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.05,
    "num_train_epochs": 1,
    "batch_size": 1,
    "gradient_acc_steps": 32,
    "max_length": 4096,
    "num_supervised_token": 10,
    "seed": 42,
}
```

Hidden layer：

```text
Qwen3-8B               Layer 20
Qwen3-14B              Layer 20
Llama-3.1-8B-Instruct  Layer 20
```

Llama + WildGuard 的 learning-rate 异常诊断见：

```text
notes/llama_wildguard_anomaly_analysis.md
```

---

## 6. 必须修改或检查的路径

### Table 2 本地模型路径

`benchmark_table2_base.py` 当前写死了旧服务器上的 Qwen3-8B snapshot：

```text
/data1/plugguard_repro/hf_cache/...
```

换服务器后必须修改。

先查新服务器上的 snapshot：

```bash
find <WORKSPACE>/hf_cache \
  -type d \
  -path "*models--Qwen--Qwen3-8B/snapshots/*"
```

然后修改 `benchmark_table2_base.py` 中的：

```python
MODEL_NAME = "..."
```

`benchmark_table2_sequential.py` 和 `benchmark_table2_parallel.py` 会复用该 `MODEL_NAME`。

### 训练数据与 checkpoint

以下路径不需要改源码，直接通过命令行传入：

```text
--train_dataset_dir
--test_dataset_dir
--save_dir
```

---

## 7. Hidden-state cache

`SafetyDataset` 会在 dataset 目录下创建：

```text
safety_cache/<model>/idx<layer>_maxlength<length>/
```

其中保存逐样本：

```text
sample_XXXXXXXX.pt
```

如果 cache 已存在，可以跳过耗时的 hidden-state 提取。

因此服务器释放前：

- 想快速恢复：备份 `safety_cache`
- 磁盘有限：可以不备份，之后重新生成
- `checkpoint` 和 `safety_cache` 不要混淆

---

## 8. Table 2 时延实验

当前脚本：

```text
benchmark_table2_base.py
benchmark_table2_sequential.py
benchmark_table2_parallel.py
benchmark_table2_8gpu.py
```

先检查输入是否仍为 1000 tokens：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_table2_base.py --check-only
```

应看到：

```text
chat-templated input tokens : 1000
paper target input tokens   : 1000
match                       : True
```

再做 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark_table2_base.py --runs 3
CUDA_VISIBLE_DEVICES=0 python benchmark_table2_sequential.py --runs 3
CUDA_VISIBLE_DEVICES=0 python benchmark_table2_parallel.py --runs 3
```

三条路径正常后，再运行正式 8 卡实验：

```bash
nohup python -u benchmark_table2_8gpu.py --launch \
  > table2_8gpu_master.log 2>&1 &
```

说明：

- Sequential 基于公开 latency demo 和当前 `models.py` 做兼容；
- Parallel 公开仓库未提供完整实现，当前版本按论文描述使用 `torch.cuda.Stream` 重建。

---

## 9. 新服务器推荐检查顺序

```text
1. nvidia-smi
2. 激活 .venv
3. 验证 torch / transformers / datasets
4. 检查 torch.cuda.is_available()
5. 检查 config.py
6. 检查模型是否能加载
7. 检查 dataset 是否能 load_from_disk()
8. 检查旧绝对路径
9. 跑 smoke test
10. 再启动正式训练或评测
```

---

## 10. 服务器释放前必须备份

GitHub 已保存：

```text
代码
config.py
notes/
部分实验日志
Table 2 脚本
```

GitHub 不保存：

```text
模型权重
本地数据集
hidden-state safety_cache
训练 checkpoint
diagnostics 输出
```

建议至少备份：

```text
<WORKSPACE>/data/
<WORKSPACE>/checkpoints/
必要的 safety_cache/
必要的 diagnostics/
```

如果模型后续下载困难，再额外备份：

```text
<WORKSPACE>/hf_cache/
```

---

## 11. 文档索引

| 文件 | 用途 |
|---|---|
| `notes/reproduction.md` | 主要复现实验与结果 |
| `notes/environment.md` | 当前服务器与软件环境 |
| `notes/requirements_repro.txt` | 完整 Python 依赖 |
| `notes/implementation_audit.md` | 论文与公开代码实现差异 |
| `notes/llama_wildguard_anomaly_analysis.md` | Llama + WildGuard 异常诊断 |

重新开始该项目时，建议先阅读本文件，再进入对应实验记录。
