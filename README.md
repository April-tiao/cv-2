# 智慧食堂业务图像二分类模型

本仓库保存一个轻量级图像二分类实验，用于判断输入图片是否属于“智慧食堂业务图像”。当前模型使用 `MobileNetV3-Small`，输入尺寸为 `160x160`，目标是在 CPU 场景下保持较高准确率，并满足端到端 P95 延迟不超过 `50ms` 的在线前置识别要求。

## 1. 项目目标

模型输出二分类结果：

```text
positive: 智慧食堂业务相关图片
negative: 非智慧食堂业务图片
```

当前正样本主要包括：

- 食物图片
- 水果图片
- 统计图表图片
- `train/` 下补充的业务图表图片目录

当前负样本主要来自公开数据集生成的非业务图片，用于构造与业务图像无关的对照样本。

核心指标：

```text
accuracy >= 90%
CPU end-to-end P95 latency <= 50ms
```

## 2. 当前结果

本次已提交的最终产物位于 `artifacts_balanced/`。

```text
model: MobileNetV3-Small
input size: 160x160
train samples: 2340 positive / 2340 negative
test samples: 33 positive / 33 negative
best test accuracy: 100%
benchmark samples: 4746
benchmark accuracy: 99.94%
CPU end-to-end P50 latency: 7.75ms
CPU end-to-end P90 latency: 8.54ms
CPU end-to-end P95 latency: 8.99ms
CPU end-to-end P99 latency: 10.77ms
target: P95 <= 50ms
result: PASS
```

说明：当前测试集规模较小，结果能证明当前数据分布下模型可用，但上线前仍建议补充真实线上图片、相似干扰图片、截图、文档、UI 页面等 hard negatives 做泛化验证。

## 3. 仓库结构

```text
.
├── README.md
├── EXPERIMENT_USAGE.md
├── canteen_binary_experiment.py
├── build_cifar10_negatives.py
├── artifacts_balanced/
│   ├── best_mobilenetv3_small.pt
│   ├── dataset_summary.json
│   ├── train_history.json
│   ├── latency_summary.json
│   └── latency_breakdown.csv
└── .gitignore
```

主要文件说明：

| 文件 | 说明 |
|---|---|
| `canteen_binary_experiment.py` | 数据扫描、缓存生成、训练、评估和 benchmark 主脚本 |
| `build_cifar10_negatives.py` | 生成公开负样本数据，默认可使用 `sklearn digits` |
| `artifacts_balanced/best_mobilenetv3_small.pt` | 当前最优 PyTorch 模型权重 |
| `artifacts_balanced/dataset_summary.json` | 训练集和测试集样本统计 |
| `artifacts_balanced/train_history.json` | 每轮训练/验证指标 |
| `artifacts_balanced/latency_summary.json` | 端到端延迟汇总 |
| `artifacts_balanced/latency_breakdown.csv` | 每张图片的读取、解码、预处理、推理和总耗时 |
| `EXPERIMENT_USAGE.md` | 更完整的实验命令记录 |

未提交的大目录：

```text
train/
test/
cache_*/
balanced_cache_*/
balanced_cache_negatives_exact_160/
public_digits_negatives_160/
public_cifar10_negatives_160/
public_datasets/
```

这些目录包含原始图片、缓存图片或可重新生成的数据，已通过 `.gitignore` 排除，避免仓库体积过大。

## 4. 环境依赖

建议使用 Python 3.10+。核心依赖包括：

```text
torch
torchvision
opencv-python
numpy
pillow
scikit-learn
```

如果在当前 Codex 桌面环境中运行，可以使用 bundled Python：

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" canteen_binary_experiment.py --help
```

## 5. 数据约定

推荐数据目录：

```text
train/
  food/
  fruit/
  chart/
  negative/
test/
  food/
  fruit/
  chart/
  negative/
```

默认正样本目录名：

```text
food, fruit, chart
```

默认负样本目录名：

```text
neg, negative, negatives, non_canteen, non-canteen, background, public, public_negative, 0
```

脚本对 `train/` 下未知目录有一个实验性规则：默认把未知训练目录当作正样本。这是为了方便把新增业务图表目录快速纳入训练。正式数据整理时，建议明确使用 `food/`、`fruit/`、`chart/`、`negative/` 等目录名，减少误标风险。

## 6. 实验复现

### 6.1 扫描数据

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" canteen_binary_experiment.py scan --inspect-limit 200
```

### 6.2 生成公开负样本

默认使用 `sklearn digits` 构造非业务负样本，不依赖外网下载：

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" build_cifar10_negatives.py --source sklearn-digits --out public_digits_negatives_160 --overwrite
```

如果网络可用，也可以尝试 CIFAR-10：

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" build_cifar10_negatives.py --source cifar10 --out public_cifar10_negatives_160 --overwrite
```

### 6.3 生成 160x160 缓存图

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" canteen_binary_experiment.py --neg-dir public_digits_negatives_160 prepare-cache --cache-dir balanced_cache_160 --overwrite
```

缓存图用于减少原图读取、解码和 resize 对端到端延迟的影响。本次实验里，原始图片的读取和预处理是主要耗时来源，模型推理本身不是瓶颈。

### 6.4 按缓存后正样本数量重建负样本

如果缓存阶段跳过了坏图，需要按缓存后的可用正样本数量重新生成严格均衡的负样本：

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" build_cifar10_negatives.py --data balanced_cache_160 --source sklearn-digits --out balanced_cache_negatives_exact_160 --overwrite
```

### 6.5 训练 MobileNetV3-Small

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" canteen_binary_experiment.py --data balanced_cache_160 --neg-dir balanced_cache_negatives_exact_160 --out artifacts_balanced --image-size 160 train --epochs 12 --batch-size 32 --workers 0 --lr 0.001
```

输出：

```text
artifacts_balanced/best_mobilenetv3_small.pt
artifacts_balanced/dataset_summary.json
artifacts_balanced/train_history.json
```

### 6.6 Benchmark

```powershell
& "C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" canteen_binary_experiment.py --data balanced_cache_160 --neg-dir balanced_cache_negatives_exact_160 --out artifacts_balanced --image-size 160 --cpu benchmark --checkpoint artifacts_balanced\best_mobilenetv3_small.pt --target-ms 50
```

输出：

```text
artifacts_balanced/latency_summary.json
artifacts_balanced/latency_breakdown.csv
```

`latency_breakdown.csv` 会记录每张图片的分段耗时：

```text
read_ms, decode_ms, preprocess_ms, inference_ms, total_ms
```

## 7. 模型选择说明

| 模块 | 当前选择 | 原因 |
|---|---|---|
| 模型 | MobileNetV3-Small | 轻量、CPU 推理快，适合在线前置识别 |
| 输入尺寸 | 160x160 | 在准确率和延迟之间折中 |
| 训练框架 | PyTorch + TorchVision | 实现简单，模型和数据增强支持成熟 |
| 负样本 | sklearn digits / CIFAR-10 | 快速构造非业务图片对照 |
| 数据均衡 | 正负样本 1:1 | 降低类别偏置 |
| 延迟指标 | 端到端 P95 | 覆盖读取、解码、预处理和推理全过程 |

## 8. 已知风险

- 当前测试集只有 `66` 张图片，泛化能力还需要更大规模独立测试集验证。
- 当前负样本相对简单，后续应补充真实业务中的误入图片、菜单截图、网页截图、报表、文档、UI 页面、餐厅相似场景等 hard negatives。
- 当前提交的是 PyTorch `.pt` 权重，生产部署时可继续导出 ONNX，并使用 OpenCV 或 ONNX Runtime 做更稳定的端到端 benchmark。
- 本次 P95 延迟基于缓存后的 `160x160` 图片，直接处理超大原图时可能明显变慢。

## 9. 变更记录

以后每次修改代码、模型、数据处理流程、训练参数、文档或部署方式，都需要在这里追加记录。

| 日期 | 修改内容 | 影响 |
|---|---|---|
| 2026-04-30 | 首次整理并推送图像分类实验到 `April-tiao/cv-2`。包含训练脚本、负样本生成脚本、最终模型权重和 benchmark 结果。 | 仓库具备可追踪的模型产物和复现实验入口。 |
| 2026-04-30 | 重写 README，补充项目目标、目录结构、数据约定、复现实验命令、指标结果、风险说明和变更记录。 | 文档从简略说明升级为完整项目说明，后续修改可以持续记录。 |
