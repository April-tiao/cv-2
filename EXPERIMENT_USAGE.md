# 智慧食堂业务图像二分类实验

当前脚本：`canteen_binary_experiment.py`

## 1. 扫描数据

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' canteen_binary_experiment.py scan --inspect-limit 200
```

默认正样本目录名：

```text
food, fruit, chart
```

默认负样本目录名：

```text
neg, negative, negatives, non_canteen, non-canteen, background, public, public_negative, 0
```

推荐目录：

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

也可以把公开负样本单独放在任意目录，然后训练时加：

```powershell
--neg-dir C:\path\to\public_negative
```

## 2. 生成公开负样本

默认用 `sklearn digits` 公开数据集作为非食堂业务负样本，并按当前可用正样本数量自动均衡。

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' build_cifar10_negatives.py --source sklearn-digits --out public_digits_negatives_160 --overwrite
```

如果网络可用，也可以尝试 CIFAR-10：

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' build_cifar10_negatives.py --source cifar10 --out public_cifar10_negatives_160 --overwrite
```

## 3. 生成 160x160 小图缓存

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' canteen_binary_experiment.py --neg-dir public_digits_negatives_160 prepare-cache --cache-dir balanced_cache_160 --overwrite
```

这个步骤用于验证和优化端到端时延。当前原图最大超过 4000 像素、3MB，本地读图和解码会拖慢 P95。

如果缓存时有坏图被跳过，按缓存后的可用正样本数重新生成严格均衡负样本：

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' build_cifar10_negatives.py --data balanced_cache_160 --source sklearn-digits --out balanced_cache_negatives_exact_160 --overwrite
```

## 4. 训练 MobileNetV3-Small

优化训练命令：

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' canteen_binary_experiment.py --data balanced_cache_160 --neg-dir balanced_cache_negatives_exact_160 --out artifacts_balanced --image-size 160 train --epochs 12 --batch-size 32 --workers 0 --lr 0.001
```

输出：

```text
artifacts_balanced/best_mobilenetv3_small.pt
artifacts_balanced/dataset_summary.json
artifacts_balanced/train_history.json
```

本次结果：

```text
train: 2340 positive / 2340 negative
test: 33 positive / 33 negative
best test accuracy: 100%
```

## 5. 端到端延迟剖析

```powershell
& 'C:\Users\MyPC\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' canteen_binary_experiment.py --data balanced_cache_160 --neg-dir balanced_cache_negatives_exact_160 --out artifacts_balanced --image-size 160 --cpu benchmark --checkpoint artifacts_balanced\best_mobilenetv3_small.pt --target-ms 50
```

输出：

```text
artifacts_balanced/latency_summary.json
artifacts_balanced/latency_breakdown.csv
```

CSV 会记录每张图：

```text
read_ms, decode_ms, preprocess_ms, inference_ms, total_ms
```

重点看 `p95_total_ms <= 50`。

本次 CPU 端到端结果：

```text
accuracy: 99.94%
p50_total_ms: 7.75
p90_total_ms: 8.54
p95_total_ms: 8.99
p99_total_ms: 10.77
target: P95 <= 50ms
result: pass
```
