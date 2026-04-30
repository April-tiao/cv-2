# 智慧食堂业务图像二分类

MobileNetV3-Small 二分类实验，用于判断图片是否属于智慧食堂业务图像。正样本包含食物、水果、图表，以及 `train/` 下补充的图表文件夹；负样本使用公开 `sklearn digits` 数据集生成，并按正样本数量均衡。

## 本次训练结果

```text
train: 2340 positive / 2340 negative
test: 33 positive / 33 negative
model: MobileNetV3-Small
input: 160x160
best test accuracy: 100%
benchmark accuracy over train+test: 99.94%
CPU end-to-end P95 latency: 8.99ms
target latency: P95 <= 50ms
```

训练产物：

```text
artifacts_balanced/best_mobilenetv3_small.pt
artifacts_balanced/dataset_summary.json
artifacts_balanced/train_history.json
artifacts_balanced/latency_summary.json
```

## 主要脚本

```text
canteen_binary_experiment.py
build_cifar10_negatives.py
```

完整命令见 `EXPERIMENT_USAGE.md`。
