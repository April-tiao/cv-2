# 智慧食堂图像三分类

本项目使用 MobileNetV3-Small 做图像三分类，用于判断图片属于：

| 标签 | 类别 |
|---:|---|
| 0 | 其他图片 |
| 1 | 食物图片 |
| 2 | 图表图片 |

## 数据结构

默认数据目录名为 `image_dataset_3class`：

```text
image_dataset_3class/
  train/
    0_other/
    1_food/
    2_chart/
  test/
    0_other/
    1_food/
    2_chart/
```

脚本会自动查找：

```text
./data/image_dataset_3class
../data/image_dataset_3class
./image_dataset_3class
../image_dataset_3class
/XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset_3class
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`，跳过 `.gif`。

## 安装

```bash
conda create -n canteen-cv python=3.10 -y
conda activate canteen-cv
pip install -r requirements.txt
```

如需指定 CUDA 版本，请先安装服务器匹配的 `torch` / `torchvision`，再安装其余依赖。

## 使用

检查数据：

```bash
python canteen_3class_experiment.py scan
```

训练：

```bash
python canteen_3class_experiment.py train --epochs 5 --batch-size 64 --image-size 160
```

评估：

```bash
python canteen_3class_experiment.py evaluate --checkpoint outputs_3class/best_model.pt
```

测速：

```bash
python canteen_3class_experiment.py latency --checkpoint outputs_3class/best_model.pt
```

显式指定数据路径：

```bash
python canteen_3class_experiment.py scan --data-dir /path/to/image_dataset_3class
```

## 本地实验结果

数据集：3501 张，三类均衡，每类 1167 张。  
训练集：2799 张；测试集：702 张。

MobileNetV3-Small，ImageNet 预训练，输入 `160x160`，训练 5 epochs。

| 指标 | 结果 |
|---|---:|
| Accuracy | 98.15% |
| Macro Precision | 98.16% |
| Macro Recall | 98.15% |
| Macro F1 | 98.15% |

CPU 单张推理时延：

| 指标 | 时延 |
|---|---:|
| Mean | 8.92 ms |
| P50 | 8.82 ms |
| P95 | 11.23 ms |
| P99 | 13.90 ms |

