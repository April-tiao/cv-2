# 智慧食堂业务图片二分类

本仓库用于训练“图片是否属于智慧食堂业务”的二分类模型。新的数据结构已经覆盖旧实验结构，训练代码默认读取服务器 `data/image_dataset`。

## 标签定义

```text
positive: 属于智慧食堂业务图片
negative: 不属于智慧食堂业务图片
```

正样本包括：

- 食物类图片
- 图表类图片
- 智慧食堂设备、食堂场景、留样/结算/消费机等业务图片

负样本包括：

- 不属于智慧食堂业务的复杂自然图像、物体图像、人物、交通、建筑等图片

## 数据目录

服务器数据目录：

```text
/XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/
```

图片数据结构：

```text
data/image_dataset/
  train/
    positive/
    negative/
  test/
    positive/
    negative/
```

训练脚本会自动查找：

```text
./data/image_dataset
../data/image_dataset
/XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset
```

支持格式：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`。

`.gif` 会直接跳过，不参与训练和测试统计。

## 安装依赖

```bash
conda create -n canteen-cv python=3.10 -y
conda activate canteen-cv
pip install -r requirements.txt
```

如果服务器需要指定 CUDA 版本，请先按服务器 CUDA 版本安装对应的 `torch` 和 `torchvision`，再执行：

```bash
pip install pillow tqdm numpy
```

## 检查数据

```bash
python canteen_binary_experiment.py scan
```

显式指定数据路径：

```bash
python canteen_binary_experiment.py scan --data-dir /XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset
```

## 训练

```bash
python canteen_binary_experiment.py train --epochs 10 --batch-size 32
```

常用参数：

```bash
python canteen_binary_experiment.py train \
  --model mobilenet_v3_small \
  --epochs 10 \
  --batch-size 32 \
  --image-size 224 \
  --workers 4 \
  --lr 3e-4 \
  --output-dir outputs
```

默认输出：

```text
outputs/best_model.pt
outputs/last_model.pt
outputs/metrics.json
```

## 测试集评估

```bash
python canteen_binary_experiment.py evaluate --checkpoint outputs/best_model.pt
```

## 单张图片预测

```bash
python canteen_binary_experiment.py predict --checkpoint outputs/best_model.pt --image /path/to/image.jpg
```

## 推理测速

```bash
python canteen_binary_experiment.py benchmark --model mobilenet_v3_small --batch-size 32
```

