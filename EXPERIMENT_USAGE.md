# 新数据集训练使用说明

旧的 `food/fruit/chart/negative`、额外 `neg-dir`、`artifacts_balanced` 等实验结构已废弃。当前只使用新的服务器数据结构：

```text
data/image_dataset/
  train/
    positive/
    negative/
  test/
    positive/
    negative/
```

脚本默认自动查找：

```text
./data/image_dataset
../data/image_dataset
/XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset
```

支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`，跳过 `.gif`。

## 1. 环境

```bash
conda create -n canteen-cv python=3.10 -y
conda activate canteen-cv
pip install -r requirements.txt
```

## 2. 扫描数据

```bash
python canteen_binary_experiment.py scan
```

如果自动路径不匹配：

```bash
python canteen_binary_experiment.py scan --data-dir /XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset
```

## 3. 训练

```bash
python canteen_binary_experiment.py train --epochs 10 --batch-size 32
```

## 4. 评估

```bash
python canteen_binary_experiment.py evaluate --checkpoint outputs/best_model.pt
```

## 5. 预测

```bash
python canteen_binary_experiment.py predict --checkpoint outputs/best_model.pt --image /path/to/image.jpg
```

