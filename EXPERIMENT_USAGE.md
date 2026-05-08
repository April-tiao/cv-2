# 三分类训练使用说明

本仓库只保留三分类方案：

```text
0_other: 其他图片
1_food: 食物图片
2_chart: 图表图片
```

## 数据目录

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

## 命令

```bash
pip install -r requirements.txt
python canteen_3class_experiment.py scan
python canteen_3class_experiment.py train --epochs 5 --batch-size 64 --image-size 160
python canteen_3class_experiment.py evaluate --checkpoint outputs_3class/best_model.pt
python canteen_3class_experiment.py latency --checkpoint outputs_3class/best_model.pt
```

