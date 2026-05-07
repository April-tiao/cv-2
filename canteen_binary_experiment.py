import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


LABEL_TO_INDEX = {"negative": 0, "positive": 1}
INDEX_TO_LABEL = {0: "negative", 1: "positive"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATA_CANDIDATES = [
    Path("data/image_dataset"),
    Path("../data/image_dataset"),
    Path("/XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset"),
]


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    tn: int
    fp: int
    fn: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_data_dir(data_dir: Optional[str]) -> Path:
    data_dir = data_dir or os.environ.get("CANTEEN_DATA_DIR")
    candidates = [Path(data_dir)] if data_dir else DEFAULT_DATA_CANDIDATES
    for candidate in candidates:
        path = candidate.expanduser().resolve()
        if (path / "train").exists() and (path / "test").exists():
            return path
    tried = "\n".join(f"- {p.expanduser().resolve()}" for p in candidates)
    raise FileNotFoundError(f"没有找到 image_dataset。已尝试：\n{tried}")


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


class CanteenImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose):
        self.root = root
        self.transform = transform
        expected = set(LABEL_TO_INDEX)
        actual = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
        if actual != expected:
            raise ValueError(f"{root} 下类别目录应为 {sorted(expected)}，实际为 {sorted(actual)}")
        self.samples: List[Tuple[str, int]] = []
        for label_name, label_index in LABEL_TO_INDEX.items():
            label_dir = root / label_name
            for path in sorted(label_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((str(path), label_index))
        if not self.samples:
            raise ValueError(f"{root} 中没有可用图片")
        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image.load()
            tensor = self.transform(image)
        return tensor, label


def make_dataset(root: Path, transform: transforms.Compose) -> CanteenImageDataset:
    return CanteenImageDataset(root, transform)


def make_loaders(data_dir: Path, image_size: int, batch_size: int, workers: int) -> Tuple[DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = make_dataset(data_dir / "train", train_tf)
    test_ds = make_dataset(data_dir / "test", eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, test_loader


def build_model(model_name: str, pretrained: bool) -> nn.Module:
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, 2)
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        raise ValueError(f"不支持的模型：{model_name}")
    return model


def count_by_class(dataset: CanteenImageDataset) -> Dict[str, int]:
    counts = {label: 0 for label in LABEL_TO_INDEX}
    for _, target in dataset.samples:
        counts[INDEX_TO_LABEL[target]] += 1
    return counts


def scan_dataset(data_dir: Path) -> Dict[str, Dict[str, int]]:
    tf = transforms.Compose([transforms.Lambda(lambda img: img.convert("RGB")), transforms.ToTensor()])
    train_ds = make_dataset(data_dir / "train", tf)
    test_ds = make_dataset(data_dir / "test", tf)
    return {
        "train": count_by_class(train_ds),
        "test": count_by_class(test_ds),
        "total": {
            label: count_by_class(train_ds)[label] + count_by_class(test_ds)[label]
            for label in LABEL_TO_INDEX
        },
    }


def compute_metrics(loss_sum: float, total: int, y_true: List[int], y_pred: List[int]) -> Metrics:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return Metrics(loss_sum / max(total, 1), accuracy, precision, recall, f1, tp, tn, fp, fn)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Metrics:
    model.eval()
    loss_sum = 0.0
    y_true: List[int] = []
    y_pred: List[int] = []
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return compute_metrics(loss_sum, len(y_true), y_true, y_pred)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_sum = 0.0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
    return loss_sum / max(total, 1)


def save_checkpoint(path: Path, model: nn.Module, args: argparse.Namespace, metrics: Metrics, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_name": args.model,
        "image_size": args.image_size,
        "epoch": epoch,
        "metrics": asdict(metrics),
        "label_to_index": LABEL_TO_INDEX,
    }, path)


def load_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, Dict]:
    checkpoint = torch.load(path, map_location=device)
    model = build_model(checkpoint.get("model_name", "mobilenet_v3_small"), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model, checkpoint


def command_scan(args: argparse.Namespace) -> None:
    data_dir = resolve_data_dir(args.data_dir)
    summary = scan_dataset(data_dir)
    print(json.dumps({"data_dir": str(data_dir), **summary}, ensure_ascii=False, indent=2))


def command_train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    train_loader, test_loader = make_loaders(data_dir, args.image_size, args.batch_size, args.workers)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = {
        "data_dir": str(data_dir),
        "device": str(device),
        "args": vars(args),
        "epochs": [],
    }
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate_model(model, test_loader, criterion, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test": asdict(metrics),
            "seconds": round(time.time() - started, 2),
        }
        history["epochs"].append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        save_checkpoint(output_dir / "last_model.pt", model, args, metrics, epoch)
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            save_checkpoint(output_dir / "best_model.pt", model, args, metrics, epoch)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_f1={best_f1:.4f}")
    print(f"saved={output_dir.resolve()}")


def command_evaluate(args: argparse.Namespace) -> None:
    data_dir = resolve_data_dir(args.data_dir)
    device = get_device(args.device)
    checkpoint = Path(args.checkpoint)
    model, ckpt = load_checkpoint(checkpoint, device)
    image_size = args.image_size or ckpt.get("image_size", 224)
    _, test_loader = make_loaders(data_dir, image_size, args.batch_size, args.workers)
    metrics = evaluate_model(model, test_loader, nn.CrossEntropyLoss(), device)
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))


@torch.no_grad()
def command_predict(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)
    image_size = args.image_size or ckpt.get("image_size", 224)
    _, eval_tf = build_transforms(image_size)
    image = Image.open(args.image)
    tensor = eval_tf(image).unsqueeze(0).to(device)
    logits = model(tensor)
    prob = torch.softmax(logits, dim=1)[0]
    pred = int(prob.argmax().item())
    print(json.dumps({
        "image": args.image,
        "label": pred,
        "label_name": INDEX_TO_LABEL[pred],
        "positive_probability": float(prob[1].item()),
        "negative_probability": float(prob[0].item()),
    }, ensure_ascii=False, indent=2))


@torch.no_grad()
def command_benchmark(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model = build_model(args.model, pretrained=False).to(device).eval()
    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    for _ in range(args.warmup):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.time()
    for _ in range(args.repeats):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - started
    images = args.batch_size * args.repeats
    print(json.dumps({
        "device": str(device),
        "model": args.model,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "images": images,
        "seconds": seconds,
        "images_per_second": images / max(seconds, 1e-12),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    data_parent = argparse.ArgumentParser(add_help=False)
    data_parent.add_argument("--data-dir", default=None, help="image_dataset 路径；也可设置 CANTEEN_DATA_DIR")
    parser = argparse.ArgumentParser(description="智慧食堂业务图片二分类训练与评估", parents=[data_parent])
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="检查数据集数量与类别", parents=[data_parent])
    scan.set_defaults(func=command_scan)

    train = subparsers.add_parser("train", help="训练模型", parents=[data_parent])
    train.add_argument("--model", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=20260507)
    train.add_argument("--device", default="auto")
    train.add_argument("--output-dir", default="outputs")
    train.add_argument("--no-pretrained", action="store_true", help="不加载 ImageNet 预训练权重")
    train.set_defaults(func=command_train)

    evaluate = subparsers.add_parser("evaluate", help="评估测试集", parents=[data_parent])
    evaluate.add_argument("--checkpoint", default="outputs/best_model.pt")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--image-size", type=int, default=None)
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=command_evaluate)

    predict = subparsers.add_parser("predict", help="预测单张图片", parents=[data_parent])
    predict.add_argument("--checkpoint", default="outputs/best_model.pt")
    predict.add_argument("--image", required=True)
    predict.add_argument("--image-size", type=int, default=None)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(func=command_predict)

    benchmark = subparsers.add_parser("benchmark", help="模型前向推理测速", parents=[data_parent])
    benchmark.add_argument("--model", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    benchmark.add_argument("--batch-size", type=int, default=32)
    benchmark.add_argument("--image-size", type=int, default=224)
    benchmark.add_argument("--warmup", type=int, default=10)
    benchmark.add_argument("--repeats", type=int, default=50)
    benchmark.add_argument("--device", default="auto")
    benchmark.set_defaults(func=command_benchmark)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
