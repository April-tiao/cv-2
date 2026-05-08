import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


LABEL_TO_INDEX = {"0_other": 0, "1_food": 1, "2_chart": 2}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_DATA_CANDIDATES = [
    Path("data/image_dataset_3class"),
    Path("../data/image_dataset_3class"),
    Path("image_dataset_3class"),
    Path("../image_dataset_3class"),
    Path("/XYAIFS00/HOME/pushi_yjliang/pushi_yjliang_1/HDD_POOL/mx/data/image_dataset_3class"),
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_dir(data_dir):
    candidates = [Path(data_dir)] if data_dir else DEFAULT_DATA_CANDIDATES
    for candidate in candidates:
        path = candidate.resolve()
        if (path / "train").exists() and (path / "test").exists():
            return path
    raise FileNotFoundError("Cannot find image_dataset_3class")


def get_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_transforms(image_size):
    train_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


class ImageDataset3(Dataset):
    def __init__(self, root: Path, transform):
        self.root = root
        self.transform = transform
        actual = {p.name for p in root.iterdir() if p.is_dir()} if root.exists() else set()
        if actual != set(LABEL_TO_INDEX):
            raise ValueError(f"{root} class dirs must be {sorted(LABEL_TO_INDEX)}, got {sorted(actual)}")
        self.samples: List[Tuple[str, int]] = []
        for label_name, label in LABEL_TO_INDEX.items():
            for path in sorted((root / label_name).rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((str(path), label))
        if not self.samples:
            raise ValueError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img.load()
            return self.transform(img), label


def make_loaders(data_dir, image_size, batch_size, workers):
    train_tf, eval_tf = build_transforms(image_size)
    train_ds = ImageDataset3(data_dir / "train", train_tf)
    test_ds = ImageDataset3(data_dir / "test", eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())
    return train_loader, test_loader


def build_model(pretrained=True):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 3)
    return model


def scan(data_dir):
    counts = {"train": {}, "test": {}}
    for split in ["train", "test"]:
        for label_name in LABEL_TO_INDEX:
            counts[split][label_name] = len([
                p for p in (data_dir / split / label_name).iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            ])
    counts["total"] = {
        k: counts["train"][k] + counts["test"][k]
        for k in LABEL_TO_INDEX
    }
    return counts


def metrics_from_predictions(loss_sum, y_true, y_pred):
    n = len(y_true)
    cm = [[0, 0, 0] for _ in range(3)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    per_class = {}
    f1s = []
    precisions = []
    recalls = []
    for cls in range(3):
        tp = cm[cls][cls]
        fp = sum(cm[r][cls] for r in range(3) if r != cls)
        fn = sum(cm[cls][c] for c in range(3) if c != cls)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[INDEX_TO_LABEL[cls]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(cm[cls]),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    correct = sum(cm[i][i] for i in range(3))
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": correct / max(n, 1),
        "macro_precision": sum(precisions) / 3,
        "macro_recall": sum(recalls) / 3,
        "macro_f1": sum(f1s) / 3,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    y_true, y_pred = [], []
    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return metrics_from_predictions(loss_sum, y_true, y_pred)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = 0.0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)
    return loss_sum / max(total, 1)


def save_checkpoint(path, model, args, metrics, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_name": "mobilenet_v3_small",
        "num_classes": 3,
        "image_size": args.image_size,
        "epoch": epoch,
        "metrics": metrics,
        "label_to_index": LABEL_TO_INDEX,
    }, path)


def command_scan(args):
    data_dir = resolve_data_dir(args.data_dir)
    print(json.dumps({"data_dir": str(data_dir), **scan(data_dir)}, ensure_ascii=False, indent=2))


def command_train(args):
    set_seed(args.seed)
    data_dir = resolve_data_dir(args.data_dir)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    train_loader, test_loader = make_loaders(data_dir, args.image_size, args.batch_size, args.workers)
    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    history = {
        "data_dir": str(data_dir),
        "device": str(device),
        "args": {key: value for key, value in vars(args).items() if key != "func"},
        "epochs": [],
    }
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "test": metrics, "seconds": round(time.time() - start, 2)}
        history["epochs"].append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        save_checkpoint(output_dir / "last_model.pt", model, args, metrics, epoch)
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            save_checkpoint(output_dir / "best_model.pt", model, args, metrics, epoch)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_macro_f1={best_macro_f1:.4f}")


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def command_evaluate(args):
    data_dir = resolve_data_dir(args.data_dir)
    device = get_device(args.device)
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)
    _, test_loader = make_loaders(data_dir, args.image_size or ckpt["image_size"], args.batch_size, args.workers)
    metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


@torch.no_grad()
def command_latency(args):
    device = get_device(args.device)
    model, ckpt = load_checkpoint(Path(args.checkpoint), device)
    image_size = args.image_size or ckpt["image_size"]
    _, tf = build_transforms(image_size)
    paths = []
    for label in LABEL_TO_INDEX:
        paths.extend(sorted((resolve_data_dir(args.data_dir) / "test" / label).glob("*"))[: args.samples // 3 + 1])
    paths = [p for p in paths if p.suffix.lower() in IMAGE_EXTS][: args.samples]
    tensors = []
    for p in paths:
        with Image.open(p) as img:
            tensors.append(tf(img).unsqueeze(0).to(device))
    for t in tensors[:20]:
        model(t)
    latencies = []
    for t in tensors:
        start = time.perf_counter()
        model(t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    latencies.sort()
    def pct(q):
        return latencies[min(int(len(latencies) * q), len(latencies) - 1)]
    summary = {
        "device": str(device),
        "samples": len(latencies),
        "image_size": image_size,
        "mean_ms": sum(latencies) / len(latencies),
        "p50_ms": pct(0.50),
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("scan").set_defaults(func=command_scan)
    train = sub.add_parser("train")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--image-size", type=int, default=160)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=20260507)
    train.add_argument("--device", default="auto")
    train.add_argument("--output-dir", default="outputs_3class")
    train.add_argument("--no-pretrained", action="store_true")
    train.set_defaults(func=command_train)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--checkpoint", default="outputs_3class/best_model.pt")
    ev.add_argument("--batch-size", type=int, default=64)
    ev.add_argument("--image-size", type=int, default=None)
    ev.add_argument("--workers", type=int, default=0)
    ev.add_argument("--device", default="auto")
    ev.set_defaults(func=command_evaluate)
    lat = sub.add_parser("latency")
    lat.add_argument("--checkpoint", default="outputs_3class/best_model.pt")
    lat.add_argument("--samples", type=int, default=120)
    lat.add_argument("--image-size", type=int, default=None)
    lat.add_argument("--device", default="auto")
    lat.set_defaults(func=command_latency)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
