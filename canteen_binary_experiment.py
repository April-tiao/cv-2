import argparse
import csv
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_POSITIVE_NAMES = {"food", "fruit", "chart"}
DEFAULT_NEGATIVE_NAMES = {
    "neg",
    "negative",
    "negatives",
    "non_canteen",
    "non-canteen",
    "background",
    "public",
    "public_negative",
    "0",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source: str


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def parse_name_set(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


def collect_samples(
    root: Path,
    split: str,
    positive_names: set[str],
    negative_names: set[str],
    neg_dirs: list[Path],
    unknown_train_positive: bool = True,
) -> list[Sample]:
    split_root = root / split
    samples: list[Sample] = []

    if split_root.exists():
        for class_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
            name = class_dir.name.lower()
            if name in positive_names:
                label = 1
            elif name in negative_names:
                label = 0
            elif split == "train" and unknown_train_positive:
                label = 1
            else:
                continue
            samples.extend(Sample(p, label, f"{split}/{class_dir.name}") for p in class_dir.rglob("*") if is_image(p))

    for neg_dir in neg_dirs:
        candidate = neg_dir / split if (neg_dir / split).exists() else neg_dir
        if candidate.exists():
            samples.extend(Sample(p, 0, str(candidate)) for p in candidate.rglob("*") if is_image(p))

    return sorted(samples, key=lambda s: str(s.path).lower())


def summarize_samples(samples: list[Sample]) -> dict:
    by_label = {"positive": 0, "negative": 0}
    by_source: dict[str, int] = {}
    for sample in samples:
        by_label["positive" if sample.label == 1 else "negative"] += 1
        by_source[sample.source] = by_source.get(sample.source, 0) + 1
    return {"total": len(samples), "by_label": by_label, "by_source": by_source}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, p))


def inspect_images(samples: list[Sample], limit: int | None = None) -> dict:
    rows = samples if limit is None else samples[:limit]
    widths: list[int] = []
    heights: list[int] = []
    sizes: list[int] = []
    bad: list[str] = []

    for sample in rows:
        try:
            with Image.open(sample.path) as img:
                widths.append(img.width)
                heights.append(img.height)
            sizes.append(sample.path.stat().st_size)
        except Exception as exc:
            bad.append(f"{sample.path}: {exc}")

    return {
        "checked": len(rows),
        "bad_count": len(bad),
        "bad_examples": bad[:10],
        "width": {"min": min(widths, default=0), "p50": percentile(widths, 50), "p95": percentile(widths, 95), "max": max(widths, default=0)},
        "height": {"min": min(heights, default=0), "p50": percentile(heights, 50), "p95": percentile(heights, 95), "max": max(heights, default=0)},
        "file_kb": {
            "min": round(min(sizes, default=0) / 1024, 2),
            "p50": round(percentile(sizes, 50) / 1024, 2),
            "p95": round(percentile(sizes, 95) / 1024, 2),
            "max": round(max(sizes, default=0) / 1024, 2),
        },
    }


class CanteenDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, train: bool):
        self.samples = samples
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(image_size + 24),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
        return self.transform(image), torch.tensor(sample.label, dtype=torch.long), str(sample.path)


def build_model(pretrained: bool) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
        except Exception:
            weights = None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def make_loader(samples: list[Sample], image_size: int, batch_size: int, train: bool, workers: int) -> DataLoader:
    dataset = CanteenDataset(samples, image_size=image_size, train=train)
    sampler = None
    shuffle = train
    if train:
        counts = {0: 0, 1: 0}
        for sample in samples:
            counts[sample.label] += 1
        weights = [1.0 / max(counts[sample.label], 1) for sample in samples]
        sampler = WeightedRandomSampler(weights, num_samples=len(samples), replacement=True)
        shuffle = False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=workers)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    tp = tn = fp = fn = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= threshold).long()
            total += labels.numel()
            correct += (preds == labels).sum().item()
            loss_sum += loss.item() * labels.numel()
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "loss": loss_sum / max(total, 1),
        "accuracy": correct / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def train(args: argparse.Namespace) -> None:
    root = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_names = parse_name_set(args.positive_names)
    negative_names = parse_name_set(args.negative_names)
    neg_dirs = [Path(p).resolve() for p in args.neg_dir]

    train_samples = collect_samples(root, "train", positive_names, negative_names, neg_dirs, args.unknown_train_positive)
    test_samples = collect_samples(root, "test", positive_names, negative_names, neg_dirs, args.unknown_train_positive)
    if args.val_ratio > 0 and not test_samples:
        random.Random(args.seed).shuffle(train_samples)
        split = max(1, int(len(train_samples) * args.val_ratio))
        test_samples = train_samples[:split]
        train_samples = train_samples[split:]

    summary = {"train": summarize_samples(train_samples), "test": summarize_samples(test_samples)}
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if summary["train"]["by_label"]["negative"] == 0:
        raise SystemExit("未找到训练负样本。请放到 train/negative、train/neg、train/0，或用 --neg-dir 指定公开负样本目录。")
    if summary["train"]["by_label"]["positive"] == 0:
        raise SystemExit("未找到训练正样本。默认正样本目录名为 food, fruit, chart。")
    if not test_samples:
        raise SystemExit("未找到测试集。请提供 test 目录，或设置 --val-ratio 从训练集中切分验证集。")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_loader = make_loader(train_samples, args.image_size, args.batch_size, train=True, workers=args.workers)
    test_loader = make_loader(test_samples, args.image_size, args.batch_size, train=False, workers=args.workers)
    model = build_model(pretrained=args.pretrained).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for images, labels, _paths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.numel()
            seen += labels.numel()

        metrics = evaluate(model, test_loader, device, args.threshold)
        row = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), **metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "image_size": args.image_size,
                    "threshold": args.threshold,
                    "positive_names": sorted(positive_names),
                    "negative_names": sorted(negative_names),
                    "metrics": metrics,
                },
                out_dir / "best_mobilenetv3_small.pt",
            )

    (out_dir / "train_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"best_accuracy={best_acc:.4f}")
    print(f"checkpoint={out_dir / 'best_mobilenetv3_small.pt'}")


def prepare_cache(args: argparse.Namespace) -> None:
    root = Path(args.data).resolve()
    out_root = Path(args.cache_dir).resolve()
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = []
    for split in ("train", "test"):
        samples.extend(
            collect_samples(
                root,
                split,
                parse_name_set(args.positive_names),
                parse_name_set(args.negative_names),
                [Path(p).resolve() for p in args.neg_dir],
                args.unknown_train_positive,
            )
        )

    written = 0
    for sample in samples:
        try:
            rel_parts = sample.path.relative_to(root).parts
        except ValueError:
            rel_parts = ("external_neg",) + sample.path.parts[-3:]
        target = out_root.joinpath(*rel_parts).with_suffix(".jpg")
        target.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imdecode(np.fromfile(str(sample.path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip_bad_image={sample.path}")
            continue
        h, w = img.shape[:2]
        scale = args.image_size / min(h, w)
        new_w = max(args.image_size, int(round(w * scale)))
        new_h = max(args.image_size, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        y0 = max((new_h - args.image_size) // 2, 0)
        x0 = max((new_w - args.image_size) // 2, 0)
        cropped = resized[y0 : y0 + args.image_size, x0 : x0 + args.image_size]
        ok, encoded = cv2.imencode(".jpg", cropped, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        if ok:
            encoded.tofile(str(target))
            written += 1
    print(f"cached_images={written}")
    print(f"cache_dir={out_root}")


def preprocess_cv2(img_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def benchmark(args: argparse.Namespace) -> None:
    root = Path(args.data).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for split in ("test", "train"):
        samples.extend(
            collect_samples(
                root,
                split,
                parse_name_set(args.positive_names),
                parse_name_set(args.negative_names),
                [Path(p).resolve() for p in args.neg_dir],
                args.unknown_train_positive,
            )
        )
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("未找到可 benchmark 的图片。")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = build_model(pretrained=False).to(device)
    threshold = args.threshold
    loaded_checkpoint = False
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        threshold = float(ckpt.get("threshold", threshold))
        loaded_checkpoint = True
    model.eval()

    # Warmup keeps one-off kernel/setup cost out of the latency distribution.
    dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    rows = []
    totals = []
    correct = 0
    with torch.no_grad():
        for sample in samples:
            t0 = time.perf_counter()
            data = np.fromfile(str(sample.path), dtype=np.uint8)
            t1 = time.perf_counter()
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            t2 = time.perf_counter()
            if img is None:
                continue
            tensor = preprocess_cv2(img, args.image_size).to(device)
            t3 = time.perf_counter()
            logits = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t4 = time.perf_counter()
            prob = float(torch.softmax(logits, dim=1)[0, 1].cpu())
            pred = int(prob >= threshold)
            correct += int(pred == sample.label)
            row = {
                "path": str(sample.path),
                "label": sample.label,
                "pred": pred,
                "prob_positive": prob,
                "read_ms": (t1 - t0) * 1000,
                "decode_ms": (t2 - t1) * 1000,
                "preprocess_ms": (t3 - t2) * 1000,
                "inference_ms": (t4 - t3) * 1000,
                "total_ms": (t4 - t0) * 1000,
                "file_kb": sample.path.stat().st_size / 1024,
            }
            rows.append(row)
            totals.append(row["total_ms"])

    csv_path = out_dir / "latency_breakdown.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    stats = {
        "count": len(rows),
        "checkpoint_loaded": loaded_checkpoint,
        "accuracy": correct / max(len(rows), 1) if loaded_checkpoint else None,
        "accuracy_note": None if loaded_checkpoint else "未加载训练好的 checkpoint，accuracy 不代表模型效果，只用于链路延迟基线。",
        "p50_total_ms": percentile(totals, 50),
        "p90_total_ms": percentile(totals, 90),
        "p95_total_ms": percentile(totals, 95),
        "p99_total_ms": percentile(totals, 99),
        "max_total_ms": max(totals, default=0.0),
        "target_p95_ms": args.target_ms,
        "meets_target": percentile(totals, 95) <= args.target_ms,
        "csv": str(csv_path),
    }
    (out_dir / "latency_summary.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def scan(args: argparse.Namespace) -> None:
    root = Path(args.data).resolve()
    positive_names = parse_name_set(args.positive_names)
    negative_names = parse_name_set(args.negative_names)
    neg_dirs = [Path(p).resolve() for p in args.neg_dir]
    train_samples = collect_samples(root, "train", positive_names, negative_names, neg_dirs, args.unknown_train_positive)
    test_samples = collect_samples(root, "test", positive_names, negative_names, neg_dirs, args.unknown_train_positive)
    summary = {
        "root": str(root),
        "train": summarize_samples(train_samples),
        "test": summarize_samples(test_samples),
        "image_inspection": {
            "train": inspect_images(train_samples, args.inspect_limit),
            "test": inspect_images(test_samples, args.inspect_limit),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileNetV3-Small canteen-business binary experiment")
    parser.add_argument("--data", default=".", help="dataset root containing train/ and test/")
    parser.add_argument("--out", default="artifacts", help="output directory")
    parser.add_argument("--positive-names", default="food,fruit,chart")
    parser.add_argument("--negative-names", default="neg,negative,negatives,non_canteen,non-canteen,background,public,public_negative,0")
    parser.add_argument("--neg-dir", action="append", default=[], help="extra negative image dir; may contain train/test or flat images")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--unknown-train-positive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="treat extra train/ subfolders as positive chart/business samples",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan")
    scan_parser.add_argument("--inspect-limit", type=int, default=None)
    scan_parser.set_defaults(func=scan)

    cache_parser = subparsers.add_parser("prepare-cache")
    cache_parser.add_argument("--cache-dir", default="cache_160")
    cache_parser.add_argument("--jpeg-quality", type=int, default=90)
    cache_parser.add_argument("--overwrite", action="store_true")
    cache_parser.set_defaults(func=prepare_cache)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--val-ratio", type=float, default=0.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--pretrained", action="store_true")
    train_parser.set_defaults(func=train)

    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("--checkpoint", default="")
    bench_parser.add_argument("--threshold", type=float, default=0.5)
    bench_parser.add_argument("--limit", type=int, default=0)
    bench_parser.add_argument("--target-ms", type=float, default=50.0)
    bench_parser.set_defaults(func=benchmark)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
