import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.datasets import load_digits
from torchvision.datasets import CIFAR10

from canteen_binary_experiment import collect_samples, parse_name_set


def save_image(array: np.ndarray, target: Path, image_size: int, jpeg_quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError(f"failed to encode {target}")
    encoded.tofile(str(target))


def positive_count(root: Path, split: str, positive_names: set[str], negative_names: set[str]) -> int:
    samples = collect_samples(root, split, positive_names, negative_names, neg_dirs=[], unknown_train_positive=True)
    return sum(1 for sample in samples if sample.label == 1)


def export_split(dataset: CIFAR10, split: str, count: int, target_dir: Path, image_size: int, jpeg_quality: int, seed: int) -> int:
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    chosen = indices[:count]
    for n, idx in enumerate(chosen):
        image, label = dataset[idx]
        class_name = dataset.classes[label]
        target = target_dir / split / "negative" / f"cifar10_{split}_{n:05d}_{class_name}.jpg"
        save_image(np.asarray(image), target, image_size, jpeg_quality)
    return len(chosen)


def export_digits(split: str, count: int, target_dir: Path, image_size: int, jpeg_quality: int, seed: int) -> int:
    digits = load_digits()
    rng = random.Random(seed)
    indices = list(range(len(digits.images)))
    rng.shuffle(indices)
    repeats = (count + len(indices) - 1) // len(indices)
    chosen = (indices * repeats)[:count]
    for n, idx in enumerate(chosen):
        gray = digits.images[idx]
        gray = np.clip(gray / max(gray.max(), 1.0) * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
        label = int(digits.target[idx])
        target = target_dir / split / "negative" / f"sklearn_digits_{split}_{n:05d}_{label}.jpg"
        save_image(rgb, target, image_size, jpeg_quality)
    return len(chosen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced public CIFAR-10 negatives for the canteen binary task")
    parser.add_argument("--data", default=".", help="project data root containing train/test")
    parser.add_argument("--download-root", default="public_datasets")
    parser.add_argument("--out", default="balanced_data_160")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--positive-names", default="food,fruit,chart")
    parser.add_argument("--negative-names", default="neg,negative,negatives,non_canteen,non-canteen,background,public,public_negative,0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--source", choices=["sklearn-digits", "cifar10"], default="sklearn-digits")
    args = parser.parse_args()

    root = Path(args.data).resolve()
    out = Path(args.out).resolve()
    if out.exists() and args.overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    positive_names = parse_name_set(args.positive_names)
    negative_names = parse_name_set(args.negative_names)
    train_pos = positive_count(root, "train", positive_names, negative_names)
    test_pos = positive_count(root, "test", positive_names, negative_names)
    if train_pos == 0:
        raise SystemExit("no positive train images found")
    if test_pos == 0:
        raise SystemExit("no positive test images found")

    if args.source == "cifar10":
        download_root = Path(args.download_root).resolve()
        train_dataset = CIFAR10(root=str(download_root), train=True, download=True)
        test_dataset = CIFAR10(root=str(download_root), train=False, download=True)
        train_written = export_split(train_dataset, "train", train_pos, out, args.image_size, args.jpeg_quality, args.seed)
        test_written = export_split(test_dataset, "test", test_pos, out, args.image_size, args.jpeg_quality, args.seed + 1)
    else:
        train_written = export_digits("train", train_pos, out, args.image_size, args.jpeg_quality, args.seed)
        test_written = export_digits("test", test_pos, out, args.image_size, args.jpeg_quality, args.seed + 1)
    print(f"positive_train={train_pos}")
    print(f"positive_test={test_pos}")
    print(f"negative_train={train_written}")
    print(f"negative_test={test_written}")
    print(f"negative_dir={out}")
    print(f"source={args.source}")


if __name__ == "__main__":
    main()
