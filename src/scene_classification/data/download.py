"""
Indoor scene dataset download + split.

Default source: MIT Indoor Scenes 67 (Quattoni & Torralba, 2009) — public research dataset.
To keep the scope close to the original paper, we can subset to {bedroom, bathroom,
livingroom, kitchen} via --classes.
"""

from __future__ import annotations

import random
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

from scene_classification.config import Settings

INDOOR67_URL = "http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar"
INDOOR67_TARBALL = "indoorCVPR_09.tar"
INDOOR67_INNER_DIR = "Images"

PAPER_CLASSES = ("bedroom", "bathroom", "livingroom", "kitchen")


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    urlretrieve(url, dest)
    return dest


def _extract(tar_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path) as tf:
        tf.extractall(out_dir)
    return out_dir / INDOOR67_INNER_DIR


def _stratified_split(
    src: Path, dst: Path, classes: list[str], ratios: tuple[float, float, float], seed: int
) -> dict[str, int]:
    rng = random.Random(seed)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for cls in classes:
        cls_src = src / cls
        if not cls_src.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {cls_src}")
        images = sorted(p for p in cls_src.iterdir() if p.is_file())
        rng.shuffle(images)
        n = len(images)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        buckets = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }
        for split, paths in buckets.items():
            out_cls = dst / split / cls
            out_cls.mkdir(parents=True, exist_ok=True)
            for p in paths:
                shutil.copy2(p, out_cls / p.name)
            counts[split] += len(paths)
    return counts


def prepare_dataset(
    settings: Settings,
    classes: list[str] | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, int]:
    settings.ensure_dirs()
    classes = list(classes) if classes else list(PAPER_CLASSES)

    tar_path = settings.raw_dir / INDOOR67_TARBALL
    _download(INDOOR67_URL, tar_path)
    images_dir = settings.raw_dir / INDOOR67_INNER_DIR
    if not images_dir.is_dir():
        _extract(tar_path, settings.raw_dir)

    split_root = settings.processed_dir / "splits"
    if split_root.exists():
        shutil.rmtree(split_root)
    return _stratified_split(images_dir, split_root, classes, ratios, settings.seed)
