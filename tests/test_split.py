from PIL import Image

from scene_classification.config import Settings
from scene_classification.data.download import _stratified_split


def _make_dummy_dataset(root, classes, per_class):
    for cls in classes:
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            img = Image.new("RGB", (32, 32), color=(i * 3 % 255, 0, 0))
            img.save(root / cls / f"{cls}_{i}.jpg")


def test_stratified_split_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    src = tmp_path / "src"
    dst = tmp_path / "splits"
    classes = ["bedroom", "kitchen"]
    _make_dummy_dataset(src, classes, per_class=20)
    s = Settings()
    counts = _stratified_split(src, dst, classes, (0.7, 0.15, 0.15), s.seed)
    assert sum(counts.values()) == 40
    for split in ("train", "val", "test"):
        for cls in classes:
            assert (dst / split / cls).is_dir()
