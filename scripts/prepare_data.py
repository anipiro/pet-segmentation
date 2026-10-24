"""Download Oxford-IIIT Pet and build a compact, training-ready copy.

Reads straight from the .tar.gz archives (no full extraction), so it needs
~250 MB of disk instead of ~2 GB. For every image listed in the official
trainval/test splits it:
  * converts to RGB (the set contains a few grayscale / RGBA / CMYK files),
  * downsizes so the longer side is at most --max-side (we train at 224 px, so
    keeping 1000 px originals only slows the data loader),
  * resizes the trimap with nearest-neighbour so labels stay {1, 2, 3},
and writes data/processed/meta.csv with a breed-stratified train/val split.

Usage:  python scripts/prepare_data.py [--max-side 320] [--delete-archives]
"""
import argparse
import csv
import io
import random
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from petseg.utils import ROOT, use_system_certs  # noqa: E402

URLS = {
    "images.tar.gz": "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz",
    "annotations.tar.gz": "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz",
}
RAW = ROOT / "data" / "oxford-iiit-pet"
OUT = ROOT / "data" / "processed"


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    use_system_certs()
    for name, url in URLS.items():
        dst = RAW / name
        if dst.exists():
            continue
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dst)


def read_split_lists(ann_tar):
    """Return {name: (split, class_id, species, breed)} from trainval/test.txt."""
    rows = {}
    for split in ("trainval", "test"):
        text = ann_tar.extractfile(f"annotations/{split}.txt").read().decode()
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            name, class_id, species, _ = line.split()
            breed = name.rsplit("_", 1)[0]
            rows[name] = (split, int(class_id), "cat" if species == "1" else "dog", breed)
    return rows


def resize_pair(img, trimap, max_side):
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.resize(size, Image.BICUBIC), trimap.resize(size, Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=320)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delete-archives", action="store_true")
    args = ap.parse_args()

    download()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)

    with tarfile.open(RAW / "annotations.tar.gz") as ann:
        rows = read_split_lists(ann)
        trimaps = {}
        for m in ann.getmembers():
            p = Path(m.name)
            if p.parent.name == "trimaps" and p.suffix == ".png" and not p.name.startswith("._"):
                if p.stem in rows:
                    trimaps[p.stem] = ann.extractfile(m).read()

    meta, skipped = [], []
    # Streaming mode ("r|gz") walks the archive once without building an index.
    with tarfile.open(RAW / "images.tar.gz", "r|gz") as imgs:
        for m in imgs:
            p = Path(m.name)
            if p.suffix != ".jpg" or p.stem not in rows or p.stem not in trimaps:
                continue
            try:
                img = Image.open(io.BytesIO(imgs.extractfile(m).read()))
                img.load()
            except Exception as e:  # a handful of files in the set are corrupt
                skipped.append((p.stem, str(e)))
                continue
            orig_w, orig_h = img.size
            img = img.convert("RGB")
            trimap = Image.open(io.BytesIO(trimaps[p.stem]))
            if trimap.size != img.size:
                skipped.append((p.stem, "size mismatch"))
                continue
            img, trimap = resize_pair(img, trimap, args.max_side)
            img.save(OUT / "images" / f"{p.stem}.jpg", quality=92)
            trimap.save(OUT / "masks" / f"{p.stem}.png")
            split, class_id, species, breed = rows[p.stem]
            meta.append(dict(name=p.stem, split=split, class_id=class_id, species=species,
                             breed=breed, width=img.width, height=img.height,
                             orig_width=orig_w, orig_height=orig_h))
            if len(meta) % 500 == 0:
                print(f"  {len(meta)} images")

    # Carve a validation set out of trainval, stratified by breed so every
    # breed is represented. The official test split stays untouched until the
    # final evaluation.
    rng = random.Random(args.seed)
    by_breed = defaultdict(list)
    for r in meta:
        if r["split"] == "trainval":
            by_breed[r["breed"]].append(r)
    for group in by_breed.values():
        group.sort(key=lambda r: r["name"])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * args.val_frac))
        for i, r in enumerate(group):
            r["split"] = "val" if i < n_val else "train"

    meta.sort(key=lambda r: r["name"])
    with open(OUT / "meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
        w.writeheader()
        w.writerows(meta)

    counts = defaultdict(int)
    for r in meta:
        counts[r["split"]] += 1
    print(f"done: {dict(counts)}; skipped {len(skipped)}: {skipped}")

    if args.delete_archives:
        for name in URLS:
            (RAW / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
