"""Run the model on your own images from the command line.

    python scripts/predict.py photo.jpg more/*.png --out predictions/ [--model models/petseg_int8.onnx]

For each input writes <name>_overlay.png, <name>_mask.png and <name>_cutout.png.
"""
import argparse
import glob
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from petseg.inference import Predictor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=str(ROOT / "models" / "petseg.onnx"))
    ap.add_argument("--out", default="predictions")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = Predictor(args.model, threshold=args.threshold)
    # Windows shells don't expand wildcards, so do it here
    paths = [f for pattern in args.images for f in (glob.glob(pattern) or [pattern])]
    for path in paths:
        r = p.predict(Image.open(path))
        stem = Path(path).stem
        r.overlay().save(out / f"{stem}_overlay.png")
        r.mask_image().save(out / f"{stem}_mask.png")
        r.cutout().save(out / f"{stem}_cutout.png")
        print(f"{path}: pet covers {r.foreground_ratio:.1%}, bbox {r.bbox()}, {r.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
