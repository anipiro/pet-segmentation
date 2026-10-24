"""Evaluate a model on the held-out test split and analyse its errors.

    python -m petseg.evaluate runs/lraspp_mbv3/best.pt
    python -m petseg.evaluate models/petseg_int8.onnx --name lraspp_int8

Runs through the same Predictor used by the API, at each image's own
resolution (predictions are upsampled from the network input size), so the
numbers describe the deployed pipeline rather than a 224x224 proxy.

Writes results/eval/<name>/{metrics.json, per_image.csv, worst.png}.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .data import DATA_DIR, read_meta, trimap_to_label
from .inference import Predictor
from .metrics import ConfusionMeter, image_iou
from .utils import ROOT, save_json

# Pet size buckets by fraction of the image covered by the pet.
SIZE_BINS = [0, 0.15, 0.35, 0.6, 1.01]
SIZE_LABELS = ["tiny (<15%)", "small (15-35%)", "medium (35-60%)", "large (>60%)"]


def evaluate(model_path, name, split="test", limit=None, threads=None, n_worst=12):
    import torch

    pred = Predictor(model_path, threads=threads)
    rows = read_meta(split)[:limit] if limit else read_meta(split)
    meter = ConfusionMeter()
    by_species = defaultdict(ConfusionMeter)
    per_image, latencies = [], []

    for r in rows:
        img = Image.open(DATA_DIR / "images" / f"{r['name']}.jpg").convert("RGB")
        target = trimap_to_label(np.array(Image.open(DATA_DIR / "masks" / f"{r['name']}.png")),
                                 border="foreground")
        p = pred.predict(img)
        mask = p.mask
        latencies.append(p.latency_ms)
        t_mask, t_target = torch.from_numpy(mask.astype(np.int64)), torch.from_numpy(target.astype(np.int64))
        meter.update(t_mask, t_target)
        by_species[r["species"]].update(t_mask, t_target)
        per_image.append({"name": r["name"], "species": r["species"], "breed": r["breed"],
                          "iou": image_iou(mask, target),
                          "fg_ratio_true": float(target.mean()), "fg_ratio_pred": float(mask.mean())})

    df = pd.DataFrame(per_image)
    df["size_bucket"] = pd.cut(df["fg_ratio_true"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    breed = df.groupby("breed")["iou"].mean().sort_values()
    metrics = {
        "name": name, "model_path": str(model_path), "split": split, "n_images": len(df),
        "input_size": pred.size,
        **meter.compute(),
        "mean_image_iou": float(df["iou"].mean()),
        "median_image_iou": float(df["iou"].median()),
        "images_iou_below_0.5": int((df["iou"] < 0.5).sum()),
        "latency_ms_mean": float(np.mean(latencies[5:] or latencies)),
        "by_species": {k: m.compute()["fg_iou"] for k, m in sorted(by_species.items())},
        "by_size": {str(k): float(v) for k, v in
                    df.groupby("size_bucket", observed=True)["iou"].mean().items()},
        "worst_breeds": {k: float(v) for k, v in breed.head(5).items()},
        "best_breeds": {k: float(v) for k, v in breed.tail(5).items()},
    }
    out = ROOT / "results" / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out / "metrics.json")
    df.to_csv(out / "per_image.csv", index=False)
    save_worst_gallery(pred, df.nsmallest(n_worst, "iou"), out / "worst.png")
    return metrics


def save_worst_gallery(pred, worst, path):
    """Image | ground truth | prediction for the lowest-IoU test images."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(worst), 3, figsize=(7.5, 2.4 * len(worst)))
    for ax_row, (_, r) in zip(np.atleast_2d(axes), worst.iterrows()):
        img = Image.open(DATA_DIR / "images" / f"{r['name']}.jpg").convert("RGB")
        target = trimap_to_label(np.array(Image.open(DATA_DIR / "masks" / f"{r['name']}.png")),
                                 border="foreground")
        p = pred.predict(img)
        for ax, im, title in zip(ax_row, [img, target, p.mask],
                                 [r["name"], "ground truth", f"pred IoU={r['iou']:.2f}"]):
            ax.imshow(im, cmap=None if im is img else "gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--name", help="defaults to the run directory name")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()
    name = args.name or Path(args.model).parent.name
    m = evaluate(args.model, name, args.split, args.limit, args.threads)
    print(f"{name}: fgIoU {m['fg_iou']:.4f}  mIoU {m['miou']:.4f}  Dice {m['dice']:.4f}  "
          f"acc {m['pixel_acc']:.4f}  ({m['latency_ms_mean']:.1f} ms/img)")


if __name__ == "__main__":
    main()
