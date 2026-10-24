"""Dataset exploration: split sizes, class balance, image sizes, label statistics.

Writes results/eda.json and results/figures/{eda.png, samples.png}.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from petseg.data import DATA_DIR, read_meta  # noqa: E402


def main():
    df = pd.DataFrame(read_meta())
    for c in ("width", "height", "orig_width", "orig_height"):
        df[c] = df[c].astype(int)

    fractions = []
    for name in df["name"]:
        t = np.array(Image.open(DATA_DIR / "masks" / f"{name}.png"))
        fractions.append([(t == 1).mean(), (t == 2).mean(), (t == 3).mean()])
    df[["pet", "background", "border"]] = np.array(fractions)

    eda = {
        "n_images": len(df),
        "splits": df["split"].value_counts().to_dict(),
        "species": df["species"].value_counts().to_dict(),
        "n_breeds": int(df["breed"].nunique()),
        "images_per_breed": {"min": int(df["breed"].value_counts().min()),
                             "max": int(df["breed"].value_counts().max())},
        "orig_size_median": [int(df["orig_width"].median()), int(df["orig_height"].median())],
        "orig_size_min": [int(df["orig_width"].min()), int(df["orig_height"].min())],
        "orig_size_max": [int(df["orig_width"].max()), int(df["orig_height"].max())],
        "pixel_fraction_mean": {k: float(df[k].mean()) for k in ("pet", "background", "border")},
        "pet_fraction_quantiles": {str(q): float(df["pet"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "pet_fraction_by_species": df.groupby("species")["pet"].mean().to_dict(),
    }
    out = ROOT / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "eda.json").write_text(json.dumps(eda, indent=2))
    print(json.dumps(eda, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    df["breed"].value_counts().plot.bar(ax=axes[0], color=["#4c72b0"])
    axes[0].set(title="Images per breed (37 breeds)", xticks=[])
    for sp, g in df.groupby("species"):
        axes[1].hist(g["pet"] + g["border"], bins=40, alpha=0.6, label=sp)
    axes[1].set(title="Pet area fraction (pet + border)", xlabel="fraction of image")
    axes[1].legend()
    axes[2].scatter(df["orig_width"], df["orig_height"], s=2, alpha=0.3)
    axes[2].set(title="Original image size", xlabel="width", ylabel="height")
    fig.tight_layout()
    fig.savefig(out / "figures" / "eda.png", dpi=100)

    sample = df.sample(6, random_state=1)
    fig, axes = plt.subplots(2, 6, figsize=(15, 5))
    for i, (_, r) in enumerate(sample.iterrows()):
        axes[0, i].imshow(Image.open(DATA_DIR / "images" / f"{r['name']}.jpg"))
        axes[0, i].set_title(r["breed"], fontsize=8)
        # trimap: 1 pet (white), 2 background (black), 3 border (grey)
        t = np.array(Image.open(DATA_DIR / "masks" / f"{r['name']}.png"))
        axes[1, i].imshow(np.select([t == 1, t == 3], [1.0, 0.5], 0.0), cmap="gray", vmin=0, vmax=1)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "figures" / "samples.png", dpi=90)


if __name__ == "__main__":
    main()
