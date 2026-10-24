"""Collect runs/*/summary.json + history.csv into results/experiments.{csv,md}
and plot validation curves to results/figures/val_curves.png."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.run_experiments import ORDER  # noqa: E402


def main():
    rows, curves = [], {}
    for name in ORDER:
        path = ROOT / "runs" / name / "summary.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())
        c = s["config"]
        rows.append({
            "experiment": name, "model": s["model"], "params_M": round(s["params"] / 1e6, 2),
            "pretrained": c["pretrained"], "loss": c["loss"], "augment": c["augment"],
            "border": c["border"], "best_epoch": s["best_val"]["epoch"],
            "val_fg_iou": round(s["best_val"]["fg_iou"], 4), "val_miou": round(s["best_val"]["miou"], 4),
            "val_dice": round(s["best_val"]["dice"], 4), "train_min": round(s["train_minutes"], 1),
        })
        curves[name] = pd.read_csv(ROOT / "runs" / name / "history.csv")

    out = ROOT / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "experiments.csv", index=False)
    (out / "experiments.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in curves.items():
        style = "--" if name.startswith("ablation") else "-"
        axes[0].plot(h["epoch"], h["val_fg_iou"], style, label=name)
        axes[1].plot(h["epoch"], h["train_loss"], style, label=name)
    axes[0].set(xlabel="epoch", ylabel="val foreground IoU", title="Validation IoU")
    axes[1].set(xlabel="epoch", ylabel="train loss", title="Training loss", yscale="log")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "val_curves.png", dpi=110)


if __name__ == "__main__":
    main()
