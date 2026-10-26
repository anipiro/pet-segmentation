"""Optimisation study: speed, size and accuracy before and after optimisation.

    python scripts/optimization_study.py --deploy lraspp_mbv3 [--limit N]

Three comparisons, all on the test split, with latency measured at batch 1 and
a fixed thread count:
  1. runtime / precision for the deployed model:
       PyTorch eager FP32 -> ONNX Runtime FP32 -> ONNX Runtime INT8 (3 calibrations)
  2. input resolution for the deployed model (ONNX FP32): 128 / 160 / 224 / 288
  3. architecture: every trained model as ONNX FP32 (accuracy vs speed frontier)

Writes results/optimization.{csv,md} and results/figures/speed_vs_iou.png.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from petseg.benchmark import bench  # noqa: E402
from petseg.evaluate import evaluate  # noqa: E402
from petseg.export import export_onnx, quantize  # noqa: E402

ARCHS = ["lraspp_mbv3", "deeplabv3_mbv3", "unet_resnet18", "unet_scratch"]


def measure(study, label, path, threads, limit):
    print(f"[{study}] {label}", flush=True)
    m = evaluate(path, f"opt_{study}_{label}", limit=limit, threads=threads, n_worst=4)
    b = bench(path, threads)
    return {"study": study, "variant": label, "input_px": b["input_size"],
            "size_mb": round(b["size_mb"], 2), "latency_ms": round(b["latency_ms_mean"], 2),
            "latency_p95_ms": round(b["latency_ms_p95"], 2), "fps": round(b["fps"], 1),
            "test_fg_iou": round(m["fg_iou"], 4), "test_miou": round(m["miou"], 4),
            "test_dice": round(m["dice"], 4)}


def plot(df, threads):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for study, marker in (("precision", "o"), ("resolution", "s"), ("architecture", "^")):
        d = df[df["study"] == study]
        ax.scatter(d["latency_ms"], d["test_fg_iou"], marker=marker, label=study, s=50)
        for _, r in d.iterrows():
            ax.annotate(r["variant"], (r["latency_ms"], r["test_fg_iou"]), fontsize=7,
                        xytext=(4, 3), textcoords="offset points")
    ax.set(xscale="log", xlabel=f"latency per image, ms (CPU, {threads} threads, log scale)",
           ylabel="test foreground IoU", title="Accuracy vs speed")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "speed_vs_iou.png", dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", default="lraspp_mbv3", help="run name of the model to deploy")
    ap.add_argument("--train-size", type=int, default=160)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, help="evaluate on the first N test images only")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results/optimization.csv")
    args = ap.parse_args()
    if args.plot_only:
        return plot(pd.read_csv(ROOT / "results" / "optimization.csv"), args.threads)

    work = ROOT / "models" / "study"
    work.mkdir(parents=True, exist_ok=True)
    ckpt = ROOT / "runs" / args.deploy / "best.pt"
    rows = []

    # 1. runtime and precision
    fp32 = work / f"{args.deploy}_{args.train_size}_fp32.onnx"
    export_onnx(ckpt, fp32, args.train_size)
    rows.append(measure("precision", "pytorch_fp32", ckpt, args.threads, args.limit))
    rows.append(measure("precision", "onnx_fp32", fp32, args.threads, args.limit))
    for calib in ("minmax", "percentile", "entropy"):
        int8 = work / f"{args.deploy}_{args.train_size}_int8_{calib}.onnx"
        quantize(fp32, int8, args.train_size, calib, n_calib=128)
        rows.append(measure("precision", f"onnx_int8_{calib}", int8, args.threads, args.limit))

    # 2. input resolution (the network is fully convolutional, so the same
    #    weights run at any size; accuracy shows how far it generalises)
    for size in (128, 160, 224, 288):
        path = work / f"{args.deploy}_{size}_fp32.onnx"
        if not path.exists():
            export_onnx(ckpt, path, size)
        rows.append(measure("resolution", f"{size}px", path, args.threads, args.limit))
    # INT8 at the best resolution (224 px): at 160 px quantisation overhead
    # cancelled the speed-up, so check whether larger inputs change that
    int8_224 = work / f"{args.deploy}_224_int8_percentile.onnx"
    quantize(work / f"{args.deploy}_224_fp32.onnx", int8_224, 224, "percentile", n_calib=128)
    rows.append(measure("resolution", "224px_int8_percentile", int8_224, args.threads, args.limit))

    # 3. architecture
    for arch in ARCHS:
        if not (ROOT / "runs" / arch / "best.pt").exists():
            continue
        path = work / f"{arch}_{args.train_size}_fp32.onnx"
        if not path.exists():
            export_onnx(ROOT / "runs" / arch / "best.pt", path, args.train_size)
        rows.append(measure("architecture", arch, path, args.threads, args.limit))

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    df.to_csv(out / "optimization.csv", index=False)
    (out / "optimization.md").write_text(df.to_markdown(index=False))
    print(df.to_string(index=False))

    plot(df, args.threads)


if __name__ == "__main__":
    main()
