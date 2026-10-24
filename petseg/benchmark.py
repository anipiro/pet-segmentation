"""Speed / size benchmark for the optimisation study.

    python -m petseg.benchmark runs/lraspp_mbv3/best.pt models/*.onnx --threads 4

For each model: file size, and single-image latency of the network call only
(batch 1, the API's workload), after warm-up, with a fixed thread count so
the numbers are comparable. Accuracy comes from petseg.evaluate.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from .inference import Predictor
from .utils import ROOT, save_json


def bench(path, threads, warmup=10, runs=100):
    p = Predictor(path, threads=threads)
    x = np.random.default_rng(0).standard_normal((1, 3, p.size, p.size)).astype(np.float32)
    for _ in range(warmup):
        p._run(x)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        p._run(x)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return {"model": str(path), "input_size": p.size, "threads": threads,
            "size_mb": Path(path).stat().st_size / 1e6, "latency_ms_mean": float(t.mean()),
            "latency_ms_p50": float(np.median(t)), "latency_ms_p95": float(np.percentile(t, 95)),
            "fps": float(1000 / t.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/benchmark.json")
    args = ap.parse_args()
    results = []
    for m in args.models:
        r = bench(m, args.threads)
        results.append(r)
        print(f"{m}: {r['size_mb']:.2f} MB, {r['latency_ms_mean']:.2f} ms "
              f"(p95 {r['latency_ms_p95']:.2f}) @ {r['input_size']}px")
    save_json(results, ROOT / args.out)


if __name__ == "__main__":
    main()
