"""Export a trained checkpoint to ONNX and quantise it to INT8.

    python -m petseg.export runs/final/best.pt --out models/petseg --size 224 --calib percentile

Produces (this exact command rebuilds the two models shipped in models/):
  <out>.onnx                    graph exported from PyTorch (static 1x3xSxS input)
  <out>_int8.onnx               post-training static quantisation (QDQ, per-channel
                                weights), activations calibrated on training images;
                                named <out>_int8_<calib>.onnx when several --calib are given

Why static rather than dynamic quantisation: dynamic quantisation only
quantises weights of MatMul/Gemm-style ops and computes activation ranges at
run time; a CNN is almost all Conv, which needs fixed activation ranges, i.e.
a calibration pass.

Calibration methods compared:
  minmax      range = observed min/max (sensitive to outliers)
  percentile  clip to the 99.999th percentile of observed activations
  entropy     choose the range that minimises KL divergence of the histogram
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image

from .data import DATA_DIR, read_meta
from .inference import preprocess
from .models import build_model

CALIB = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile,
         "entropy": CalibrationMethod.Entropy}


class PetCalibrationReader(CalibrationDataReader):
    """Feeds N training images (never val/test) through the exact inference preprocessing."""

    def __init__(self, input_name, size, n=128, seed=0):
        rows = read_meta("train")
        idx = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
        self.batches = iter([
            {input_name: preprocess(Image.open(DATA_DIR / "images" / f"{rows[i]['name']}.jpg"), size)}
            for i in idx
        ])

    def get_next(self):
        return next(self.batches, None)


def export_onnx(ckpt_path, onnx_path, size):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model"], pretrained=False).eval()
    model.load_state_dict(ckpt["state_dict"])
    dummy = torch.randn(1, 3, size, size)
    # TorchScript-based exporter: mature, and the static shape lets ORT
    # constant-fold and fuse Conv+BN+activation.
    torch.onnx.export(model, dummy, onnx_path, input_names=["image"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(onnx_path))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"image": dummy.numpy()})[0]
    print(f"  fp32 export ok, max |torch - onnx| = {np.abs(ref - got).max():.2e}")


def quantize(fp32_path, int8_path, size, calib, n_calib):
    prep = fp32_path.with_name(fp32_path.stem + "_prep.onnx")
    quant_pre_process(str(fp32_path), str(prep))  # shape inference + graph optimisation
    reader = PetCalibrationReader("image", size, n_calib)
    quantize_static(str(prep), str(int8_path), reader, quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, calibrate_method=CALIB[calib])
    prep.unlink()
    print(f"  int8 ({calib}) -> {int8_path} ({int8_path.stat().st_size / 1e6:.2f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--out", default="models/petseg")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--calib", nargs="*", default=["minmax"], choices=list(CALIB))
    ap.add_argument("--n-calib", type=int, default=128)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fp32 = out.with_name(f"{out.name}.onnx")
    export_onnx(args.checkpoint, fp32, args.size)
    print(f"  fp32 -> {fp32} ({fp32.stat().st_size / 1e6:.2f} MB)")
    for c in args.calib:
        # one calibration -> <out>_int8.onnx (what the app loads); several -> suffixed
        suffix = "_int8" if len(args.calib) == 1 else f"_int8_{c}"
        quantize(fp32, out.with_name(f"{out.name}{suffix}.onnx"), args.size, c, args.n_calib)


if __name__ == "__main__":
    main()
