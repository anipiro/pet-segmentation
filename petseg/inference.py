"""Inference on arbitrary images, with either a PyTorch checkpoint or an ONNX model.

    from petseg.inference import Predictor
    p = Predictor("models/petseg.onnx")   # or models/petseg_int8.onnx, runs/<name>/best.pt
    result = p.predict(Image.open("cat.jpg"))
    result.cutout().save("cat_cutout.png")

Pipeline: resize to the model's square input -> normalise -> network ->
softmax -> bilinearly upsample the pet probability to the original size ->
threshold. Upsampling probabilities (not the hard mask) gives smoother edges.
"""
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .utils import MEAN, STD


def preprocess(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return x.transpose(2, 0, 1)[None]  # (1, 3, H, W)


def softmax_fg(logits: np.ndarray) -> np.ndarray:
    """Pet probability from 2-class logits (1, 2, h, w) -> (h, w)."""
    # softmax over two classes == sigmoid of the logit difference
    return 1.0 / (1.0 + np.exp(logits[0, 0] - logits[0, 1]))


@dataclass
class Prediction:
    image: Image.Image
    prob: np.ndarray        # (H, W) float32, pet probability
    threshold: float
    latency_ms: float

    @property
    def mask(self) -> np.ndarray:
        return (self.prob >= self.threshold).astype(np.uint8)

    @property
    def foreground_ratio(self) -> float:
        return float(self.mask.mean())

    def bbox(self):
        """(x0, y0, x1, y1) of the pet pixels, or None if nothing found."""
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def mask_image(self) -> Image.Image:
        return Image.fromarray(self.mask * 255)

    def overlay(self, color=(255, 64, 129), alpha=0.5) -> Image.Image:
        rgb = np.asarray(self.image.convert("RGB"), dtype=np.float32)
        m = self.mask[..., None].astype(np.float32) * alpha
        out = rgb * (1 - m) + np.array(color, dtype=np.float32) * m
        return Image.fromarray(out.astype(np.uint8))

    def cutout(self, soft=True, crop=True) -> Image.Image:
        """Background removal: RGBA image whose alpha is the pet mask.

        soft=True uses the probability as alpha (feathered edges)."""
        alpha = self.prob if soft else self.mask.astype(np.float32)
        alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)
        rgba = np.dstack([np.asarray(self.image.convert("RGB")), alpha])
        img = Image.fromarray(rgba)  # (H, W, 4) uint8 -> RGBA
        box = self.bbox()
        return img.crop(box) if crop and box else img


class Predictor:
    def __init__(self, model_path, size=None, threshold=0.5, threads=None):
        self.path = Path(model_path)
        self.threshold = threshold
        if self.path.suffix == ".onnx":
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            self.session = ort.InferenceSession(str(self.path), opts,
                                                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            # Static-shape export -> read the size from the model itself.
            self.size = size or (shape[2] if isinstance(shape[2], int) else 224)
            self._run = lambda x: self.session.run(None, {self.input_name: x})[0]
        else:
            import torch

            from .models import build_model

            if threads:
                torch.set_num_threads(threads)
            ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            self.model = build_model(ckpt["model"], pretrained=False).eval()
            self.model.load_state_dict(ckpt["state_dict"])
            self.size = size or ckpt["config"]["size"]

            def run(x):
                with torch.inference_mode():
                    return self.model(torch.from_numpy(x)).numpy()

            self._run = run

    def predict_prob(self, img: Image.Image):
        """Pet probability at the original resolution, and model latency in ms."""
        x = preprocess(img, self.size)
        t0 = time.perf_counter()
        logits = self._run(x)
        latency = (time.perf_counter() - t0) * 1000
        prob = softmax_fg(logits)
        # float32 array -> PIL mode "F", so the resize interpolates probabilities
        prob = np.asarray(Image.fromarray(prob.astype(np.float32)).resize(img.size, Image.BILINEAR))
        return prob, latency

    def predict(self, img: Image.Image) -> Prediction:
        img = img.convert("RGB")
        prob, latency = self.predict_prob(img)
        return Prediction(img, prob, self.threshold, latency)
