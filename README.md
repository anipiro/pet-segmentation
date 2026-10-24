# Pet segmentation and background removal

An end-to-end computer vision project. A model finds the cat or dog in a photo
and separates it from the background, pixel by pixel. The result drives a
background-removal API and a small web app.

Data: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (7,349 images, 37 breeds, pixel-level trimaps).

| | |
|---|---|
| Task | Binary semantic segmentation (pet vs background) |
| Deployed model | LR-ASPP + MobileNetV3-Large, INT8 ONNX (see [report](docs/REPORT.md)) |
| Serving | FastAPI + ONNX Runtime on CPU, Docker image without PyTorch |
| Everything trained on | a laptop CPU (Ryzen 7 7730U), no GPU, no paid services |

Results, experiments and design decisions are in **[docs/REPORT.md](docs/REPORT.md)**.
How AI tools were used: [docs/AI_USAGE.md](docs/AI_USAGE.md).

## Quick start: run the demo

The trained model is in the repo (`models/petseg_int8.onnx`, a few MB), so the
demo needs no training and no dataset.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install -r requirements-serve.txt
uvicorn app.main:app --port 8000
```

Open http://localhost:8000, then drop in a photo or click a sample.

Or with Docker:

```bash
docker build -t petseg .
docker run -p 8000:8000 petseg
```

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | model name, input size |
| POST | `/predict` | multipart `file` | JSON: pet area, bbox, timings, base64 PNGs of mask, overlay and cutout |
| POST | `/cutout?crop=true` | multipart `file` | `image/png`, the pet on a transparent background |

```bash
curl -F file=@my_dog.jpg http://localhost:8000/cutout -o my_dog_cutout.png
curl -F file=@my_dog.jpg http://localhost:8000/predict
```

Interactive docs: http://localhost:8000/docs.

### From Python

```python
from PIL import Image
from petseg.inference import Predictor

p = Predictor("models/petseg_int8.onnx")      # or runs/<name>/best.pt
r = p.predict(Image.open("cat.jpg"))
print(r.foreground_ratio, r.bbox(), r.latency_ms)
r.cutout().save("cat_cutout.png")
r.overlay().save("cat_overlay.png")
```

## Full pipeline: reproduce everything

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt

python scripts/prepare_data.py --delete-archives   # download ~800 MB, keep a ~190 MB resized copy
python scripts/explore_data.py                     # dataset statistics -> results/eda.json, figures
python scripts/run_experiments.py                  # all 8 training runs -> runs/, results/experiments.md
python -m petseg.evaluate runs/lraspp_mbv3/best.pt # test-set metrics + error analysis -> results/eval/
python -m petseg.export runs/lraspp_mbv3/best.pt --out models/petseg --calib minmax percentile entropy
python scripts/optimization_study.py               # speed / size / accuracy table -> results/optimization.md
pytest                                             # unit + API tests
```

One run: `python -m petseg.train configs/unet_resnet18.yaml [--epochs N --size 224 --limit 64]`.
Every config in `configs/` overrides a few keys of `configs/base.yaml`.

On CPU the full set of experiments takes about 5 hours. `notebooks/colab_train.ipynb`
runs the same on a free Colab GPU.

## Project structure

```
petseg/                 the library
  data.py               dataset, trimap -> label mapping, augmentations
  models.py             U-Net (scratch), U-Net/ResNet-18, LR-ASPP, DeepLabV3
  losses.py             cross-entropy, CE + soft Dice (with ignore pixels)
  metrics.py            confusion-matrix IoU / Dice / accuracy, per-image IoU
  train.py              training loop, checkpoints on best validation IoU
  evaluate.py           test-set evaluation + error analysis (per breed, size, worst cases)
  export.py             ONNX export + INT8 static quantisation
  benchmark.py          latency / size measurement
  inference.py          Predictor (PyTorch or ONNX) + cutout / overlay helpers
app/                    FastAPI service and web demo
scripts/                data preparation, EDA, experiment runner, summaries
configs/                one YAML per experiment
models/                 exported models used by the app
results/                metrics, tables and figures quoted in the report
docs/                   technical report, AI-usage note
notebooks/              Colab notebook for GPU training
tests/                  pytest suite
```
