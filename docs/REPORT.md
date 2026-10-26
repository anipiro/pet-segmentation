# Technical report: pet segmentation and background removal

All numbers below come from files in `results/` and can be regenerated with
the commands in the README. Everything was trained and measured on a laptop
CPU (AMD Ryzen 7 7730U, 8 cores, 16 GB RAM, no GPU).

## 1. Problem

Given a photo of a cat or dog, mark which pixels belong to the animal. The
mask has a direct use: **background removal**. The API returns the pet on a
transparent background, an overlay, the mask, the pet's bounding box and the
share of the image it covers. Product photos for pet shops, adoption-site
listings, stickers and avatar tools all need this. It also has to run cheaply
on a CPU, which is why speed and model size matter as much as accuracy here.

Formally: binary semantic segmentation, pet vs background.

## 2. Data

**Oxford-IIIT Pet** (Parkhi et al., 2012): 7,349 images of 37 breeds (12 cat,
25 dog), each with a pixel-level *trimap*: 1 = pet, 2 = background,
3 = border (a band around the outline where the annotators were not sure).

Why this dataset: it is the right size for CPU training (about 3.7k training
images), it has dense masks rather than boxes, and it fits a concrete product
(background removal). Penn-Fudan has only 170 images. BDD100K and VOC are much
larger, and too big for the compute budget to allow several experiments.

Exploration (`scripts/explore_data.py`, `results/eda.json`, `results/figures/eda.png`):

| | |
|---|---|
| Images | 7,349 (cats 2,371, dogs 4,978) |
| Per breed | 184 to 200, so breeds are balanced but species are 2:1 dogs |
| Original size | median 500×375, from 114×103 up to 3264×2606 |
| Pixel share | pet 30%, background 58%, **border 12%** |
| Pet area per image | 5th pct 7.6%, median 27%, 95th pct 60% |

Two findings shaped the project:

1. **The border class is 12% of all pixels.** How it is labelled is not a detail,
   so it became an experiment (section 4.2), and it produced the largest gain of the project.
2. **14 trimaps are empty** (no pet pixels at all: 6 train, 1 val, 7 test), and a
   few look inverted. These were found while inspecting the worst test predictions
   (section 5.3). They are kept in the test set so the numbers stay comparable with the
   official split. They are reported separately.

Preparation (`scripts/prepare_data.py`): images are read straight from the
downloaded archives, converted to RGB (the set contains grayscale/RGBA/CMYK
files), and downsized so the longer side is at most 320 px. The trimap is
resized with nearest-neighbour so labels stay valid. This cuts the dataset
from about 800 MB to 190 MB and makes data loading cheap, since training runs at 160 px.

Splits: the official `test` list (3,669 images) is the test set and is used
only for final evaluation. The official `trainval` list is split 90/10,
**stratified by breed**, into train (3,311) and validation (369). All model
selection (best epoch, choosing an architecture, choosing ablations) uses validation only.

Augmentation (torchvision v2, applied jointly to image and mask): random
resized crop (scale 0.5 to 1), horizontal flip, colour jitter, 5% grayscale.

## 3. Models

All models output two-class logits at the input resolution (`petseg/models.py`).

| Name | Idea | Params |
|---|---|---|
| `unet_scratch` | Classic U-Net, random init, 16 base channels (CPU budget) | 1.9M |
| `unet_resnet18` | U-Net decoder on an ImageNet ResNet-18 encoder (skips at /2, /4, /8, /16) | 14.3M |
| `lraspp_mbv3` | LR-ASPP head on MobileNetV3-Large (ImageNet backbone), built for mobile | 3.2M |
| `deeplabv3_mbv3` | DeepLabV3 (atrous spatial pyramid pooling) on MobileNetV3-Large | 11.0M |

Only the backbones are pretrained, on ImageNet classification. I deliberately
did **not** use torchvision's COCO/VOC segmentation weights: VOC contains
"cat" and "dog" classes, which would hide how much the fine-tuning itself contributes.

Training (`petseg/train.py`): AdamW (lr 1e-3, weight decay 1e-4), one epoch of
linear warm-up then cosine decay, batch 16, 160×160 px, 10 epochs.
Loss = cross-entropy + soft Dice, with border pixels ignored by default. The
checkpoint with the best validation foreground IoU is kept.

**Metrics** (`petseg/metrics.py`). The headline metric is **foreground IoU**:
the pet is the class the product cares about, and background IoU is inflated
by large easy backgrounds. Also reported: mIoU (pet and background), Dice,
pixel accuracy (misleading alone, since a model that predicts "background"
everywhere gets 58%), plus per-image IoU for error analysis. The evaluation
target always counts border pixels as pet, so models trained either way are
scored against the same labels. Test evaluation runs through the deployed
inference path at each image's own resolution: predictions are upsampled from
the network size, not scored at 160×160.

## 4. Experiments

### 4.1 Architecture comparison

Same recipe for all four models (full training set, 10 epochs, 160 px).
Latency is the ONNX Runtime FP32 network call, batch 1, 4 threads.

| Model | Params | Val IoU | **Test IoU** | Test Dice | Latency | ONNX size |
|---|---|---|---|---|---|---|
| **LR-ASPP / MobileNetV3** | 3.2M | 0.835 | **0.829** | 0.906 | **4.0 ms** | 12.9 MB |
| DeepLabV3 / MobileNetV3 | 11.0M | 0.830 | 0.819 | 0.901 | 7.1 ms | 44.1 MB |
| U-Net / ResNet-18 | 14.3M | 0.816 | 0.804 | 0.891 | 14.2 ms | 57.3 MB |
| U-Net from scratch | 1.9M | 0.765 | 0.764 | 0.866 | 8.6 ms | 7.8 MB |

(`results/experiments.md`, `results/optimization.md`, curves in `results/figures/val_curves.png`)

- **LR-ASPP wins on both axes**: it is the most accurate and the fastest. It was the
  obvious choice to deploy and to use as the base for the ablations.
- **Pretraining matters more than architecture.** The scratch U-Net is 6.5 points
  behind, and its curve was still climbing at epoch 10. With ten times the epochs it
  would close part of the gap, but that is exactly the compute I don't have.
- The bigger pretrained models did not beat the small one at this budget. A
  plausible reason is that 10 epochs at 160 px under-trains the larger decoders
  (the U-Net/ResNet-18 decoder is randomly initialised and wide). Train-time minutes
  are not comparable between runs, because some runs shared the CPU with another job.

### 4.2 Ablations (one change at a time)

To fit five runs into the CPU budget, the ablations use a cheaper protocol:
LR-ASPP, a random 50% of the training images, 6 epochs. Each is compared with
a baseline trained under the same protocol (`ablation_baseline`), so the
comparisons are fair even though the absolute numbers are lower than in 4.1.

| Change vs baseline | Val IoU | Δ | Reading |
|---|---|---|---|
| baseline (pretrained, CE+Dice, augmentation, border ignored) | 0.821 | | |
| no ImageNet pretraining | 0.655 | **−16.6** | by far the most important ingredient |
| cross-entropy only (no Dice) | 0.818 | −0.3 | small but consistent gain from Dice |
| no augmentation | 0.820 | −0.1 | equal at 6 epochs, but train loss 0.050 vs 0.087: already memorising, would overfit with longer training |
| **border labelled as pet** | **0.872** | **+5.1** | biggest single gain |

**Why the border result is so large.** With the border ignored, the loss
never says what to do with the 12% of pixels around the outline, and the model
learns to stop at the inner, confident edge. Fur, whiskers and ears are
under-segmented. Evaluation (and the product: you don't want ears cut off a
cutout) counts the band as pet. Labelling it as pet aligns the training
target with what is measured and what users want. In hindsight it is a
train/eval objective mismatch. The ablation exposed it. Making the eval target
match the training choice would have hidden it.

### 4.3 Final model

LR-ASPP + border-as-pet, full training set, 10 epochs (`configs/final.yaml`):

| | Val | Test (PyTorch, 160 px) |
|---|---|---|
| Foreground IoU | 0.888 | **0.893** |
| mIoU | 0.903 | 0.906 |
| Dice | 0.941 | 0.944 |
| Pixel accuracy | | 0.952 |

That is +6.5 IoU points over the same architecture from 4.1, from a one-line
label change.

## 5. Evaluation and error analysis (final model, test set)

`results/eval/final/` (`metrics.json`, `per_image.csv`, `worst.png`)

### 5.1 Where it is good and where it is weak

| Slice | Test IoU |
|---|---|
| cats / dogs | 0.888 / 0.896 |
| pet covers < 15% of image | **0.728** |
| 15 to 35% | 0.850 |
| 35 to 60% | 0.898 |
| > 60% | 0.935 |
| worst breeds | Bengal 0.818, Miniature Pinscher 0.830, Chihuahua 0.845, Sphynx 0.848 |
| best breeds | Havanese 0.915, Samoyed 0.913, British Shorthair 0.909 |

Per image: mean IoU 0.879, median 0.901; 29 of 3,669 images are below 0.5.

- **Small pets are the main weakness.** At 160 px a pet covering 5% of the frame is a
  few hundred pixels, and the outline is lost. The border change helped most here
  (0.657 → 0.728 for the smallest bucket), and so does a larger inference size (6.2).
- **Breeds that look like their background** do worst: Bengal's spotted coat on
  patterned blankets, and the small, thin-legged Pinscher and Chihuahua. Sphynx is hairless,
  so it is skin-coloured and often on skin-coloured furniture.
- Cats are slightly harder than dogs, even though cats are the minority class
  in training. The gap (0.8 points) is smaller than the breed spread.

### 5.2 Looking at the worst cases

`results/eval/final/worst.png` shows the 12 lowest-IoU test images:

- **7 of the 12 are annotation errors, not model errors.** Their ground truth is
  completely empty, and the model's prediction is a correct pet silhouette. Two more
  (Maine_Coon_268, Persian_268) have trimaps that seem to mark the background as pet.
- The genuine failures: a small cat far away in a garden (missed), a black cat in a
  dark cupboard (barely detected), and a Bengal lying on a white fur rug (the rug is
  segmented as pet).

### 5.3 Label noise

Scanning all masks found 14 empty trimaps (6 train, 1 val, 7 test). In
training they teach "this pet is background" a few times, which is minor at
3.3k images. In testing, every one of them scores IoU 0, whatever the model
does. I kept them so that the test numbers use the official, untouched split,
but cleaning them out of training is the first thing on the improvement list.

## 6. Optimisation

Starting point: the final model in PyTorch eager mode, FP32, 160 px.
All latencies are ONNX Runtime / PyTorch on CPU, batch 1, 4 threads, mean of
100 runs after warm-up. Accuracy is on the full test set (`scripts/optimization_study.py`,
`results/optimization.md`, `results/figures/speed_vs_iou.png`).

### 6.1 Runtime and precision (160 px)

| Variant | Size | Latency | Test IoU |
|---|---|---|---|
| PyTorch eager FP32 (baseline) | 13.1 MB | 17.0 ms | 0.893 |
| **ONNX Runtime FP32** | 12.9 MB | **4.5 ms** (3.8× faster) | 0.893 (identical) |
| ONNX INT8, min-max calibration | **3.7 MB** (3.5× smaller) | 4.6 ms | 0.869 (−2.5) |
| ONNX INT8, percentile calibration | 3.7 MB | 4.9 ms | 0.876 (−1.7) |
| ONNX INT8, entropy calibration | 3.7 MB | 4.6 ms | 0.869 (−2.5) |

- **Exporting to ONNX is the big win.** It is 3.8× faster with bit-for-bit the same
  accuracy (export is checked at max |Δlogit| ≈ 1e-6). ORT fuses Conv+BN+activation and
  avoids Python/dispatcher overhead, which dominates a small model at batch 1.
- **INT8 static quantisation** (QDQ, per-channel weights, 128 training images for
  calibration) gives a **3.5× smaller** file but **no speed-up at 160 px**, and costs
  1.7 to 2.5 IoU points. Percentile calibration is best because it clips activation
  outliers that stretch the min-max range and waste the 256 levels.
- Why no speed-up, as a hypothesis consistent with the measurements below: this CPU
  (Zen 3) has AVX2 but no VNNI int8 instructions, so int8 convolutions gain little. Meanwhile
  MobileNetV3's hard-swish, sigmoid and element-wise ops stay in float and need
  quantise/dequantise pairs around them. At 160 px those fixed overheads eat the gain.
  At 224 px the convolutions dominate and INT8 does get faster (6.2).

### 6.2 Input resolution (same weights, ONNX FP32)

| Input | Latency | Test IoU |
|---|---|---|
| 128 px | 3.0 ms | 0.870 |
| 160 px (training size) | 4.1 ms | 0.893 |
| **224 px** | **6.6 ms** | **0.902** |
| 288 px | 10.0 ms | 0.892 |
| 224 px, INT8 (percentile) | 4.6 ms | 0.886 |

A model trained at 160 px is *more* accurate at 224 px. My explanation: the
random-resized-crop augmentation trains on zoomed-in crops (scale 0.5 to 1), so
pets appear larger during training than in a whole image resized to 160. Running
at 224 brings the apparent object size back towards the training
distribution. At 288 the objects are larger than anything seen in training,
and accuracy falls again. At 224 px INT8 is 1.4× faster than FP32 (4.6 vs 6.6 ms).

### 6.3 What was deployed and why

| Use | Model | Size | Latency | Test IoU |
|---|---|---|---|---|
| **Default (API, demo, Docker)** | `models/petseg.onnx`, FP32, 224 px | 12.9 MB | 6.6 ms | **0.902** |
| Size-constrained (mobile, browser download) | `models/petseg_int8.onnx`, INT8, 224 px | **3.7 MB** | 4.6 ms | 0.886 |
| Original baseline | PyTorch FP32, 160 px | 13.1 MB | 17.0 ms | 0.893 |

On a server CPU 6.6 ms is already far below the cost of decoding an upload, so
I chose accuracy: the deployed model is **2.6× faster and slightly more accurate**
than the starting point. The INT8 file is the right choice where the 9 MB
difference matters more than 1.6 IoU points. The serving image needs only ONNX
Runtime, not PyTorch (`requirements-serve.txt`), which I checked in a
clean environment.

## 7. Application

- **API** (`app/main.py`, FastAPI): `POST /predict` returns JSON (pet area, bbox,
  timings, base64 mask/overlay/cutout); `POST /cutout` returns a transparent PNG;
  `GET /health`. Uploads are size-limited, invalid images return a clear 400, and very
  large images are downscaled before inference.
- **Web demo** (`/`): drop a photo or pick one of five test-set samples. A
  before/after slider compares the original with the cutout, and there are overlay
  and mask views plus a PNG download. `?sample=<name>` opens a sample directly.
- **CLI**: `python scripts/predict.py photos/*.jpg --out predictions/`.
- **Python**: `Predictor(path).predict(img).cutout()`. It works with a `.pt` checkpoint
  or an `.onnx` file through the same interface.
- **Docker**: CPU-only image with ONNX Runtime.
- **Tests** (`pytest`, 12): metric values against hand calculations, ignore-pixel
  handling in the loss, output shapes on odd-sized inputs for every model, the
  probability→mask path, and the API end to end on a tiny exported model.

## 8. Problems and how they were solved

| Problem | Solution |
|---|---|
| No GPU | Designed for CPU: 160 px training, a 16-channel scratch U-Net, a cheaper but fair ablation protocol (50% data, 6 epochs, matched baseline). Colab notebook for GPU reruns. |
| Disk full while extracting the 800 MB archive | Preprocess straight from the `.tar.gz` stream, downscale on the fly (190 MB result), delete the archives. |
| TLS-intercepting network: pip and weight downloads failed with `CERTIFICATE_VERIFY_FAILED` | Use the OS certificate store (`truststore`), both in pip and in code (`use_system_certs()`). |
| Long runs killed (memory pressure from other apps) and slowed (CPU shared with another job) | The experiment runner is resumable (skips finished runs). Training was relaunched as an independent low-priority process, and fewer DataLoader workers lowered RAM use. |
| Validation IoU looked capped around 0.83 | The border ablation revealed a train/eval target mismatch (4.2). |
| Worst test cases looked like failures | Inspection showed most were empty or inverted annotations (5.2). |
| INT8 gave no speed-up | Measured instead of assumed: speed-up appears at 224 px. Reported honestly together with the accuracy cost. |

## 9. Strengths and weaknesses

**Strengths**
- Every decision is backed by an experiment with a controlled comparison: four architectures, five ablations, three calibration methods, five resolutions.
- Small and fast: 3.2M parameters, 6.6 ms per image on a laptop CPU, 3.7 MB as INT8.
- Clean protocol: test set used only at the end, breed-stratified validation, evaluation through the real inference path at native resolution.
- Useful as is: API, demo, CLI and Docker image, with PyTorch not needed at serving time.

**Weaknesses**
- Small, distant pets (IoU 0.73 when the pet is under 15% of the image).
- Coarse edges: 224 px inference upsampled to full resolution blurs fur and whiskers. That is fine for a sticker, but not for professional cut-outs.
- Only cats and dogs, one animal per image as in the dataset. With several pets it returns one combined mask, and it cannot tell instances apart.
- Short CPU training (10 epochs, 160 px). The bigger models were probably under-trained, so the architecture ranking may not hold at a larger budget.
- 14 noisy masks left in the training data.

## 10. With more time and compute

1. Remove the 14 empty or inverted masks from training, and report test IoU on the cleaned subset alongside the official one.
2. Train at 224 to 320 px for 30 to 50 epochs on a GPU (the Colab notebook is ready). Re-rank the architectures. Fine-tune the encoder with a lower learning rate than the head.
3. Better edges: a boundary-aware loss, a light refinement step at full resolution, or soft alpha matting for fur.
4. Small-object recall: multi-scale or tiled inference, or a crop-and-refine second pass around the predicted bbox.
5. Quantisation-aware training to recover the 1.6 points INT8 loses; benchmark on a CPU with VNNI and on mobile (ONNX Runtime Mobile / NNAPI).
6. Instance segmentation (Mask R-CNN-style) for photos with several pets.
7. Out-of-distribution handling: today a photo with no pet can still get a small mask. A pet/no-pet classifier head, or a threshold on predicted area and confidence, would catch it.
