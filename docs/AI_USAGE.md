# Use of AI tools

## What was used, and for what

**Claude Code** (Anthropic's coding agent, model Claude Opus 5.5) was used throughout,
working directly in the project directory on my laptop. It:

- laid out three candidate directions (pet segmentation, pedestrian detection,
  road-scene detection) with trade-offs; I chose pet segmentation and local CPU
  training with a Colab fallback;
- wrote the code: data preparation, models, training loop, losses and metrics,
  evaluation and error analysis, ONNX export and quantisation, benchmark, API,
  web demo, tests, Dockerfile, Colab notebook;
- ran the experiments on my machine and collected the numbers into `results/`;
- drafted the README and the technical report from those result files.

## How the output was checked

The guiding rule was that no claim goes into the report unless a file in
`results/` produced by the code supports it. Concretely:

- **Unit and integration tests** (`tests/test_core.py`, 12 tests): metric values
  against numbers computed by hand; the loss gives zero gradient to ignored pixels;
  every model keeps the input size on odd-sized inputs; softmax/probability
  handling; the API end to end on a tiny exported model, including an invalid upload.
- **Export correctness**: every ONNX export is compared against PyTorch on the same
  input (max |Δ| about 1e-6), and re-running the documented export command reproduced
  the shipped `models/petseg.onnx` exactly.
- **Data leakage**: the test split is read only by `evaluate.py` and the
  optimisation study; checkpoints and all choices use the validation split.
- **Results that looked too good or too bad were inspected, not accepted.** The
  +5.1 IoU from the border ablation was traced to a train/eval target mismatch
  before being used for the final model. The worst test predictions were viewed
  image by image, which revealed empty ground-truth masks rather than model errors.
  The expectation that INT8 would be faster was measured, found false at 160 px and
  true at 224 px, and reported that way.
- **Serving claims were run, not assumed**: the API was started and called with
  real images, the demo page was rendered and checked at desktop and narrow widths,
  and the "no PyTorch needed to serve" claim was tested in a clean virtual
  environment with only `requirements-serve.txt`.
- Plausibility against literature: published results on this dataset reach foreground
  IoU of about 0.9 with pretrained backbones, in line with 0.89 to 0.90 here.

## Where AI output needs care

- Library details (torchvision v2 mask transforms, ONNX Runtime quantisation
  options, Pillow deprecations) change between versions. They were confirmed by
  running the code on the pinned versions in `requirements.txt`.
- Explanations of *why* something happened (for example why 224 px inference beats the
  160 px training size, or why INT8 did not speed up at 160 px) are hypotheses that
  fit the measurements, and the report labels them as such.
