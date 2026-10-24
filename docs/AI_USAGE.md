# Use of AI tools

> Draft. The factual parts below describe how the project was actually built.
> Sections marked **[fill in]** can only be written by me, and must be before submission.

## What was used, and for what

- **Claude Code (Anthropic, Claude Opus 5.5)**, an AI coding agent working in the
  project directory. It:
  - proposed the task framing (binary pet segmentation for background removal) and
    the dataset, which I chose from three options it laid out;
  - wrote the first version of the code: data preparation, models, training loop,
    evaluation, ONNX export and quantisation, API, web demo, tests;
  - ran the experiments on my laptop and collected the numbers into `results/`;
  - drafted this README, the report and this note.
- **[fill in]** any other tools (ChatGPT, Copilot, ...), or "none".

## How the AI output was checked

Mechanical checks built into the repo, which hold regardless of who wrote the code:

- `tests/test_core.py`: metrics are checked against values computed by hand; the loss
  is checked to give zero gradient to ignored pixels; every model is checked to keep
  the input's spatial size on odd-sized inputs; the API is tested end to end on a tiny
  exported model, including a bad upload.
- `petseg/export.py` compares PyTorch and ONNX Runtime outputs after every export
  (max difference ~1e-5).
- The test split is never used for training or model selection; checkpoints are
  chosen on the validation split (`train.py`), and test images are only read by
  `evaluate.py` and the optimisation study.
- Results were checked for plausibility against published numbers for this dataset
  (foreground IoU in the high 0.8s to low 0.9s for pretrained models at higher
  resolution and longer training).

**[fill in]** what I checked myself, for example:
- which files I read line by line and what I changed;
- design decisions I questioned or changed (and why);
- bugs I found in the generated code;
- what I re-ran myself to confirm the reported numbers.

## Where I would not trust it blindly

- Library API details (for example the torchvision v2 transforms' handling of masks,
  ONNX Runtime quantisation options) change between versions; they were verified by
  running the code and the tests here, not taken on faith.
- Explanations of *why* a result happened (for example why INT8 was not faster on
  this CPU) are hypotheses supported by measurements, not proofs.
