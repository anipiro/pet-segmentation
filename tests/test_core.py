import numpy as np
import pytest
import torch
from PIL import Image

from petseg.data import IGNORE, trimap_to_label
from petseg.inference import Predictor, softmax_fg
from petseg.losses import SegLoss
from petseg.metrics import ConfusionMeter, image_iou
from petseg.models import MODELS, build_model


def test_trimap_mapping():
    t = np.array([[1, 2, 3]], dtype=np.uint8)
    assert trimap_to_label(t, "ignore").tolist() == [[1, 0, IGNORE]]
    assert trimap_to_label(t, "foreground").tolist() == [[1, 0, 1]]


def test_confusion_meter_matches_hand_computation():
    target = torch.tensor([[1, 1, 0, 0]])
    pred = torch.tensor([[1, 0, 0, 1]])
    m = ConfusionMeter()
    m.update(pred, target)
    r = m.compute()
    # fg: tp=1, fp=1, fn=1 -> IoU 1/3, Dice 2/4
    assert r["fg_iou"] == pytest.approx(1 / 3)
    assert r["dice"] == pytest.approx(0.5)
    assert r["pixel_acc"] == pytest.approx(0.5)


def test_confusion_meter_skips_ignore():
    m = ConfusionMeter()
    m.update(torch.tensor([[1, 0]]), torch.tensor([[1, IGNORE]]))
    assert m.cm.sum() == 1


def test_image_iou_empty_is_perfect():
    z = np.zeros((4, 4), np.uint8)
    assert image_iou(z, z) == 1.0


def test_loss_ignores_pixels():
    logits = torch.randn(1, 2, 4, 4, requires_grad=True)
    target = torch.full((1, 4, 4), IGNORE)
    target[0, 0, 0] = 1
    loss = SegLoss("ce_dice")(logits, target)
    loss.backward()
    # only the single labelled pixel receives gradient
    assert (logits.grad.abs().sum(1)[0] > 0).sum() == 1


@pytest.mark.parametrize("name", MODELS)
def test_models_keep_spatial_size(name):
    m = build_model(name, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 100, 132))  # deliberately not divisible by 32
    assert out.shape == (1, 2, 100, 132)


def test_softmax_fg_is_two_class_softmax():
    logits = np.random.randn(1, 2, 3, 3).astype(np.float32)
    ref = torch.from_numpy(logits).softmax(1)[0, 1].numpy()
    assert np.allclose(softmax_fg(logits), ref, atol=1e-6)


@pytest.fixture(scope="session")
def tiny_onnx(tmp_path_factory):
    path = tmp_path_factory.mktemp("m") / "tiny.onnx"
    model = build_model("unet_scratch", pretrained=False).eval()
    torch.onnx.export(model, torch.randn(1, 3, 64, 64), path, input_names=["image"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    return path


def test_predictor_outputs_original_size(tiny_onnx):
    p = Predictor(tiny_onnx)
    assert p.size == 64
    img = Image.new("RGB", (123, 77), (200, 100, 50))
    r = p.predict(img)
    assert r.prob.shape == (77, 123)
    assert r.mask.dtype == np.uint8 and set(np.unique(r.mask)) <= {0, 1}
    assert r.overlay().size == (123, 77)
    assert r.cutout(crop=False).mode == "RGBA"


def test_api(tiny_onnx, monkeypatch):
    import io

    monkeypatch.setenv("PETSEG_MODEL", str(tiny_onnx))
    import importlib

    import app.main
    importlib.reload(app.main)
    from fastapi.testclient import TestClient

    client = TestClient(app.main.app)
    assert client.get("/health").json()["input_size"] == 64

    buf = io.BytesIO()
    Image.new("RGB", (90, 60), (10, 20, 30)).save(buf, format="JPEG")
    files = {"file": ("x.jpg", buf.getvalue(), "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"]) == (90, 60)
    assert body["mask_png"]

    r = client.post("/cutout", files=files)
    assert r.headers["content-type"] == "image/png"

    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
