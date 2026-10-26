"""Train one experiment.

    python -m petseg.train configs/lraspp_mbv3.yaml [--epochs 2 --limit 64]

Writes runs/<name>/{best.pt, history.csv, summary.json}. "best" = highest
validation foreground IoU; the test split is never touched here.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from .data import PetSegDataset
from .losses import SegLoss
from .metrics import ConfusionMeter
from .models import build_model
from .utils import ROOT, count_params, load_config, save_json, seed_everything, use_system_certs


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = ConfusionMeter()
    for x, y in loader:
        logits = model(x.to(device))
        meter.update(logits.argmax(1).cpu(), y)
    return meter.compute()


def train(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    use_system_certs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.get("threads"):
        torch.set_num_threads(cfg["threads"])
    out = ROOT / "runs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)

    train_ds = PetSegDataset("train", cfg["size"], train=True, augment=cfg["augment"],
                             border=cfg["border"], limit=cfg.get("limit"))
    val_ds = PetSegDataset("val", cfg["size"], border="foreground", limit=cfg.get("limit"))
    if cfg.get("train_fraction", 1.0) < 1.0:
        # Random (not alphabetical, which would drop whole breeds) subset of the
        # training images; used by the cheaper ablation protocol. Validation stays full.
        g = torch.Generator().manual_seed(cfg["seed"])
        n = int(len(train_ds) * cfg["train_fraction"])
        train_ds = Subset(train_ds, torch.randperm(len(train_ds), generator=g)[:n].tolist())
    loader_kw = dict(num_workers=cfg["num_workers"], persistent_workers=cfg["num_workers"] > 0,
                     pin_memory=device.type == "cuda")
    train_dl = DataLoader(train_ds, cfg["batch_size"], shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, cfg["batch_size"] * 2, **loader_kw)

    model = build_model(cfg["model"], pretrained=cfg["pretrained"]).to(device)
    criterion = SegLoss(cfg["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(train_dl)
    scheduler = cosine_with_warmup(optimizer, warmup_steps=len(train_dl), total_steps=steps)
    print(f"{cfg['name']}: {cfg['model']} {count_params(model) / 1e6:.2f}M params, "
          f"{len(train_ds)} train / {len(val_ds)} val, {device}")

    history, best = [], {"fg_iou": -1.0}
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(x)
            n += len(x)
        train_time = time.time() - t0
        val = validate(model, val_dl, device)
        row = {"epoch": epoch, "train_loss": total / n, "lr": scheduler.get_last_lr()[0],
               "epoch_sec": train_time, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        flag = ""
        if val["fg_iou"] > best["fg_iou"]:
            best = {**val, "epoch": epoch}
            torch.save({"model": cfg["model"], "state_dict": model.state_dict(), "config": cfg,
                        "val": val, "epoch": epoch}, out / "best.pt")
            flag = " *"
        print(f"  ep {epoch:2d} loss {row['train_loss']:.4f} val fgIoU {val['fg_iou']:.4f} "
              f"mIoU {val['miou']:.4f} ({train_time:.0f}s){flag}", flush=True)

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    summary = {"name": cfg["name"], "model": cfg["model"], "params": count_params(model),
               "best_val": best, "config": cfg,
               "train_minutes": sum(h["epoch_sec"] for h in history) / 60, "device": str(device)}
    save_json(summary, out / "summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="use only N images per split (smoke tests)")
    ap.add_argument("--size", type=int)
    ap.add_argument("--name")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("name", Path(args.config).stem)
    for k in ("epochs", "limit", "size", "name"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    train(cfg)


if __name__ == "__main__":
    main()
