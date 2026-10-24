"""Run every experiment config in sequence, then build the comparison table.

    python scripts/run_experiments.py              # all configs
    python scripts/run_experiments.py lraspp_mbv3  # just some

Runs that already have runs/<name>/summary.json are skipped, so the script
can be restarted after an interruption.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORDER = [
    # architecture comparison
    "lraspp_mbv3", "deeplabv3_mbv3", "unet_resnet18", "unet_scratch",
    # ablations on the LR-ASPP baseline (one change each)
    "ablation_no_pretrain", "ablation_loss_ce", "ablation_no_aug", "ablation_border_fg",
]


def main():
    names = sys.argv[1:] or ORDER
    for name in names:
        if (ROOT / "runs" / name / "summary.json").exists():
            print(f"skip {name} (done)")
            continue
        subprocess.run([sys.executable, "-m", "petseg.train", f"configs/{name}.yaml"],
                       cwd=ROOT, check=True)
    subprocess.run([sys.executable, "scripts/summarize.py"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
