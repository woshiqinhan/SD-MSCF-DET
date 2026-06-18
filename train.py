"""Train the SD-MSCF-DET multimodal detector."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from ultralytics import YOLOMM


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "configs" / "SD-MSCF-DET.yaml")
    parser.add_argument("--data", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="CUDA device, e.g. 0 or 0,1; omit for automatic selection")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="SD-MSCF-DET")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLOMM(str(args.model))
    train_args = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "imgsz": args.imgsz,
        "amp": args.amp,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
    }
    if args.device is not None:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()
