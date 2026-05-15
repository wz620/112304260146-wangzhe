"""Train YOLO on competition data (GPU) and write submission.csv next to this script.

Uses yolov8n.yaml so no pretrained .pt download is required (works when GitHub is blocked).
Optional: place weights/yolov8n.pt for transfer learning — see --weights.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "第4次实验数据及提交格式"
DATA_YAML = DATA / "data.yaml"
TEST_IMAGES = DATA / "test" / "images"
SUBMISSION = ROOT / "submission.csv"
WEIGHTS_DIR = ROOT / "weights"
DEFAULT_YAML = "yolov8n.yaml"


def pick_model(weights: Path | None) -> str:
    if weights and weights.is_file():
        return str(weights.resolve())
    return DEFAULT_YAML


def train(weights: Path | None) -> Path:
    assert DATA_YAML.is_file(), DATA_YAML
    os.chdir(DATA)
    model = YOLO(pick_model(weights))
    model.train(
        data=str(DATA_YAML.name),
        epochs=30,
        imgsz=640,
        device=0,
        batch=16,
        workers=4,
        patience=10,
        project=str(ROOT / "runs"),
        name="traffic_signs",
        exist_ok=True,
    )
    best = ROOT / "runs" / "traffic_signs" / "weights" / "best.pt"
    assert best.is_file(), best
    return best


def predict_to_csv(pt: Path) -> None:
    model = YOLO(str(pt))
    image_paths = sorted(p for p in TEST_IMAGES.iterdir() if p.is_file())
    with SUBMISSION.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "class_id",
                "x_center",
                "y_center",
                "width",
                "height",
                "confidence",
            ],
        )
        writer.writeheader()
        for result in model.predict(
            source=[str(p) for p in image_paths],
            conf=0.001,
            save=False,
            verbose=False,
            device=0,
        ):
            image_id = Path(result.path).name
            if result.boxes is None:
                continue
            for box in result.boxes:
                x_center, y_center, width, height = box.xywhn[0].tolist()
                writer.writerow(
                    {
                        "image_id": image_id,
                        "class_id": int(box.cls[0].item()),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": float(box.conf[0].item()),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--infer-only",
        action="store_true",
        help="Skip training; only run test inference with --weights",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional pretrained .pt before training (e.g. weights/yolov8n.pt), "
        "or required for --infer-only (trained best.pt).",
    )
    args = parser.parse_args()

    if args.infer_only:
        if not args.weights or not args.weights.is_file():
            print("Use: python train_and_submit.py --infer-only --weights runs/.../best.pt")
            return 1
        predict_to_csv(args.weights.resolve())
        print("Wrote:", SUBMISSION)
        return 0

    best = train(args.weights)
    predict_to_csv(best)
    print("Wrote:", SUBMISSION)
    print("Best weights:", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
