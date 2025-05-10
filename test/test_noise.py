import base64
import gc
import io
import json
import logging
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests
from dotenv import load_dotenv
from noise_eval.fairness_checker import FairnessChecker
from noise_eval.pipeline import EvalPipeline
from PIL import Image
from test_utils import batched
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 4

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")
USE_CUDA = True  # Set to False to disable GPU-accelerated SSIM (if CUDA is unavailable or causing issues)

FAIRNESS_CONFIG = Path(__file__).parent / "noise_eval" / "eval_thresholds_v2.yaml"


def convert_to_np_hwc(b64: str) -> np.ndarray:
    # Decode base64 string
    img_bytes = base64.b64decode(b64)

    # Convert to numpy HWC uint8 array
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_array = np.array(img, dtype=np.uint8)  # Shape: (H, W, C)
    return img_array


def sample_generator(
    instances: Sequence[Mapping[str, Any]],
    data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / "images" / instance["file_name"], "rb") as img_file:
            img_data = img_file.read()
        yield {
            "key": instance["id"],
            "b64": base64.b64encode(img_data).decode("ascii"),
        }


def score_noise(
    preds: Sequence[str | None],
    instances: Sequence[Path],
) -> tuple[float, float]:
    print("Scoring noise...")
    np_noised = [convert_to_np_hwc(pred) if pred else None for pred in preds]
    del preds  # free memory
    gc.collect()

    np_original = [
        np.array(Image.open(instance).convert("RGB"), dtype=np.uint8)
        for instance in instances
    ]

    # Build per-image bbox lists from COCO annotations.
    # instances are paths like data_dir/filename, so parent is data_dir.
    data_dir = instances[0].parent.parent
    with open(data_dir / "annotations.json") as f:
        annotations = json.load(f)

    filename_to_image_id = {
        img["file_name"]: img["id"] for img in annotations["images"]
    }
    anns_by_image_id: dict[int, list] = {}
    for ann in annotations.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in anns_by_image_id:
            anns_by_image_id[img_id] = []
        anns_by_image_id[img_id].append(ann["bbox"])

    boxes_list = []
    for path in instances:
        image_id = filename_to_image_id.get(path.name)
        bboxes = anns_by_image_id.get(image_id, []) if image_id is not None else []
        boxes_list.append(np.array(bboxes) if bboxes else np.zeros((0, 4)))

    pipeline = EvalPipeline(use_cuda=USE_CUDA)
    summary = pipeline.evaluate_batched_with_boxes(np_original, np_noised, boxes_list)
    print("Noise evaluation summary:")
    print(summary)

    # Image-level fairness check: evaluate all images at once
    checker = FairnessChecker(FAIRNESS_CONFIG)
    per_image_metrics = [img_report.to_dict() for img_report in summary.per_image]
    results = [checker.evaluate(metrics) for metrics in per_image_metrics]
    num_passing = sum(1 for r in results if r.passed)
    num_images = len(results)
    pass_rate = num_passing / num_images if num_images > 0 else 0.0
    print(
        f"\nImage-level fairness: {num_passing}/{num_images} images pass ({pass_rate * 100:.1f}%)"
    )

    return pass_rate


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/cv")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # load images
    with open(data_dir / "annotations.json", "r") as f:
        annotations = json.load(f)
    instances = annotations["images"][:500]  # limit to 500 images for testing

    batch_generator = batched(sample_generator(instances, data_dir), n=BATCH_SIZE)

    output_images = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        response = requests.post(
            "http://localhost:5003/noise",
            data=json.dumps(
                {
                    "instances": batch,
                }
            ),
        )
        output_images.extend(response.json()["predictions"])

    score = score_noise(
        output_images,
        [data_dir / "images" / instance["file_name"] for instance in instances],
    )
    print("Noise Score (Pass Rate):", score)


if __name__ == "__main__":
    main()
