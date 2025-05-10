import base64
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import jiwer
import requests
from dotenv import load_dotenv
from test_utils import batched
from tqdm import tqdm

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 4
print("imported successfully")

wer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " ", "—": " ", "–": " "}),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

cer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": "", "—": "", "–": ""}),
        jiwer.RemoveWhiteSpace(replace_by_space=False),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)


def sample_generator(
    instances: Sequence[Mapping[str, Any]],
    data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / instance["audio"], "rb") as audio_file:
            audio_bytes = audio_file.read()
        yield {
            "key": instance["key"],
            "b64": base64.b64encode(audio_bytes).decode("ascii"),
        }


def score_asr(ground_truth: list[tuple[str, str]], preds: list[str]) -> float:
    language_pred_gt_mapping = {
        lang: {
            "hypothesis": [],
            "reference": [],
            "scorer": (
                partial(
                    jiwer.cer,
                    reference_transform=cer_transforms,
                    hypothesis_transform=cer_transforms,
                )
                if lang == "chinese"
                else partial(
                    jiwer.wer,
                    reference_transform=wer_transforms,
                    hypothesis_transform=wer_transforms,
                )
            ),
        }
        for lang in ["english", "chinese", "malay", "tamil"]
    }
    for pred, (gt, lang) in zip(preds, ground_truth):
        language_pred_gt_mapping[lang]["hypothesis"].append(pred)
        language_pred_gt_mapping[lang]["reference"].append(gt)
    # take average score of all 4 languages
    language_error_rates = {
        lang: lang_dict["scorer"](lang_dict["reference"], lang_dict["hypothesis"])
        for lang, lang_dict in language_pred_gt_mapping.items()
    }
    for lang, error_rate in language_error_rates.items():
        print(
            f"{lang} error rate ({'CER' if lang == 'chinese' else 'WER'}): {error_rate:.4f}"
        )
    mean_error_rate = (
        sum(language_error_rates[lang] for lang in language_pred_gt_mapping.keys()) / 4
    )

    return max(1 - mean_error_rate, 0)


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/asr")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "asr.jsonl") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]

    batch_generator = batched(sample_generator(instances, data_dir), n=BATCH_SIZE)

    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        response = requests.post(
            "http://localhost:5001/asr",
            data=json.dumps(
                {
                    "instances": batch,
                }
            ),
        )
        results.extend(response.json()["predictions"])

    results_path = results_dir / "asr_results.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)

    ground_truths = [
        (instance["transcript"], instance["language"]) for instance in instances
    ]
    score = score_asr(ground_truths, results)
    print("1 - MER:", score)


if __name__ == "__main__":
    main()
