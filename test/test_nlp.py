import json
import logging
import math
import os
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from time import sleep
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from test_utils import batched
from tqdm import tqdm, trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv()
TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")

BATCH_SIZE = 4


@dataclass
class AEResult:
    index: int
    equivalent: bool
    confidence: float
    prob_equivalent: float

    def to_dict(self) -> dict:
        return {
            "equivalent": self.equivalent,
            "confidence": self.confidence,
            "prob_equivalent": round(self.prob_equivalent, 4),
        }


class AnswerEquivalenceEvaluator:
    """
    Wraps a fine-tuned encoder for answer-equivalence inference.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved checkpoint directory (containing config.json,
        model.safetensors / pytorch_model.bin, tokenizer files).
    threshold : float
        Decision boundary for "equivalent." Default 0.5.
        Raise for stricter evaluation (higher precision), lower for
        more lenient (higher recall).
    device : str or None
        "cuda", "cpu", or None for auto-detect.
    max_length : int
        Max token length. Longer inputs will be truncated. Default 128.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        device: str | None = None,
        max_length: int = 128,
    ):
        self.threshold = threshold
        self.max_length = max_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Loading model from {model_path} on {self.device}")
        # check if model path exists, else copy from /home/jupyter/TEAM_TRACK/nlp/models/
        if not Path(model_path).exists():
            logger.info(
                f"Model path {model_path} does not exist, copying from /home/jupyter/{TEAM_TRACK}/nlp/models"
            )
            existing_model_path = (
                Path(f"/home/jupyter/{TEAM_TRACK}/nlp/models")
                / f"{Path(model_path).name}.zip"
            )
            if not existing_model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {model_path} or {existing_model_path}"
                )

            # extract to local directory
            with zipfile.ZipFile(existing_model_path, "r") as zip_ref:
                zip_ref.extractall(Path(model_path).parent)
            logger.info(f"Extracted model to {Path(model_path).parent}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path)
        ).to(self.device)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {n_params:,} parameters")

    def _format_input(self, question: str, reference: str, candidate: str) -> str:
        return (
            f"Question: {question} "
            f"Reference: {reference} "
            f"Candidate: {candidate}"
        )

    @torch.no_grad()
    def __call__(self, question: str, reference: str, candidate: str) -> AEResult:
        """Score a single (question, reference, candidate) triple."""
        text = self._format_input(question, reference, candidate)
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoding).logits
        probs = F.softmax(logits, dim=-1).squeeze(0)
        prob_eq = probs[1].item()

        return AEResult(
            index=0,
            equivalent=prob_eq >= self.threshold,
            confidence=max(prob_eq, 1 - prob_eq),
            prob_equivalent=prob_eq,
        )

    @torch.no_grad()
    def batch_evaluate(
        self,
        triples: list[tuple[str, str, str]],
        batch_size: int = 64,
    ) -> list[AEResult]:
        """
        Score a list of (question, reference, candidate) triples.

        Returns results in the same order as input.
        """
        # if either string is empty, check if the other is empty or not
        empty_str_results = []
        non_empty_indexed_triples = []

        for i, (q, r, c) in enumerate(triples):
            if r == "" or c == "":
                empty_str_results.append(
                    AEResult(
                        index=i,
                        equivalent=(r == c),
                        confidence=1.0,
                        prob_equivalent=1.0 if r == c else 0.0,
                    )
                )
            else:
                non_empty_indexed_triples.append((i, q, r, c))

        # otherwise, pass the rest onto the model as usual

        texts = [
            (i, self._format_input(q, r, c)) for i, q, r, c in non_empty_indexed_triples
        ]
        all_results = []

        for i in trange(0, len(texts), batch_size):
            batch_indices, batch_texts = zip(*texts[i : i + batch_size])
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**encoding).logits
            probs = F.softmax(logits, dim=-1)

            for prob_idx, prob in enumerate(probs):
                prob_eq = prob[1].item()
                all_results.append(
                    AEResult(
                        index=batch_indices[prob_idx],
                        equivalent=prob_eq >= self.threshold,
                        confidence=max(prob_eq, 1 - prob_eq),
                        prob_equivalent=prob_eq,
                    )
                )

        # combine and reorder results
        all_results.extend(empty_str_results)
        all_results.sort(key=lambda r: r.index)

        return all_results

    def aggregate_score(self, results: list[AEResult]) -> dict:
        """
        Compute summary statistics over a batch of results.
        """
        n = len(results)
        if n == 0:
            return {"n": 0, "equiv_rate": 0.0, "mean_prob": 0.0}

        equiv_count = sum(1 for r in results if r.equivalent)
        mean_prob = sum(r.prob_equivalent for r in results) / n

        return {
            "n": n,
            "equiv_rate": round(equiv_count / n, 4),
            "mean_prob": round(mean_prob, 4),
            "equivalent_count": equiv_count,
            "not_equivalent_count": n - equiv_count,
        }


evaluator = AnswerEquivalenceEvaluator(
    model_path=f"./test/models/nlp_eval_512",
    threshold=0.5,
    device=None,
    max_length=512,
)


def poll_endpoint_for_loading(max_retries=None, delay_sec=10):
    retry_num = 0
    while max_retries is None or retry_num < max_retries:
        try:
            response = requests.post(
                "http://localhost:5004/nlp",
                data=json.dumps({"instances": [{"poll": "true"}]}),
            ).json()["predictions"]
            if len(response) == 1 and response[0] == "loaded":
                print("Model server is loaded.")
                return True
            elif len(response) == 1 and response[0] == "error":
                print("Model server is reporting an error.")
                return False
            elif len(response) == 1 and response[0] == "loading":
                print(f"Retry {retry_num}: Model server is still loading the corpus.")
        except Exception as e:
            print(f"Error occurred while polling endpoint: {e}")
        sleep(delay_sec)
        retry_num += 1
    return False


def sample_generator(
    instances: Sequence[Mapping[str, Any]],
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        yield {"question": instance["question"]}


def score_nlp(preds: Sequence[str], ground_truth: Sequence[Mapping[str, Any]]) -> float:
    triples = []
    for pred, gt in zip(preds, ground_truth):
        triples.append(
            (gt["question"], gt["answer"] if gt["answer"] is not None else "", pred)
        )

    results = evaluator.batch_evaluate(triples)
    summary = evaluator.aggregate_score(results)
    logger.info(f"Answer Equivalence Evaluation Summary: {summary}")
    return summary["equiv_rate"]


def main():
    data_dir = Path(f"/home/jupyter/{TEAM_TRACK}/nlp")
    results_dir = Path(f"/home/jupyter/{TEAM_NAME}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # load all questions
    with open(data_dir / "nlp.jsonl") as f:
        instances = [json.loads(line.strip()) for line in f if line.strip()]

    # send corpus to model server
    documents_dir = data_dir / "documents"
    doc_contents = []
    for doc_file in documents_dir.glob("*.txt"):
        with open(doc_file, "r") as f:
            content = f.read()
            doc_contents.append(content)
    response = requests.post(
        "http://localhost:5004/nlp",
        data=json.dumps({"instances": [{"documents": doc_contents}]}),
    )

    # verify response to make sure server is healthy and loaded the corpus
    if response.status_code != 200 or response.json()["predictions"][0] == "error":
        logger.error(f"Failed to load corpus: {response.text}")
        return
    elif response.status_code == 200 or response.json()["predictions"][0] == "loading":
        logger.info("Corpus load initiated, polling for completion...")
        if not poll_endpoint_for_loading(max_retries=30, delay_sec=10):
            logger.error("Corpus failed to load within expected time.")
            return
    else:
        logger.info("Corpus loaded successfully, proceeding with QA evaluation")

    batch_generator = batched(sample_generator(instances), n=BATCH_SIZE)

    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        response = requests.post(
            "http://localhost:5004/nlp",
            data=json.dumps(
                {
                    "instances": batch,
                }
            ),
        )
        results.extend(response.json()["predictions"])

    results_path = results_dir / "nlp_results.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)

    score = score_nlp(results, instances)
    print("QA Accuracy:", score)


if __name__ == "__main__":
    main()
