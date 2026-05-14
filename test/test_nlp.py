import json
import logging
import math
import os
import sys
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import printable
from time import sleep
from typing import Any

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
RETRIEVAL_ONLY_SCORE = 0.4
MAX_CANDIDATE_TOKEN_LENGTH = 64


@dataclass
class AEResult:
    index: int
    score: float
    equivalent: bool
    prob_equivalent: float

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "equivalent": self.equivalent,
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
        _printable = "".join(filter(lambda x: x in printable, candidate))

        # truncate to required max length, then re-encode as text
        tokens = self.tokenizer.tokenize(
            _printable, max_length=MAX_CANDIDATE_TOKEN_LENGTH, truncation=True
        )
        reconstructed_candidate = self.tokenizer.convert_tokens_to_string(tokens)

        return (
            f"Question: {question} "
            f"Reference: {reference} "
            f"Candidate: {reconstructed_candidate}"
        )

    @torch.no_grad()
    def batch_evaluate(
        self,
        data: list[tuple[list[str], list[str], str, str, str]],
        batch_size: int = 64,
    ) -> list[AEResult]:
        """
        Score a list of (question, reference, candidate) triples.

        Returns results in the same order as input.
        """
        # if either string is empty, check if the other is empty or not
        empty_str_results = []
        non_empty_indexed_triples = []

        for i, (docs, pred_docs, q, r, c) in enumerate(data):
            overlap_docs = len(set(docs).intersection(set(pred_docs))) >= 1
            if len(docs) == 0 and len(pred_docs) == 0 and r == "" and c == "":
                # all empty = L4; mark correct
                empty_str_results.append(
                    AEResult(
                        index=i,
                        score=1.0,
                        equivalent=True,
                        prob_equivalent=1.0,
                    )
                )
            elif (r == "" or c == "") and overlap_docs:
                # L5: requires document overlap + empty string due to false premise
                _equivalent = r == c
                empty_str_results.append(
                    AEResult(
                        index=i,
                        score=1.0 if _equivalent else RETRIEVAL_ONLY_SCORE,
                        equivalent=_equivalent,
                        prob_equivalent=1.0 if _equivalent else 0.0,
                    )
                )
            elif overlap_docs:
                # if there's at least one top3 overlap between the documents, go to next stage
                non_empty_indexed_triples.append((i, q, r, c))
            else:
                # document retrieval failure, no points
                empty_str_results.append(
                    AEResult(
                        index=i,
                        score=0.0,
                        equivalent=False,
                        prob_equivalent=0.0,
                    )
                )
        # pass the rest onto the model for evaluation
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
                _equivalent = prob_eq >= self.threshold
                all_results.append(
                    AEResult(
                        index=batch_indices[prob_idx],
                        score=1.0 if _equivalent else RETRIEVAL_ONLY_SCORE,
                        equivalent=_equivalent,
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
            return {
                "n": 0,
                "equiv_rate": 0.0,
                "mean_prob": 0.0,
                "equivalent_count": 0,
                "not_equivalent_count": 0,
            }

        equiv_count = sum(r.score for r in results)
        mean_prob = sum(r.prob_equivalent for r in results) / n

        return {
            "n": n,
            "equiv_rate": round(equiv_count / n, 3),
            "mean_prob": round(mean_prob, 3),
            "equivalent_count": equiv_count,
            "not_equivalent_count": n - equiv_count,
        }


evaluator = AnswerEquivalenceEvaluator(
    model_path=f"./test/models/nlp_eval_512",
    threshold=0.9,
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
            if len(response) == 1 and response[0].get("status") == "loaded":
                print("Model server is loaded.")
                return True
            elif len(response) == 1 and response[0].get("status") == "error":
                print("Model server is reporting an error.")
                return False
            elif len(response) == 1 and response[0].get("status") == "loading":
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


def score_nlp(
    preds: Sequence[dict[str, list[str] | str]],
    ground_truth: Sequence[Mapping[str, Any]],
) -> float:
    data = []

    for pred, gt in zip(preds, ground_truth):
        data.append(
            (
                gt["source_docs"],
                pred["documents"][:3],
                gt["question"],
                gt["answer"] if gt["answer"] is not None else "",
                pred["answer"],
            )
        )

    results = evaluator.batch_evaluate(data)
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
            doc = {
                "id": doc_file.stem,
                "document": content,
            }
            doc_contents.append(doc)
    response = requests.post(
        "http://localhost:5004/nlp",
        data=json.dumps({"instances": [{"documents": doc_contents}]}),
    )

    # verify response to make sure server is healthy and loaded the corpus
    if (
        response.status_code != 200
        or response.json()["predictions"][0].get("status") == "error"
    ):
        logger.error(f"Failed to load corpus: {response.text}")
        return
    elif (
        response.status_code == 200
        or response.json()["predictions"][0].get("status") == "loading"
    ):
        logger.info("Corpus load initiated, polling for completion...")
        if not poll_endpoint_for_loading(max_retries=30, delay_sec=10):
            logger.error("Corpus failed to load within expected time.")
            return
    else:
        logger.info("Corpus loaded successfully, proceeding with QA evaluation")

    batch_generator = batched(sample_generator(instances), n=BATCH_SIZE)

    results: list[dict[str, list[str] | str]] = []
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
    print("NLP RAG QA Accuracy:", score)


if __name__ == "__main__":
    main()
