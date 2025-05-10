"""Runs the NLP server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


from fastapi import FastAPI, Request
from nlp_manager import NLPManager

app = FastAPI()
manager = NLPManager()


@app.post("/nlp")
async def nlp(request: Request) -> dict[str, list[str]]:
    """Performs NLP RAG QA tasks.

    Args:
        request: The API request. Contains a list of questions.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str` answers, in the same order as which appears in `request`.
    """

    inputs_json = await request.json()

    # Load the corpus if it hasn't been loaded yet
    if not manager.loaded and inputs_json["instances"][0].get("documents") is not None:
        manager.load_corpus(inputs_json["instances"][0]["documents"])
        return {"predictions": ["loaded" if manager.loaded else "failed"]}

    predictions = []
    for instance in inputs_json["instances"]:

        # Reads the question from the request.
        question = instance["question"]

        # Performs NLP QA and appends the result.
        answer = manager.qa(question)
        predictions.append(answer)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
