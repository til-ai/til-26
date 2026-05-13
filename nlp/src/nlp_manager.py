"""Manages the NLP model."""


class NLPManager:
    loaded = False

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        # TODO
        pass

    def load_corpus(self, documents: list[str]) -> None:
        """Loads the corpus of documents for RAG QA."""
        # Your corpus loading code goes here.
        # TODO
        self.loaded = True

    def qa(self, question: str) -> dict[str, list[str] | str]:
        """Performs question answering on an image of a document.

        Args:
            question: The question to answer.

        Returns:
            A dictionary with two keys:
            - "documents": list of strings containing the most relevant document ids. Only the first 3 will be considered
            - "answer": string containing the answer to the question.
        """

        # Your inference code goes here.
        # TODO

        return {"documents": [], "answer": ""}
