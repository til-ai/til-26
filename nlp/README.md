# NLP

Your NLP challenge is to answer questions using RAG.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications).


To load the test corpus, the first request sent to your endpoint will be of the following structure:

```JSON
{
  "instances": [
    {
      "documents": [
        "Text of document one.",
        "Text of document two.",
        ...
      ]
    }
  ]
}
```

This is expected to be parsed by your NLP RAG QA system to be used as context for RAG. You can thus do your embedding/chunking/etc on this data. Once your model has completed processing it, return the following:

```JSON
{
  "predictions": ["loaded"]
}
```

This will be taken as the signal that your system is ready to move on to receiving input.

### Input

The input is sent via a POST request to the `/nlp` route on port 5004. It is a JSON document structured as such:

```JSON
{
  "instances": [
    {
      "question": "QUESTION_TEXT"
    },
    ...
  ]
}
```

The `question` key of each object in the `instances` list contains the text of the question to be answered by your NLP RAG QA system. The length of the `instances` list is variable.

### Output

Your route handler function must return a `dict` with this structure:

```Python
{
    "predictions": [
        {"documents": ["DOC-0001", "DOC-0003"], "answer": "Answer one."},
        {"documents": [], "answer": ""},
        ...
    ]
}
```

where each item in `predictions` corresponds to the relevant document IDs and the predicted NLP answer for the corresponding question.

The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where n is the number of input instances. The length of `predictions` must equal that of `instances`.
