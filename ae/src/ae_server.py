"""Runs the AE server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


from ae_manager import AEManager
from fastapi import FastAPI, Request

app = FastAPI()
manager = AEManager()


@app.post("/ae")
async def ae(request: Request) -> dict[str, list[dict[str, int]]]:
    """Feeds an observation into the AE model.

    Returns action taken given current observation (int)
    """

    # get observation, feed into model
    input_json = await request.json()

    predictions = []
    # each is a dict with one key "observation" and the value as a dictionary observation
    for instance in input_json["instances"]:
        observation = instance["observation"]
        # reset environment on a new round
        if observation["step"] == 0:
            await reset({})
        predictions.append({"action": manager.ae(observation)})
    return {"predictions": predictions}


@app.post("/reset")
async def reset(_: Request) -> None:
    """Resets the `AEManager` for a new round."""

    # The Docker container is not restarted between rounds (during Qualifiers).
    # Your model is reset via this endpoint by creating a new instance. You
    # should avoid storing persistent state information outside your
    # `AEManager` instance; but if you must, you should also reset it here.

    global manager  # pylint: disable=global-statement
    manager = AEManager()

    return


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}
