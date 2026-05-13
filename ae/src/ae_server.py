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
        # You will have to do your own internal counting and reset your own system between rounds!
        # if observation["step"] == 0:
            # do internal resetting here
        predictions.append({"action": manager.ae(observation)})
    return {"predictions": predictions}


# ------------------------------ RESET REMOVED ------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Health check function for your model."""
    return {"message": "health ok"}
