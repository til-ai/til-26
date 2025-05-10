"""Runs the adversarial noising server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64

from fastapi import FastAPI, Request
from noise_manager import NoiseManager

app = FastAPI()
manager = NoiseManager()


@app.post("/noise")
async def noise(request: Request) -> dict[str, list[str]]:
    """Performs adversarial noising on image frames.

    Args:
        request: The API request. Contains a list of images, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `str`s containing your adversarially noised outputs, in the same order as
        which appears in `request`. See `noise/README.md` for the expected format.
    """

    inputs_json = await request.json()

    predictions = []
    for instance in inputs_json["instances"]:

        # Reads the base-64 encoded image and decodes it into bytes.
        image_bytes = base64.b64decode(instance["b64"])

        # Performs adversarial noising and appends the result.
        noised_image = manager.noise(image_bytes)
        predictions.append(noised_image)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
