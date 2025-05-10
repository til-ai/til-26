"""Manages the noise model."""

import base64
import io

from PIL import Image


class NoiseManager:

    def __init__(self):
        # This is where you can initialize your model and any static configurations.
        # TODO
        pass

    def noise(self, image: bytes) -> str:
        """Performs adversarial noising on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A string containing your output image encoded in base64.
        """

        img = Image.open(io.BytesIO(image))
        try:
            # Your noising code goes here.
            # TODO

            # convert back to b64
            buffered = io.BytesIO()
            Image.fromarray(img).save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("ascii")
        except Exception as e:
            print(f"Error occurred: {e}")
            return base64.b64encode(image).decode("ascii")
