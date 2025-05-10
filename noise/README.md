# TODO: Adversarial Noising

An additional component to the CV challenge is for you to adversarially noise images for your competitors to use as inputs to their CV models.

This Readme provides a brief overview of the interface format; see the Wiki for the full [challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications).

## Input

The input is sent via a POST request to the `/noise` route on port 5003. It is a JSON document structured as such:

```JSON
{
  "instances": [
    {
      "key": 0,
      "b64": "BASE64_ENCODED_IMAGE"
    },
    ...
  ]
}
```

The `b64` key of each object in the `instances` list contains the base64-encoded bytes of the input image in JPEG format. The length of the `instances` list is variable.

## Output

Your route handler function must return a `dict` with this structure:

```JSON
{
    "predictions": [
        "BASE_64_ENCODED_IMAGE",
        ...
    ]
}
```

where each string in `predictions` is your adversarially noised version of the corresponding input image.


The $k$-th element of `predictions` must be the prediction corresponding to the $k$-th element of `instances` for all $1 \le k \le n$, where n is the number of input instances. The length of `predictions` must equal that of `instances`.
