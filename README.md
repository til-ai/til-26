# DSTA BrainHack TIL-AI 2026

**Contents**
1. [DSTA BrainHack TIL-AI 2026](#dsta-brainhack-til-ai-2026)
   1. [Get started](#get-started)
   2. [Understanding this repo](#understanding-this-repo)
   3. [Build, test, and submit](#build-test-and-submit)
      1. [Build](#build)
      2. [Test](#test)
      3. [Submit](#submit)
   4. [Links](#links)

## Get started

Here's a quick overview of the initial setup instructions. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-26/wiki).

Use this repository as a template to create your own, and clone it into your GCP Workbench instance. You'll want to keep your repository private, so you'll need to [create a GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens).

You'll also need to initialize the Git submodules:

```bash
git submodule update --init
```

This repository requires Python 3.10 or newer to work. While it should theoretically all work fine with all packages installed directly into your base Python environment, it is likely a best practice to you create isolated virtual environments for each task. You can use any tool you'd like to do this; [`virtualenv`](https://virtualenv.pypa.io/en/latest/), [`venv`](https://docs.python.org/3/library/venv.html), [`poetry`](https://python-poetry.org/), etc. The competition instance on GCP comes with [`conda`](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) installed, allowing you to create and activate a new virtual environment with the following steps:

```bash
conda create --name til-asr python=3.13
conda activate til-asr
```

Finally, install the development dependencies into your newly created virtual environment.

```bash
pip install -r requirements-dev.txt
```

You should also considering using [`uv`](https://docs.astral.sh/uv/) for python versioning and dependency management all in one.

## Understanding this repo

There's a subdirectory for each challenge: [`asr/`](/asr), [`cv/`](/cv) and its subcategory [`noise/`](/noise), [`nlp/`](/nlp), and [`ae/`](/ae). Each contains:

* A `src/` directory, where your code lives.
  * `*_manager.py`, which manages your model. This is where your inference and computation takes place.
  * `*_server.py`, which runs a local web server that talks to the rest of the competition infrastructure.
* `Dockerfile`, which is used to build your Docker image for each model.
* `requirements.txt`, which lists the dependencies you need to have bundled into your Docker image.
* `README.md`, which contains specifications for the format of each challenge.

You should also see another subdirectory, [`test/`](/test). This contains tools to test and score your model locally, and are automatically run when you use the `til test TASK` command on your GCP Workbench instance.

There are also two Git submodules, `til-26-finals` and `til-26-ae`. `til-26-finals` contains code that will be pulled into your repo for Semifinals and Finals. `til-26-ae` contains the `til_environment` package, which will allow you to train and test your AE model, and is installed by `pip` during setup. Don't delete or modify the contents of `til-26-finals/`, `til-26-ae/`, or `.gitmodules`.

## Build, test, and submit
Submitting your model for evaluation is simple: just build your Docker image, test it, and submit. You can find a more detailed tutorial, including advanced usage for power users, in the [Wiki](https://github.com/til-ai/til-26/wiki).

On the GCP Workbench instance, your environments come pre-set up with a command line utility `til` that will help you build, test, and submit your trained model containers. If you encounter any issues, look through [#hackoverflow](https://discord.com/channels/1488845200523661454/1488845611032903691) on Discord to see if anyone has encountered your problem; if not, post a new question.

tl;dr:
```bash
til build asr
til test asr
til submit asr
```
Done!

### Build
You can build your containers using `til build CHALLENGE [tag]`. For example:
```bash
til build asr
til build ae algo-update
```

The script first runs `cd` into the directory of the model you want to build (e.g. `/asr`). Then, it builds the image using Docker, automatically adhering to the required naming scheme `TEAM_ID-CHALLENGE:TAG` using any Docker tag you give it, defaulting to `latest` if not provided. You should then test your model using `til test` before using `til submit` to submit your image for evaluation.

```bash
# cd into the directory. For example, `cd ./asr/`
cd CHALLENGE

# Build your image. Remember the . at the end.
docker build -t TEAM_ID-CHALLENGE:TAG .
```
### Test
You can test your containers locally using `til test CHALLENGE [tag]`. For example:

```bash
til test cv
til test noise extra-noisy
```

This will deploy your container on a local Docker network without internet access, and test querying it with all the training data in your track directory (either `/home/jupyter/novice` or `/home/jupyter/advanced`). For all the details, check out the [Wiki](https://github.com/til-ai/til-26/wiki).

### Submit
You can submit your containers for automated evaluation using `til submit CHALLENGE [tag]`. For example:
```bash
til submit nlp
til submit cv epoch-100
```

For all the details of what the submission command does, check out the [Wiki](https://github.com/til-ai/til-26/wiki).

## Links

* The repo [Wiki](https://github.com/til-ai/til-26/wiki) contains tutorials, specifications, resources, and more.
* Your [~~Vertex AI~~ Agent Platform Workbench Instance](https://console.cloud.google.com/agent-platform/workbench/instances?project=til-ai-2026) on Google Cloud Platform is where you'll do most of your development.
* The [Strategist's Handbook](https://tribegroup.notion.site/BrainHack-2026-TIL-AI-Strategist-s-Handbook-33a5263ef45a80429a9dc47c569e40c3) houses the Leaderboard and info about the competition.
* [TIL-AI Curriculum](https://drive.google.com/drive/folders/18zP4pHt5E6YqA3usey16ETEzKNeAn5X9) on Google Drive contains educational materials specially crafted for TIL-AI.
* The [#hackoverflow](https://discord.com/channels/1488845200523661454/1488845611032903691) channel on the TIL-AI Discord server is a forum just for Strategists like you.

---

Code in this repo is licensed under the MIT License.
