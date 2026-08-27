# DoseRAD2026 Example Algorithm

This repository is a **starting point for participants of the [DoseRAD2026](https://doserad2026.grand-challenge.org/) challenge on grand-challenge.org**. It provides an example on how to package a (dummy) dose calculation algorithm as a Docker container that implements Grand Challenge's **`invoke` (HTTP) API**.

Clone it, replace the example logic with your own model, test it locally, and upload it to your Algorithm on Grand Challenge. The example itself does **not** compute real dose, it generates placeholder dose maps (zeros by default) so you can see the full input/output plumbing end to end before dropping in your own inference.

Example code based on [original](https://github.com/chrisvanrun/DoseRAD2026-example-algorithm-invoke-API) by [@chrisvanrun](https://github.com/chrisvanrun).

## What the algorithm does

For each job, the container receives a set of source images plus beam-level metadata, and must produce up to 10 stacks of radiation dose maps.

**Inputs** (mounted read-only at `/input`):

- `images/<source-image-base>-{1..10}/`: ten source images (CT or MR, depending on the task)
- `<beam-level-metadata>.json`: stacked per-beam metadata (photon or proton, depending on the task)

**Outputs** (written to `/output`):

- `images/stacked-radiation-dose-map-{1..10}/output.mha`: ten stacked dose maps, one `.mha` per output slot, unused slots **MUST** get a valid placeholder .mha (see [inference.py](inference.py)) image to satisfy grand-challenge output checks. File compression using SimpleITK (`useCompression=True`) can be very slow, avoid it.

## The four interfaces (tasks)

The challenge covers four combinations of imaging modality and particle type. This example selects one via the `TASK` setting near the top of [inference.py](inference.py). IMPORTANT: If you run [do_test_run.sh](do_test_run.sh) you need to also set the correct TASK (line 49).

| `TASK`         | Source images | Beam-level metadata                   |
| -------------- | ------------- | ------------------------------------- |
| `photon-ct`    | CT images     | `stacked-photon-beam-level-metadata`  |
| `proton-ct`    | CT images     | `stacked-proton-beam-level-metadata`  |
| `photon-mri`   | MR images     | `stacked-photon-beam-level-metadata`  |
| `proton-mri`   | MR images     | `stacked-proton-beam-level-metadata`  |

Photon and proton metadata are nested differently:

- **photon:** `image → beams → control_points → output_info`
- **proton:** `image → beams → rays → beamlets → output_info`

You can hard-code `TASK` for a single container, or switch to the `TASK` environment variable (see the commented line in [inference.py](inference.py)) to build one image that serves multiple interfaces.

## Repository layout

| Path                    | Purpose                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| [inference.py](example-algorithm/inference.py)     | **Your algorithm.** Reads `/input`, writes `/output`. Replace the example dose simulation with your model. |
| [app.py](example-algorithm/app.py)                 | The inference server (FastAPI). Exposes `/health` and `/invoke`. Loads the model once at startup. |
| [Dockerfile](example-algorithm/Dockerfile)         | Container definition. Note the required `org.grand-challenge.api-method="invoke"` label. |
| [requirements.txt](example-algorithm/requirements.txt) | Python dependencies to install into the image. Add yours here. |
| [model/](example-algorithm/model/)                 | Optional model weights/resources, mounted at `/opt/ml/model` at runtime. Uploaded separately as a tarball. |
| `test-data/input/`           | Example inputs used by the local test run.                              |
| `test-data/output/`          | Where local test results are collected.                                 |
| `do_build.sh` / `do_test_run.sh` / `do_save.sh` | Build, test, and package helper scripts.            |

## How the invoke API works

Unlike the older `exec` mode (run once, then exit), the `invoke` API keeps a long-lived HTTP server running inside the container ([app.py](example-algorithm/app.py)):

1. **Startup** — the server boots and loads your model. Load it *here*, not at invoke time, so inference stays fast.
2. **`GET /health`** — polled repeatedly until it returns `200 OK`, signalling the model is ready.
3. **`POST /invoke`** — runs inference over the data in `/input`, writes results to `/output`, and returns `201 CREATED`.

The required Docker label `org.grand-challenge.api-method="invoke"` tells Grand Challenge to use this API. Without it, the platform falls back to `exec` mode. The test script checks for this label.

## Adapting it for your submission

1. Choose your `TASK` (or wire up the `TASK` env var) in [inference.py](example-algorithm/inference.py).
2. Replace the example dose generation in `run()` with your model's inference. The example writes zeros/noise/gaussian placeholders, see `DOSE_SIMULATION`.
3. Add any Python dependencies to [requirements.txt](example-algorithm/requirements.txt).
4. Load model weights in `init_model()` ([app.py](example-algorithm/app.py)) from `/opt/ml/model`, and place those files under [model/](example-algorithm/model/).
5. Keep the input/output contract intact: read the ten source images and beam metadata, write the ten stacked dose maps.

## Running the container locally

To build and test the container against the example inputs, run:

    ./do_test_run.sh

This script will:

1. Build the image and verify the `invoke` API label is present
2. Start the inference server and load your model
3. Wait until the health endpoint returns `200 OK`
4. Invoke the algorithm and wait for it to return `201 CREATED`

It runs inside an isolated Docker network with no internet access, mimicking the Grand Challenge runtime. During inference the container reads from `./test/input` and writes results to `./test/output`.

> **GPU note:** GPU access is controlled by `--gpus all` in [do_test_run.sh](do_test_run.sh). Comment/uncomment it depending on your local setup.

## Saving and uploading the container

To package the image for upload to grand-challenge.org, run:

    ./do_save.sh

This produces:

- a gzipped Docker image (`.tar.gz`) — upload this as your Algorithm's container image, and
- `model.tar.gz` — upload this **separately** as a Model on your Algorithm (weights/resources are not baked into the image).

## Further documentation

This repository is supplementary to the official Grand Challenge documentation:

- Algorithms overview: https://grand-challenge.org/documentation/algorithms/
- Building and testing the container (step-by-step tutorial): https://grand-challenge.org/documentation/building-and-testing-the-container/
- Runtime environment on the platform: https://grand-challenge.org/documentation/runtime-environment/

If the documentation does not answer your question, reach out in the challenge forum [DoseRAD2026 Forum](https://doserad2026.grand-challenge.org/forum/topics/).
