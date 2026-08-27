"""
Example dose-calculation algorithm for Grand-Challenge.org.

Runs inside a container. Locally:
    ./do_test_run.sh   # reads ./test/input, writes ./test/output
    ./do_save.sh       # packages the container for upload

Runtime docs: https://grand-challenge.org/documentation/runtime-environment/

The TASK environment variable selects one of four interfaces:

    TASK         input images                   beam-level metadata
    ----------   ---------------------------    ------------------------------------
    photon-ct    ...-source-ct-image-{1..10}    stacked-photon-beam-level-metadata
    proton-ct    ...-source-ct-image-{1..10}    stacked-proton-beam-level-metadata
    photon-mri   ...-source-mri-image-{1..10}   stacked-photon-beam-level-metadata
    proton-mri   ...-source-mri-image-{1..10}   stacked-proton-beam-level-metadata

Instead of predicting real dose, each slice is filled with a dummy dose map
(zeros, uniform noise, or a Gaussian) and thresholded by the per-slice
minimum_cutoff taken from the metadata.
"""

import glob
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

DEFAULT_TASK = "proton-ct"
DOSE_SIMULATION = "gaussian"  # "zeros", "noise" or "gaussian"
NUM_OUTPUT_FILES = 10

CT_DIR_BASE = "radiation-dose-calculation-source-ct-image"
MR_DIR_BASE = "radiation-dose-calculation-source-mri-image"
PHOTON_JSON_NAME = "stacked-photon-beam-level-metadata"
PROTON_JSON_NAME = "stacked-proton-beam-level-metadata"

# Each TASK maps to (input image directory base, beam-level metadata file).
TASK_CONFIG = {
    "photon-ct":  (CT_DIR_BASE, PHOTON_JSON_NAME),
    "proton-ct":  (CT_DIR_BASE, PROTON_JSON_NAME),
    "photon-mri": (MR_DIR_BASE, PHOTON_JSON_NAME),
    "proton-mri": (MR_DIR_BASE, PROTON_JSON_NAME),
}

TASK = os.environ.get("TASK", DEFAULT_TASK) # This is NOT available on the Grand Challenge platform unless set in your Dockerfile, only added here for flexibility changing tasks.
if TASK not in TASK_CONFIG:
    raise ValueError(f"Unknown TASK {TASK!r}; expected one of {sorted(TASK_CONFIG)}")

INPUT_DIR_BASE, INPUT_JSON_NAME = TASK_CONFIG[TASK]


def run(model):
    print(
        f"Running TASK {TASK!r} "
        f"(input_dir_base={INPUT_DIR_BASE!r}, input_json_name={INPUT_JSON_NAME!r})"
    )
    device = select_device()

    print("Loading json metadata:")
    metadata = load_json_file(INPUT_PATH / f"{INPUT_JSON_NAME}.json")
    output_infos = flatten_output_infos(metadata)

    # Group output_infos per output file, keyed by their slice position.
    per_output = [dict() for _ in range(NUM_OUTPUT_FILES)]
    for oi in output_infos:
        per_output[oi["output_file_idx"]][oi["idx_in_output"]] = oi

    stack_sizes = [max(slot) + 1 if slot else 0 for slot in per_output]
    print(f"Stack sizes: {stack_sizes}")

    for output_index in range(NUM_OUTPUT_FILES):
        slot = per_output[output_index]
        stack_size = stack_sizes[output_index]

        output_dir = OUTPUT_PATH / f"images/stacked-radiation-dose-map-{output_index + 1}"
        os.makedirs(output_dir, exist_ok=True)

        if stack_size == 0:
            # Empty stack: write a placeholder to honor the output contract.
            sitk.WriteImage(
                sitk.Image(1, 1, sitk.sitkFloat32), output_dir / "output.mha"
            )
            continue

        # Every slice in a stack shares the same source image.
        input_image = load_input_by_index(slot[0]["input_file_idx"])

        print(
            f"Writing dummy dose stack for output file index {output_index + 1} "
            f"with {stack_size} slices"
        )
        dose_slices = []
        for slice_index in range(stack_size):
            print(f"Generating dummy dose map {slice_index} for output file index {output_index + 1}")
            # In a real algorithm this is where inference would run.
            dose_np = simulate_dose(input_image, slot.get(slice_index), device)
            dose_slice = sitk.GetImageFromArray(dose_np)
            dose_slice.CopyInformation(input_image)
            dose_slices.append(dose_slice)

        stacked = sitk.JoinSeries(dose_slices)
        print(stacked.GetSize())
        sitk.WriteImage(stacked, output_dir / "output.mha", useCompression=False)

    return 0


def simulate_dose(input_image, output_info, device):
    """Build one dummy dose slice and zero out everything below its cutoff."""
    if DOSE_SIMULATION == "gaussian":
        shape = tuple(reversed(input_image.GetSize()))
        dose = make_gaussian_dose(shape, sigma=20, device=device, dtype=torch.float32)
        # Copy: the tensor is cached, and on CPU .numpy() shares its buffer,
        # so the in-place cutoff below would otherwise corrupt the cache.
        dose_np = dose.cpu().numpy().copy()
    elif DOSE_SIMULATION == "noise":
        dose_np = make_noise_dose(input_image, device=device)
    elif DOSE_SIMULATION == "zeros":
        dose_np = make_zeros_dose(input_image, device=device)

    # Threshold below the cutoff to keep the written file small.
    minimum_cutoff = float(output_info["minimum_cutoff"])
    dose_np[dose_np < minimum_cutoff] = 0.0
    return dose_np


def flatten_output_infos(metadata):
    """Collect every output_info, tagged with its source image index.

    Photon and proton metadata nest differently:
        photon: image -> beams -> control_points -> output_info
        proton: image -> beams -> rays -> beamlets -> output_info
    """
    is_proton = TASK in ("proton-ct", "proton-mri")
    output_infos = []
    for image in metadata:
        image_idx = image["image_file_idx"]
        for beam in image["beams"]:
            if is_proton:
                leaves = (
                    beamlet for ray in beam["rays"] for beamlet in ray["beamlets"]
                )
            else:
                leaves = beam["control_points"]
            for leaf in leaves:
                output_info = leaf["output_info"]
                output_info["input_file_idx"] = image_idx
                output_infos.append(output_info)
    return output_infos


def select_device():
    """Prefer CUDA, fall back to CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, falling back to CPU")
    return device


def load_json_file(location):
    with open(location) as f:
        return json.load(f)


def load_sitk_image(location):
    """Read the first .mha file found in a directory."""
    mha_files = glob.glob(str(location / "*.mha"))
    print(f"Searching for input images in {location}")
    if mha_files:
        return sitk.ReadImage(mha_files[0])
    else:
        raise FileNotFoundError("!!!")

@lru_cache(maxsize=1)
def load_input_by_index(input_file_idx):
    location = INPUT_PATH / f"images/{INPUT_DIR_BASE}-{input_file_idx + 1}"
    image = load_sitk_image(location)
    print(
        f"Loaded input image {input_file_idx + 1} "
        f"with shape {image.GetSize()} and spacing {image.GetSpacing()}"
    )
    return image


def make_zeros_dose(reference_image, device=None):
    """3D array of zeros matching the reference image shape."""
    shape = tuple(reversed(reference_image.GetSize()))
    if device is not None:
        return torch.zeros(shape, device=device, dtype=torch.float32).cpu().numpy()
    return np.zeros(shape, dtype=np.float32)


def make_noise_dose(reference_image, device=None):
    """3D array of uniform noise in [0, 1) matching the reference image shape."""
    shape = tuple(reversed(reference_image.GetSize()))
    if device is not None:
        return torch.rand(shape, device=device, dtype=torch.float32).cpu().numpy()
    return np.random.rand(*shape).astype(np.float32)


@lru_cache(maxsize=1)
def make_gaussian_dose(
    shape: tuple[int, int, int],
    sigma: float | tuple[float, float, float],
    center: tuple[float, float, float] | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    amplitude: float = 1e-3,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """3D Gaussian in tensor order (z, y, x); sigma and spacing in physical units."""
    depth, height, width = shape

    if center is None:
        center = ((depth - 1) / 2, (height - 1) / 2, (width - 1) / 2)
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma),) * 3

    z = torch.arange(depth, device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

    dz = (zz - center[0]) * spacing[0]
    dy = (yy - center[1]) * spacing[1]
    dx = (xx - center[2]) * spacing[2]

    return amplitude * torch.exp(
        -0.5 * ((dz / sigma[0]) ** 2 + (dy / sigma[1]) ** 2 + (dx / sigma[2]) ** 2)
    )


if __name__ == "__main__":
    from app import init_model
    raise SystemExit(run(model=init_model()))