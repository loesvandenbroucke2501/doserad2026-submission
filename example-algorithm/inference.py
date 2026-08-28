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
import torch.nn.functional as F 

import matplotlib.pyplot as plt


INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

DEFAULT_TASK = "photon-ct"
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

        output_dir = OUTPUT_PATH / f"images/stacked-radiation-dose-map-{output_index + 1}" # hier ziet je eigenlijk al in een bepaalde output slot
        os.makedirs(output_dir, exist_ok=True)
    
        if stack_size == 0:
            # Empty stack: write a placeholder to honor the output contract.
            sitk.WriteImage(
                sitk.Image(1, 1, sitk.sitkFloat32), output_dir / "output.mha"
            )
            continue

        # Every slice in a stack shares the same source image.
        input_image = load_input_by_index(slot[0]["input_file_idx"]) # ct images (sitk Image)
        ct_arr, ct_aff = convert_sitk_to_numpy(input_image) # ct images (numpy array, affine matrix)
        ct_arr, ct_aff = preprocess_ct(ct_arr, ct_aff)

        input_parameters = []
        input_parameters_index = []
        input_parameters_minimum_cutoff = []
        for entry in metadata:
                if entry["image_file_idx"] == slot[0]["input_file_idx"]:
                    for beam_idx, beam in enumerate(entry["beams"]):
                        for cp_idx, control_point in enumerate(beam["control_points"]):
                            if control_point["output_info"]["output_file_idx"] == output_index:
                                
                                input_parameters.append(
                                    project_mlc(entry, beam_idx, cp_idx, input_image)
                                )            
                                input_parameters_index.append(control_point["output_info"]["idx_in_output"])
                                input_parameters_minimum_cutoff.append(control_point["output_info"]["minimum_cutoff"])
        
        for slice_index in range(stack_size):
            ...
       
        '''
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
        '''

    return 0

class MachineToPatientSpace:

    def __init__(self,
                 machine_pixel_size_mm = [1,1], # mm
                 patient_pixel_size_mm = [2, 2, 2], # mm
                 sad_mm = 1000, # mm
                 ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale_factor = [machine_pixel_size_mm[0] / patient_pixel_size_mm[0], machine_pixel_size_mm[1] / patient_pixel_size_mm[2]]
        self.sad_mm = sad_mm

        self.patient_pixel_size_mm = patient_pixel_size_mm
        self.machine_pixel_size_mm = machine_pixel_size_mm  
    
    def _resample_mlc(self, tensor, scale_factors):
        return torch.nn.functional.interpolate(tensor, scale_factor = scale_factors, mode = 'bilinear')

    def _backproject_scaling(self, mlc_tensor, iso, target_shape):

        mlc_volume = torch.zeros((1, 1, target_shape[0], target_shape[1], target_shape[2]), dtype=torch.float32).to(self.device)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, mlc_tensor.shape[2], device=self.device),
            torch.linspace(-1, 1, mlc_tensor.shape[3], device=self.device),
            indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)[None, ...]

        for y in range(target_shape[1]):

            backproject_distance_mm = (iso[1] - y) * self.patient_pixel_size_mm[1]             
        
            scaling_factor = 1 / ((self.sad_mm - backproject_distance_mm) / self.sad_mm)
            scaled_grid = grid * scaling_factor            

            mlc_projected = F.grid_sample(mlc_tensor, scaled_grid, mode='bilinear', align_corners=True)
            mlc_projected = mlc_projected.unsqueeze(3) 

            # padding (isocenter is in center of the mlc aperture)
            pad_x_left = iso[0] - mlc_projected.shape[2] // 2
            pad_x_right = target_shape[0] - iso[0] - mlc_projected.shape[2] // 2 - (mlc_projected.shape[2] % 2)

            crop_x_left = 0
            crop_x_right = 0
            if pad_x_left < 0:
                crop_x_left = - pad_x_left
                pad_x_left = 0
            if pad_x_right < 0:
                crop_x_right = - pad_x_right
                pad_x_right = 0

            pad_y_left = y
            pad_y_right = target_shape[1] - y - 1

            pad_z_left = iso[2] - mlc_projected.shape[4] // 2
            pad_z_right = target_shape[2] - iso[2] - mlc_projected.shape[4] // 2 - (mlc_projected.shape[4] % 2)

            crop_z_left = 0
            crop_z_right = 0
            if pad_z_left < 0:
                crop_z_left = - pad_z_left
                pad_z_left = 0
            if pad_z_right < 0:
                crop_z_right = - pad_z_right
                pad_z_right = 0

            mlc_projected = F.pad(mlc_projected, (pad_z_left, pad_z_right, pad_y_left, pad_y_right, pad_x_left, pad_x_right), mode='constant', value=0)
            mlc_projected = mlc_projected[:, :, crop_x_left:mlc_projected.shape[2]-crop_x_right, :, crop_z_left:mlc_projected.shape[4]-crop_z_right]

            if y == 0:
                mlc_volume = mlc_projected
            else:
                mlc_volume = mlc_volume + mlc_projected

        return mlc_volume

    @staticmethod
    def _apply_gantry_rotation(tensor, angle, isocenter_coo):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _, _, W, H, D = tensor.shape

        # shift the tensor so that the isocenter is at the center of rotation
        shifts = (
            W // 2 - isocenter_coo[0],
            H // 2 - isocenter_coo[1],
            D // 2 - isocenter_coo[2],
        )
        tensor = torch.roll(tensor, shifts=shifts, dims=(2, 3, 4))
    
        # create the rotation matrix for the gantry rotation around the isocenter (which is now the center of the tensor)
        angle_rad = torch.tensor(angle * np.pi / 180, dtype=torch.float32).to(device)
        gantry_rotation_matrix = torch.tensor([
                [1, 0, 0, 0],
                [0, torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                [0, torch.sin(angle_rad), torch.cos(angle_rad), 0],
                ], device=device, dtype=torch.float32)

        gantry_rotation_matrix = gantry_rotation_matrix.unsqueeze(0).repeat(tensor.shape[0], 1, 1)
        gantry_grid = F.affine_grid(gantry_rotation_matrix, tensor.shape, align_corners=False)
        tensor = F.grid_sample(tensor, gantry_grid, mode='bilinear', align_corners=False)

        # shift the tensor back to the original position
        shifts = (
            W // 2 + isocenter_coo[0],
            H // 2 + isocenter_coo[1],
            D // 2 + isocenter_coo[2],
        )
        tensor = torch.roll(tensor, shifts=shifts, dims=(2, 3, 4))
 
        return tensor
    
    def process(self, mlc_aperture, gantry_angle, isocenter_wc, ct_arr, ct_aff):

        iso = [int(np.round((isocenter_wc[i] - ct_aff[i, 3]) / ct_aff[i,i]))  for i in range(3)]

        geo_enc = torch.zeros(ct_arr.shape, dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        mlc_aperture = torch.tensor(mlc_aperture, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        _, _, W, H, D = geo_enc.shape

        mlc_aperture = self._resample_mlc(mlc_aperture, self.scale_factor)

        geo_enc = self._backproject_scaling(mlc_aperture, iso, ct_arr.shape)
    
        geo_enc = self._apply_gantry_rotation(geo_enc, gantry_angle, iso)
        geo_enc = geo_enc.squeeze().cpu().numpy()

        body_mask = np.zeros_like(ct_arr, dtype=np.uint8)
        body_mask[ct_arr != -1024] = 1

        geo_enc = geo_enc * body_mask

        return geo_enc 

def project_mlc(beam_parameters, beam_idx, cp_idx, input_image):

    ct_arr, ct_aff = convert_sitk_to_numpy(input_image)
    ct_arr = ct_arr.transpose(2, 1, 0)  

    leaf_thickness = 5 # mm
    machine_pixel_size_mm = [1, 1] # mm
    patient_pixel_size_mm = [2, 2, 2] # mm

    num_leaf_pairs = len(beam_parameters["beams"][beam_idx]["control_points"][cp_idx]["mlc_left_int_mm"])
    size_array_mm = (num_leaf_pairs * leaf_thickness, num_leaf_pairs * leaf_thickness) # mm
    size_array_pixels = (int(size_array_mm[0] / machine_pixel_size_mm[0]), int(size_array_mm[1] / machine_pixel_size_mm[1])) # pixels

    GeoEncoder = MachineToPatientSpace(
            machine_pixel_size_mm = machine_pixel_size_mm,
            patient_pixel_size_mm = patient_pixel_size_mm,
            sad_mm = beam_parameters["beams"][beam_idx]["SAD_mm"]
    )
    iso_center = beam_parameters["beams"][beam_idx]["iso_center"]
            
    mlc_aperture = np.zeros((size_array_pixels[0], size_array_pixels[1]), dtype=np.uint8)
    gantry_angle = beam_parameters["beams"][beam_idx]["control_points"][cp_idx]["gantry_angle"]

    # mm
    mlc_left = beam_parameters['beams'][beam_idx]['control_points'][cp_idx]['mlc_left_int_mm']
    mlc_right = beam_parameters['beams'][beam_idx]['control_points'][cp_idx]['mlc_right_int_mm']
    mlc_left = np.repeat(mlc_left, leaf_thickness / (size_array_mm[0] / size_array_pixels[0]))
    mlc_right = np.repeat(mlc_right, leaf_thickness / (size_array_mm[0] / size_array_pixels[0]))
    # mm to indices
    mlc_left = np.round((mlc_left + (size_array_mm[0] / 2)) * (size_array_pixels[0] / size_array_mm[0]))
    mlc_right = np.round((mlc_right + (size_array_mm[0] / 2)) * (size_array_pixels[0] / size_array_mm[0]))

    for r_i in range(len(mlc_left)):
        mlc_aperture[r_i, int(mlc_left[r_i]) : int(mlc_right[r_i])] = 1
    mlc_aperture = mlc_aperture.T

    geo_enc = GeoEncoder.process(mlc_aperture, gantry_angle, iso_center, ct_arr, ct_aff)
    geo_enc, geo_enc_aff = resize_image(geo_enc, ct_aff)
    geo_enc = geo_enc.transpose(2, 1, 0)            

    return geo_enc, geo_enc_aff

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
    print('Input file index:', input_file_idx + 1)  
    location = INPUT_PATH / f"images/{INPUT_DIR_BASE}-{input_file_idx + 1}"
    image = load_sitk_image(location)
    print(
        f"Loaded input image {input_file_idx + 1} "
        f"with shape {image.GetSize()} and spacing {image.GetSpacing()}"
    )
    return image


def convert_sitk_to_numpy(sitk_img):
    
    img_arr = sitk.GetArrayFromImage(sitk_img)

    origin = np.array(sitk_img.GetOrigin())
    spacing = np.array(sitk_img.GetSpacing())
    direction = np.array(sitk_img.GetDirection()).reshape(3, 3)

    img_aff = np.eye(4)
    img_aff[0:3, 0:3] = direction @ np.diag(spacing)
    img_aff[0:3, 3] = origin
    
    return img_arr, img_aff

def convert_numpy_to_sitk(img_arr, img_aff):

    sitk_img = sitk.GetImageFromArray(img_arr)

    sitk_img.SetOrigin(img_aff[0:3, 3])
    sitk_img.SetSpacing(np.linalg.norm(img_aff[0:3, 0:3], axis=0))
    sitk_img.SetDirection((img_aff[0:3, 0] / np.linalg.norm(img_aff[0:3, 0])).tolist() + 
                         (img_aff[0:3, 1] / np.linalg.norm(img_aff[0:3, 1])).tolist() + 
                         (img_aff[0:3, 2] / np.linalg.norm(img_aff[0:3, 2])).tolist())
    return sitk_img


def preprocess_ct(ct_arr, ct_aff):
    ct_arr = ct_arr.transpose(2, 1, 0)
    ct_arr = hu_to_ed(ct_arr)
    ct_arr, ct_aff = resize_image(ct_arr, ct_aff, target_shape=[256, 256, 112])
    return np.transpose(ct_arr, (2, 1, 0)), ct_aff


def resize_image(img_arr, img_aff, target_shape=[256, 256, 112]):
    # around center

    assert len(img_arr.shape) == 3, f"Expected 3D image array, got shape {img_arr.shape}"

    aff = np.eye(4)
    current_shape = img_arr.shape
    difference = np.array(target_shape) - np.array(current_shape)

    crop_start = np.maximum(-difference // 2, 0)
    crop_end   = np.maximum(-difference - (-difference // 2), 0)  # remainder side
    pad_start = np.maximum(difference // 2, 0)
    pad_end   = np.maximum(difference - (difference // 2), 0)       # remainder side

    img_arr = img_arr[crop_start[0]:current_shape[0]-crop_end[0], crop_start[1]:current_shape[1]-crop_end[1], crop_start[2]:current_shape[2]-crop_end[2]]
    img_arr = np.pad(img_arr, ((pad_start[0], pad_end[0]), (pad_start[1], pad_end[1]), (pad_start[2], pad_end[2])), mode='constant', constant_values=0)

    assert img_arr.shape == tuple(target_shape), f"Resized image shape {img_arr.shape} does not match target shape {target_shape}"

    for i in range(3):
        aff[i,i] = img_aff[i,i]
        aff[i, 3] = img_aff[i, 3] - aff[i,i] * pad_start[i] + aff[i,i] * crop_start[i]

    return img_arr, aff

def load_model(model_dir):
    ...

def hu_to_ed(ct_hu_arr):
    
    hu_ed_curve = {
        "entries": [
            {"hu": -1024, "density_g_cm3": 1.200000e-03},
            {"hu":  -999, "density_g_cm3": 1.210000e-03},
            {"hu":  -200, "density_g_cm3": 8.043754e-01},
            {"hu":  -199, "density_g_cm3": 8.183035e-01},
            {"hu":   -10, "density_g_cm3": 1.006579e+00},
            {"hu":    -9, "density_g_cm3": 9.966749e-01},
            {"hu":   120, "density_g_cm3": 1.126553e+00},
            {"hu":   121, "density_g_cm3": 1.095097e+00},
            {"hu":  3000, "density_g_cm3": 3.027294e+00},
            {"hu":  4000, "density_g_cm3": 3.698428e+00}
        ]
    }

    hu_values = [entry["hu"] for entry in hu_ed_curve["entries"]]
    density_values = [entry["density_g_cm3"] for entry in hu_ed_curve["entries"]]

    ct_ed_arr = np.interp(ct_hu_arr, hu_values, density_values)
    return ct_ed_arr


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