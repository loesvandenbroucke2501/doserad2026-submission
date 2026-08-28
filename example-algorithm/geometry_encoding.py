import torch
import torch.nn.functional as F
import numpy as np

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