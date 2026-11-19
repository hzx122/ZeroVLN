"""Read-only Gaussian inference model; same PLY fields/activations as CUDA."""
import torch
from .gaussian_data import load_gs_ply


class GaussianModel:
    def __init__(self, sh_degree, *, app_opt_cfg=None, use_label=False):
        if app_opt_cfg is not None or use_label:
            raise ValueError("CPU renderer requires the standard SAGE SH-only model")
        self.active_sh_degree = self.max_sh_degree = sh_degree

    def load_ply(self, path):
        data = load_gs_ply(path, self.max_sh_degree)
        if data.rot.shape[1:] != (4,) or data.scale.shape[1:] != (3,):
            raise ValueError("SAGE CPU rendering requires XYZ scales and WXYZ quaternions")
        self._xyz = torch.from_numpy(data.xyz)
        self._rotation = torch.from_numpy(data.rot)
        self._scaling = torch.from_numpy(data.scale)
        self._opacity = torch.from_numpy(data.opacity)
        self._features = torch.from_numpy(data.sh).reshape(len(data), -1, 3)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_features(self):
        return self._features

    @property
    def get_features_dc(self):
        return self._features[:, :1, :]
