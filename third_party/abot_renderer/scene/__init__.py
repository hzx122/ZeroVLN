"""Select a renderer model without importing CUDA extensions in CPU services."""
import os


def __getattr__(name):
    if name != "GaussianModel":
        raise AttributeError(name)
    if os.environ.get("SAGE_RENDER_DEVICE") == "cpu":
        from .cpu_gaussian_model import GaussianModel
    else:
        from .gaussian_model import GaussianModel
    return GaussianModel
