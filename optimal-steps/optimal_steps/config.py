"""Registration parameters and device selection."""

import logging
import sys
from dataclasses import dataclass
from typing import List, Optional, Union

import torch


@dataclass
class RegistrationConfig:
    # Scale & Steps
    outer_steps: int = 4
    euler_precision_step_mm: float = 1.0
    solver_precision_mm: float = 1e-4

    # Registration & Metrics
    lambda_reg: Union[float, List[float]] = 0.5
    metric_type: str = "plane2plane"
    metric_alpha: float = 1.0
    metric_beta: float = 0.1
    use_symmetric_correspondences: bool = True
    use_sinkhorn: bool = True

    # Features & FPFH
    landmark_weight: float = 0.0
    use_fpfh: bool = True
    # weight of the user-provided custom features, relative to point distances
    # (scalar, or one value per scale like fpfh_weight)
    feature_weight: Union[float, List[float]] = 5.0
    sigmas: Optional[List[float]] = None
    sigma_init: float = 30.0
    sigma_final: float = 7.0
    n_scales: int = 4
    fpfh_weight: Union[float, List[float]] = 0.0
    fpfh_radius: float = 10.0
    normal_weight: float = 0.1
    trust_symmetric: Union[float, List[float]] = 0.0

    # Incompressibility
    incompressibility_weight: Union[float, List[float]] = 0.0
    incompressibility_radius: Optional[float] = None

    # Device: "auto" (CUDA if available, else CPU), "cuda" or "cpu"
    device: str = "auto"


# lambda_reg is multiplied by n_source_points / LAMBDA_REG_REFERENCE_POINTS, so that
# the same value gives a similar stiffness whatever the sampling of the source.
LAMBDA_REG_REFERENCE_POINTS = 50_000

_DEVICE_NOTICE_SHOWN = False


def resolve_device(requested: str = "auto") -> torch.device:
    """Turn a device string into a torch.device, telling the user what happened.

    "auto"  -> CUDA if a GPU is visible, CPU otherwise.
    "cuda"  -> CUDA if available, CPU otherwise (with a warning).
    "cpu"   -> CPU.

    The chosen device is announced once per process, so a CPU run never looks
    like a hang. Note that KeOps only supports CUDA and CPU (no MPS backend),
    so Apple Silicon machines run on CPU.
    """
    global _DEVICE_NOTICE_SHOWN

    requested = (requested or "auto").lower()
    has_cuda = torch.cuda.is_available()

    if requested == "cpu":
        device, message = torch.device("cpu"), "[info] running on CPU (device='cpu')"
    elif has_cuda:
        device = torch.device("cuda" if requested == "auto" else requested)
        name = torch.cuda.get_device_name(device.index or 0)
        message = f"[info] running on CUDA ({name})"
    elif requested == "auto":
        device, message = torch.device("cpu"), "[info] no CUDA found, running on CPU"
    else:
        device = torch.device("cpu")
        message = (
            "[info] device='cuda' was requested but no CUDA device is visible, "
            "running on CPU"
        )

    if not _DEVICE_NOTICE_SHOWN:
        _DEVICE_NOTICE_SHOWN = True
        if device.type == "cpu":
            message += " - registrations will be 10 to 100 times slower than on a GPU"
        print(message, file=sys.stderr, flush=True)
        logging.getLogger(__name__).info(message)

    return device


def per_scale(value, n_scales: int, name: str, factor: float = 1.0) -> List[float]:
    """One value per scale: a scalar is repeated, a list must have one value per scale."""
    if isinstance(value, (list, tuple)):
        if len(value) != n_scales:
            raise ValueError(
                f"{name} list length ({len(value)}) must match the number of scales ({n_scales})"
            )
        return [float(v) * factor for v in value]
    return [float(value) * factor] * n_scales
