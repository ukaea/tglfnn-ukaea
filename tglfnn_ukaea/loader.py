from __future__ import annotations

import pathlib
from typing import Literal

import numpy as np

WEIGHTS_DIR = pathlib.Path(__file__).parent / "weights"

# Network version loaded for each machine when no version is given.
DEFAULT_VERSIONS = {
    "multimachine": "3.2.1",
    "step": "1.0.1",
}


def available_versions(
    machine: Literal["step", "multimachine"],
) -> list[str]:
    """Network versions distributed for `machine`."""
    _check_machine(machine)
    return sorted(p.stem for p in (WEIGHTS_DIR / machine).glob("*.npz"))


def load(
    machine: Literal["step", "multimachine"] = "multimachine",
    version: str | None = None,
) -> dict[str, np.ndarray]:
    """Loads the network parameters and normalisations for `machine`.

    The weights are stored as weights/{machine}/{version}.npz. If `version` is
    None, DEFAULT_VERSIONS[machine] is loaded.

    The arrays are exported from the ONNX models by scripts/onnx_to_npz.py; see
    that script for the layout and the forward pass they define.
    """
    _check_machine(machine)
    if version is None:
        version = DEFAULT_VERSIONS[machine]

    npz_file = WEIGHTS_DIR / machine / f"{version}.npz"
    if not npz_file.exists():
        raise ValueError(
            f"Unknown version '{version}' for machine '{machine}' (available:"
            f" {', '.join(available_versions(machine))})"
        )
    with np.load(npz_file, allow_pickle=False) as data:
        params = {key: data[key] for key in data.files}

    if str(params["version"]) != version:
        raise ValueError(
            f"{npz_file} contains version {params['version']}, expected {version}"
        )
    return params


def _check_machine(machine: str) -> None:
    if machine not in DEFAULT_VERSIONS:
        raise ValueError(
            f"Unknown machine type: '{machine}' (must be 'step' or 'multimachine')"
        )
