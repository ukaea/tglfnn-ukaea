"""Generate reference outputs of the ONNX models, distributed as test data.

For each distributed network version, this evaluates the original ONNX models
with onnxruntime on random inputs and writes the inputs and outputs to
tglfnn_ukaea/reference/{machine}/{version}.npz. Implementations of the networks
(e.g. using the parameters from tglfnn_ukaea.load) can be tested against them,
using tglfnn_ukaea.load_reference.

Each npz contains:

    version            ()                      network version, e.g. "3.2.1"
    model_name         ()                      model directory
    input_labels       (n_in,)                 input order
    output_labels      (n_out,)                fluxes, e.g. ["efe_gb", "efi_gb", "pfi_gb"]
    inputs             (num_samples, n_in)  f32
    {flux}             (num_samples, 2)     f64   ONNX (mean, variance)

The inputs are drawn uniformly within +/- 2 standard deviations of every flux's
input normalisation.

Usage:
    python scripts/make_onnx_reference.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnxruntime as ort

import tglfnn_ukaea
from tglfnn_ukaea.loader import REFERENCE_DIR

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
NUM_SAMPLES = 64


def make_reference(machine, version, seed=0):
    """The ONNX models' outputs for `machine` and `version` on random inputs."""
    params = tglfnn_ukaea.load(machine, version)
    model_dir = REPO_DIR / str(params["model_name"])

    mean, std = params["input_mean"], params["input_std"]
    low = (mean - 2 * std).min(axis=0)
    high = (mean + 2 * std).max(axis=0)
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(low, high, (NUM_SAMPLES, len(low))).astype(np.float32)

    reference = {
        "version": params["version"],
        "model_name": params["model_name"],
        "input_labels": params["input_labels"],
        "output_labels": params["output_labels"],
        "inputs": inputs,
    }
    for flux in params["output_labels"]:
        path = model_dir / f"regressor_{flux}_onnx.onnx"
        if path.read_bytes()[:40].startswith(b"version https://git-lfs"):
            raise SystemExit(f"{path} is a git-lfs pointer; run `git lfs pull` first")
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        flux_mean, flux_variance = session.run(
            None, {session.get_inputs()[0].name: inputs}
        )
        reference[str(flux)] = np.stack([flux_mean[:, 0], flux_variance[:, 0]], axis=-1)
    return reference


if __name__ == "__main__":
    for machine in tglfnn_ukaea.DEFAULT_VERSIONS:
        for version in tglfnn_ukaea.available_versions(machine):
            output = REFERENCE_DIR / machine / f"{version}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "wb") as f:
                np.savez(f, **make_reference(machine, version))
            print(f"wrote {output}")
