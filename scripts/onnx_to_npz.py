"""Export TGLFNN-ukaea network parameters from the ONNX files to an npz.

The ONNX exports are treated as the reference implementation of each network.
This script reads the weights, normalisations and variance epsilon directly
from the ONNX graphs, so the npz reproduces them exactly.

The npz contains only plain arrays (no pickled objects):

    version            ()               network version, e.g. "3.2.1"
    model_name         ()               model directory, e.g. "MultiMachineHyper_1Aug25"
    input_labels       (n_in,)          input order expected by the network
    output_labels      (n_out,)         fluxes, e.g. ["efe_gb", "efi_gb", "pfi_gb"]
    input_mean         (n_out, n_in)  f64   input normalisation
    input_std          (n_out, n_in)  f64
    output_mean        (n_out,)       f64   output normalisation
    output_std         (n_out,)       f64
    variance_epsilon   (n_out,)       f32   added to each member's softplus variance
    num_estimators     (n_out,)       int   ensemble size
    num_layers         (n_out,)       int   linear layers per member
    hidden_size        (n_out,)       int
    activation         ()               "relu"
    {flux}/layer_{j}/weight  (num_estimators, out, in)  f32
    {flux}/layer_{j}/bias    (num_estimators, out)      f32
    onnx_sha256        (n_out,)         provenance

Arrays with a leading n_out axis hold one entry per flux, in the order of
output_labels, as each flux's network has its own copy of these values.

Each ensemble member computes, for normalised inputs z = (x - input_mean) / input_std
(evaluated in float64, then cast to float32):

    h = relu(W_j h + b_j) for j < num_layers - 1;  (mu, s) = W_last h + b_last
    var = softplus(s) + variance_epsilon

The members are combined as mean = E[mu], variance = E[var] + Var[mu], and
un-normalised as mean * output_std + output_mean, variance * output_std**2.

Usage:
    python scripts/onnx_to_npz.py MultiMachineHyper_1Aug25 --version 3.2.1 \
        -o tglfnn_ukaea/weights/multimachine/3.2.1.npz --verify \
        --input-labels RLNS_1 RLTS_1 RLTS_2 TAUS_2 RMIN_LOC DRMAJDX_LOC Q_LOC SHAT XNUE KAPPA_LOC DELTA_LOC ZEFF VEXB_SHEAR

    python scripts/onnx_to_npz.py JETTO-TGLF-EM-17Feb25 --version 1.0.1 \
        -o tglfnn_ukaea/weights/step/1.0.1.npz --verify \
        --input-labels RLNS_1 RLTS_1 RLTS_2 TAUS_2 RMIN_LOC DRMAJDX_LOC Q_LOC SHAT XNUE KAPPA_LOC S_KAPPA_LOC DELTA_LOC S_DELTA_LOC BETAE ZEFF
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import pathlib
import re

import numpy as np
import onnx
from onnx import numpy_helper

DEFAULT_FLUXES = ("efe_gb", "efi_gb", "pfi_gb")

# Regex for extracting the ensemble parameters.
# For example, models.3.model.2.weight is the weight of layer 2
# of ensemble member 3.
WEIGHT_NAME = re.compile(r"models\.(\d+)\.model\.(\d+)\.weight")


# ONNX graph helpers -----------------------------------------------------------


class GraphWalker:
    """Steps through an ONNX graph, recording which nodes have been checked."""

    def __init__(self, graph):
        self.graph = graph
        self.initializers = {
            i.name: numpy_helper.to_array(i) for i in graph.initializer
        }
        # Nodes are identified by their first output, as they may be unnamed.
        self.visited = set()
        self.used_initializers = set()

    def consumers(self, tensor):
        """All nodes that read `tensor`."""
        return [node for node in self.graph.node if tensor in node.input]

    def next_nodes(self, tensor, op_types):
        """The nodes that read `tensor`, in the order given by `op_types`.

        Raises unless `tensor` is read by exactly one node of each of `op_types`.
        """
        consumers = self.consumers(tensor)
        found = sorted(node.op_type for node in consumers)
        if found != sorted(op_types):
            raise ValueError(f"expected {tensor} to feed {op_types}, got {found}")
        return [
            next(node for node in consumers if node.op_type == op_type)
            for op_type in op_types
        ]

    def next_node(self, tensor, op_type):
        """The single node that reads `tensor`, which must be an `op_type` node."""
        (node,) = self.next_nodes(tensor, [op_type])
        return node

    def check(self, node, inputs, **attributes):
        """Checks a node's inputs and attributes exactly, and marks it visited.

        Each entry of `inputs` is either a tensor name (a string), or the value
        expected of a Constant node's output (None to accept any value).
        """
        if len(node.input) != len(inputs):
            raise ValueError(
                f"{node.op_type} has inputs {list(node.input)}, expected {inputs}"
            )
        for actual, expected in zip(node.input, inputs):
            if not isinstance(expected, str):
                value = self.constant(actual)
                if expected is not None and not np.array_equal(value, expected):
                    raise ValueError(
                        f"{node.op_type} input {actual} is {value.tolist()},"
                        f" expected {expected}"
                    )
            elif actual != expected:
                raise ValueError(
                    f"{node.op_type} has inputs {list(node.input)}, expected {inputs}"
                )

        actual_attributes = {
            a.name: onnx.helper.get_attribute_value(a) for a in node.attribute
        }
        if actual_attributes != attributes:
            raise ValueError(
                f"{node.op_type} has attributes {actual_attributes},"
                f" expected {attributes}"
            )
        self.visited.add(node.output[0])
        return node

    def step(self, tensor, op_type, inputs, **attributes):
        """Moves to the single node after `tensor` and checks it."""
        return self.check(self.next_node(tensor, op_type), inputs, **attributes)

    def constant(self, tensor):
        """The value of a tensor produced by a Constant node."""
        producers = [node for node in self.graph.node if tensor in node.output]
        if len(producers) != 1 or producers[0].op_type != "Constant":
            raise ValueError(f"{tensor} is not the output of a Constant node")
        (attribute,) = producers[0].attribute
        if attribute.name != "value":
            raise ValueError(f"Constant {tensor} has attribute {attribute.name}")
        self.visited.add(tensor)
        return numpy_helper.to_array(attribute.t)

    def parameter(self, name, dtype, shape):
        """An initializer, checking its dtype and shape (None matches any size)."""
        if name not in self.initializers:
            raise ValueError(f"missing initializer {name}")
        array = self.initializers[name]
        if array.dtype != dtype or len(array.shape) != len(shape):
            raise ValueError(
                f"{name} is {array.dtype} {array.shape}, expected {dtype} {shape}"
            )
        for actual, expected in zip(array.shape, shape):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} has shape {array.shape}, expected {shape}")
        self.used_initializers.add(name)
        return array

    def check_complete(self):
        """Raises if any node or initializer was not reached by the walk."""
        unvisited = [
            f"{node.op_type} -> {node.output[0]}"
            for node in self.graph.node
            if node.output[0] not in self.visited
        ]
        if unvisited:
            raise ValueError(f"unexpected nodes {unvisited}")
        unused = set(self.initializers) - self.used_initializers
        if unused:
            raise ValueError(f"unexpected initializers {sorted(unused)}")


# Network parsing functions ------------------------------------------------------


def check_onnx_structure(graph):
    """Checks that the graph is exactly a ParallelDeepEnsemble regressor.

    Walks the graph from the input to the two outputs, in the order of the
    forward pass described at the top of this file, checking every node's
    type, inputs, attributes and constants, and every parameter's dtype and
    shape. Any node or initializer the walk does not reach is an error, so
    anything unexpected raises ValueError rather than producing a silently
    wrong export.
    """
    walk = GraphWalker(graph)
    DOUBLE, FLOAT = onnx.TensorProto.DOUBLE, onnx.TensorProto.FLOAT

    # Input x: float32, shape (batch, num_inputs).
    (graph_input,) = graph.input
    x = graph_input.name
    input_type = graph_input.type.tensor_type
    dims = input_type.shape.dim
    if input_type.elem_type != FLOAT or len(dims) != 2 or not dims[1].dim_value:
        raise ValueError(f"input {x} is not float32 with shape (batch, num_inputs)")
    num_inputs = dims[1].dim_value

    # Input normalisation: z = float32((float64(x) - mean_inputs) / std_inputs).
    walk.parameter("mean_inputs", np.float64, (1, num_inputs))
    walk.parameter("std_inputs", np.float64, (1, num_inputs))
    x64 = walk.step(x, "Cast", [x], to=DOUBLE).output[0]
    centred = walk.step(x64, "Sub", [x64, "mean_inputs"]).output[0]
    scaled = walk.step(centred, "Div", [centred, "std_inputs"]).output[0]
    z = walk.step(scaled, "Cast", [scaled], to=FLOAT).output[0]

    # Ensemble members: z feeds the first layer of every member. Each member
    # is a chain of Gemm layers with a ReLU between each pair, using only its
    # own parameters models.{member}.model.{layer}.{weight,bias}. The final
    # layer's two outputs are split into (mu, s), and var = softplus(s) + eps.
    mus, variances, layer_shapes, epsilons = {}, {}, {}, set()
    for gemm in walk.consumers(z):
        match = WEIGHT_NAME.fullmatch(gemm.input[1]) if len(gemm.input) > 1 else None
        if gemm.op_type != "Gemm" or match is None:
            raise ValueError(
                f"{z} feeds {gemm.op_type} {list(gemm.input)}, expected the"
                " first layer of an ensemble member"
            )
        member = int(match[1])
        if member in mus:
            raise ValueError(f"ensemble member {member} appears twice")

        h, size, shapes = z, num_inputs, []
        while True:
            match = (
                WEIGHT_NAME.fullmatch(gemm.input[1]) if len(gemm.input) > 1 else None
            )
            if match is None or int(match[1]) != member:
                raise ValueError(
                    f"Gemm in ensemble member {member} reads {list(gemm.input)}"
                )
            prefix = f"models.{member}.model.{match[2]}"
            weight = walk.parameter(f"{prefix}.weight", np.float32, (None, size))
            size = weight.shape[0]
            walk.parameter(f"{prefix}.bias", np.float32, (size,))
            walk.check(
                gemm,
                [h, f"{prefix}.weight", f"{prefix}.bias"],
                alpha=1.0,
                beta=1.0,
                transB=1,
            )
            shapes.append((match[2], weight.shape))
            h = gemm.output[0]

            # Either a ReLU and another layer follow, or this was the last layer.
            (after,) = walk.consumers(h) or [None]
            if after is None or after.op_type != "Relu":
                break
            h = walk.check(after, [h]).output[0]
            gemm = walk.next_node(h, "Gemm")

        if size != 2:
            raise ValueError(f"member {member}: final layer outputs {size}, not 2")
        mu, s = walk.step(h, "Split", [h, [1, 1]], axis=1).output
        softplus = walk.step(s, "Softplus", [s]).output[0]
        add_epsilon = walk.step(softplus, "Add", [softplus, None])
        epsilons.add(walk.constant(add_epsilon.input[1]).astype(np.float32).item())

        mus[member] = mu
        variances[member] = add_epsilon.output[0]
        layer_shapes[member] = shapes

    num_members = len(mus)
    if num_members == 0 or sorted(mus) != list(range(num_members)):
        raise ValueError(f"non-contiguous ensemble members {sorted(mus)}")
    if any(layer_shapes[m] != layer_shapes[0] for m in mus):
        raise ValueError("ensemble members have different layers")
    if len(epsilons) != 1:
        raise ValueError(f"inconsistent variance epsilons {epsilons}")

    # Ensemble statistics: each member's mu and var are stacked on a new
    # leading axis, then mean = E[mu] and variance = E[var] + E[mu^2] - E[mu]^2.
    def stack(member_tensors):
        unsqueezed = [
            walk.step(t, "Unsqueeze", [t, [0]]).output[0] for t in member_tensors
        ]
        return walk.step(unsqueezed[0], "Concat", unsqueezed, axis=0).output[0]

    all_mu = stack([mus[m] for m in range(num_members)])
    all_var = stack([variances[m] for m in range(num_members)])

    mean_node, mu_squared_node = walk.next_nodes(all_mu, ["ReduceMean", "Pow"])
    mean = walk.check(mean_node, [all_mu], axes=[0], keepdims=0).output[0]
    mu_squared = walk.check(mu_squared_node, [all_mu, 2.0]).output[0]
    mean_of_squares = walk.step(
        mu_squared, "ReduceMean", [mu_squared], axes=[0], keepdims=0
    ).output[0]
    square_of_mean_node, mean_cast_node = walk.next_nodes(mean, ["Pow", "Cast"])
    square_of_mean = walk.check(square_of_mean_node, [mean, 2.0]).output[0]
    variance_of_mu = walk.step(
        mean_of_squares, "Sub", [mean_of_squares, square_of_mean]
    ).output[0]
    mean_var = walk.step(all_var, "ReduceMean", [all_var], axes=[0], keepdims=0).output[
        0
    ]
    variance = walk.step(mean_var, "Add", [mean_var, variance_of_mu]).output[0]

    # Output un-normalisation, in float64:
    # mean * std_output + mean_output, and variance * std_output**2.
    walk.parameter("mean_output", np.float64, (1,))
    walk.parameter("std_output", np.float64, (1,))
    mean64 = walk.check(mean_cast_node, [mean], to=DOUBLE).output[0]
    scaled_mean = walk.step(mean64, "Mul", [mean64, "std_output"]).output[0]
    output_mean = walk.step(scaled_mean, "Add", [scaled_mean, "mean_output"]).output[0]

    _, std_squared_node = walk.next_nodes("std_output", ["Mul", "Pow"])
    std_squared = walk.check(std_squared_node, ["std_output", 2.0]).output[0]
    variance64 = walk.step(variance, "Cast", [variance], to=DOUBLE).output[0]
    output_variance = walk.step(variance64, "Mul", [variance64, std_squared]).output[0]

    # The graph outputs (mean, variance), in that order, and nothing else.
    outputs = [o.name for o in graph.output]
    if outputs != [output_mean, output_variance]:
        raise ValueError(
            f"graph outputs {outputs}, expected {[output_mean, output_variance]}"
        )
    walk.check_complete()


def read_layers(initializers):
    """Weights and biases of each layer, stacked over the ensemble members."""
    layers_by_member = collections.defaultdict(list)
    for name in initializers:
        match = WEIGHT_NAME.fullmatch(name)
        if match:
            layers_by_member[int(match[1])].append(int(match[2]))
    members = sorted(layers_by_member)
    layers = sorted(layers_by_member[0])

    weights, biases = [], []
    for layer in layers:
        weights.append(
            np.stack(
                [initializers[f"models.{m}.model.{layer}.weight"] for m in members]
            )
        )
        biases.append(
            np.stack([initializers[f"models.{m}.model.{layer}.bias"] for m in members])
        )
    return weights, biases


def parse_onnx(path):
    """Reads one flux's regressor, after checking the graph structure strictly.

    Returns a dict of the network's parameters. "weights" and "biases" are lists
    with one entry per layer, stacked over the ensemble: weights[j] has shape
    (num_estimators, out, in) and biases[j] (num_estimators, out). They are
    lists because the layer shapes differ (in = n_in for the first layer,
    out = 2 for the last).
    """
    graph = onnx.load(str(path)).graph
    try:
        check_onnx_structure(graph)
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from e

    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    weights, biases = read_layers(initializers)
    num_inputs = weights[0].shape[2]

    # The variance epsilon is the constant in Softplus -> Add(epsilon), which
    # check_onnx_structure has confirmed is the same for every member.
    walk = GraphWalker(graph)
    softplus = next(node for node in graph.node if node.op_type == "Softplus")
    add_epsilon = walk.next_node(softplus.output[0], "Add")
    variance_epsilon = walk.constant(add_epsilon.input[1]).astype(np.float32).item()

    return {
        "weights": weights,
        "biases": biases,
        "input_mean": initializers["mean_inputs"].astype(np.float64).reshape(num_inputs),
        "input_std": initializers["std_inputs"].astype(np.float64).reshape(num_inputs),
        "output_mean": float(initializers["mean_output"].item()),
        "output_std": float(initializers["std_output"].item()),
        "variance_epsilon": variance_epsilon,
    }


# Building the npz -------------------------------------------------------------


def check_input_labels(networks, input_labels):
    """Every flux's network must take one input per input label."""
    for flux, network in networks.items():
        num_inputs = len(network["input_mean"])
        if len(input_labels) != num_inputs:
            raise SystemExit(
                f"{len(input_labels)} input labels but the {flux} network takes"
                f" {num_inputs} inputs"
            )


def build_arrays(networks, onnx_paths, model_dir, version, input_labels):
    """The npz contents, as documented at the top of this file."""
    fluxes = list(networks)

    def per_flux(key, dtype=None):
        return np.array([networks[f][key] for f in fluxes], dtype=dtype)

    arrays = {
        "version": np.array(version),
        "model_name": np.array(model_dir.name),
        "input_labels": np.array(input_labels),
        "output_labels": np.array(fluxes),
        "input_mean": per_flux("input_mean"),
        "input_std": per_flux("input_std"),
        "output_mean": per_flux("output_mean"),
        "output_std": per_flux("output_std"),
        "variance_epsilon": per_flux("variance_epsilon", np.float32),
        # Weights are (num_estimators, out, in); the first layer's out is the
        # hidden size.
        "num_estimators": np.array(
            [len(networks[f]["weights"][0]) for f in fluxes], dtype=np.int64
        ),
        "num_layers": np.array(
            [len(networks[f]["weights"]) for f in fluxes], dtype=np.int64
        ),
        "hidden_size": np.array(
            [networks[f]["weights"][0].shape[1] for f in fluxes], dtype=np.int64
        ),
        "activation": np.array("relu"),
        "onnx_sha256": np.array(
            [hashlib.sha256(onnx_paths[f].read_bytes()).hexdigest() for f in fluxes]
        ),
    }

    for flux, network in networks.items():
        for j, (w, b) in enumerate(zip(network["weights"], network["biases"])):
            arrays[f"{flux}/layer_{j}/weight"] = w
            arrays[f"{flux}/layer_{j}/bias"] = b
    return arrays


# Verification -----------------------------------------------------------------


def forward(data, x):
    """NumPy forward pass of an exported npz, mirroring the ONNX graph.

    Returns {flux: (mean, variance)}.
    """
    out = {}
    for i, flux in enumerate(data["output_labels"]):
        z = (x.astype(np.float64) - data["input_mean"][i]) / data["input_std"][i]
        num_layers = int(data["num_layers"][i])
        h = z.astype(np.float32)[None]  # (1, batch, n_in), broadcast over the ensemble
        for j in range(num_layers):
            w = data[f"{flux}/layer_{j}/weight"]
            b = data[f"{flux}/layer_{j}/bias"]
            h = np.einsum("ebi,eoi->ebo", h, w) + b[:, None, :]
            if j < num_layers - 1:
                h = np.maximum(h, 0)

        member_mean = h[..., 0]
        member_var = np.logaddexp(np.float32(0), h[..., 1]) + data["variance_epsilon"][i]
        mean = member_mean.mean(0)
        # The graph forms the ensemble variance as E[x^2] - E[x]^2.
        variance = member_var.mean(0) + ((member_mean**2).mean(0) - mean**2)

        std = data["output_std"][i]
        out[str(flux)] = (
            mean.astype(np.float64) * std + data["output_mean"][i],
            variance.astype(np.float64) * std**2,
        )
    return out


def verify(npz_path, onnx_paths, num_samples=10_000, seed=0):
    """Compares the npz forward pass with onnxruntime on random inputs."""
    import onnxruntime as ort

    with np.load(npz_path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}

    rng = np.random.default_rng(seed)
    passed = True
    for i, (flux, path) in enumerate(onnx_paths.items()):
        # Inputs within +/- 2 standard deviations of this flux's normalisation.
        input_mean, input_std = data["input_mean"][i], data["input_std"][i]
        x = input_mean + input_std * rng.uniform(-2, 2, (num_samples, len(input_mean)))
        x = x.astype(np.float32)

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        onnx_mean, onnx_var = (
            o.ravel() for o in session.run(None, {session.get_inputs()[0].name: x})
        )
        npz_mean, npz_var = forward(data, x)[flux]
        std = data["output_std"][i]

        mean_error = np.abs(npz_mean - onnx_mean).max() / std
        # The ensemble variance is E[x^2] - E[x]^2 in float32, so where members
        # agree (normalised variance ~ eps) it is only defined to f32 rounding;
        # compare against that floor rather than the variance itself.
        var_error = np.max(np.abs(npz_var - onnx_var) / (onnx_var + 1e-4 * std**2))
        print(
            f"  verify {flux}: max |d mean|/output_std = {mean_error:.1e}, "
            f"max |d var|/(var + 1e-4 output_std^2) = {var_error:.1e}"
        )
        passed &= mean_error < 1e-4 and var_error < 1e-2

    if not passed:
        raise SystemExit("Verification FAILED")
    print("  verification passed")


# Export -----------------------------------------------------------------------


def onnx_to_npz(
    model_dir, output, fluxes, version, input_labels, run_verification=False
):
    """Exports the ONNX regressors in `model_dir` to the npz file `output`.

    Optionally verifies the exported npz against onnxruntime.
    """
    model_dir = pathlib.Path(model_dir).resolve()
    output = pathlib.Path(output)

    onnx_paths, networks = {}, {}
    for flux in fluxes:
        path = model_dir / f"regressor_{flux}_onnx.onnx"
        if path.read_bytes()[:40].startswith(b"version https://git-lfs"):
            raise SystemExit(f"{path} is a git-lfs pointer; run `git lfs pull` first")
        onnx_paths[flux] = path
        networks[flux] = parse_onnx(path)

    check_input_labels(networks, input_labels)
    arrays = build_arrays(networks, onnx_paths, model_dir, version, input_labels)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        np.savez(f, **arrays)
    size_mb = output.stat().st_size / 1e6
    print(
        f"wrote {output} ({size_mb:.1f} MB): {model_dir.name} v{version},"
        f" inputs {input_labels}, outputs {fluxes}"
    )

    if run_verification:
        verify(output, onnx_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "model_dir",
        type=pathlib.Path,
        help="model directory, e.g. MultiMachineHyper_1Aug25",
    )
    parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--fluxes",
        nargs="+",
        default=list(DEFAULT_FLUXES),
        help=f"fluxes to export (default: {' '.join(DEFAULT_FLUXES)})",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="network version, e.g. 3.2.1",
    )
    parser.add_argument(
        "--input-labels",
        nargs="+",
        required=True,
        help="input names, in the order the network expects them",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare the exported parameters against onnxruntime",
    )
    args = parser.parse_args()
    onnx_to_npz(
        args.model_dir,
        args.output,
        args.fluxes,
        args.version,
        args.input_labels,
        run_verification=args.verify,
    )
