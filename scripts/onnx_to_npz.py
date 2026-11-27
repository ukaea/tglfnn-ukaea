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

    python scripts/onnx_to_npz.py STEP-effectiveion-EM-SAT1-16D-asinh1-arch512x8-single-e5ab7d902649 --version 5.0.2 \
        -o tglfnn_ukaea/weights/step/5.0.2.npz --verify \
        --input-labels RLNS_1 RLTS_1 RLTS_2 TAUS_2 RMIN_LOC DRMAJDX_LOC Q_LOC SHAT XNUE KAPPA_LOC S_KAPPA_LOC DELTA_LOC S_DELTA_LOC ZEFF VEXB_SHEAR BETAE
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import math
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
RESNET_BLOCK_WEIGHT = re.compile(
    r"base\.members\.(\d+)\.model\.blocks\.(\d+)\.net\.0\.weight"
)


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


def parse_onnx_resnet_v42(graph):
    """Reads a STEP EM 5.0.x V42PhysicalVariance(V42Ensemble(ResNetMLP)) graph."""
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    walk = GraphWalker(graph)

    (graph_input,) = graph.input
    input_type = graph_input.type.tensor_type
    dims = input_type.shape.dim
    if input_type.elem_type != onnx.TensorProto.FLOAT or len(dims) != 2 or not dims[1].dim_value:
        raise ValueError("input is not float32 with shape (batch, num_inputs)")
    num_inputs = dims[1].dim_value

    outputs = [o.name for o in graph.output]
    if outputs != ["mean", "variance"]:
        raise ValueError(f"graph outputs {outputs}, expected ['mean', 'variance']")

    blocks_by_member = collections.defaultdict(list)
    for name in initializers:
        match = RESNET_BLOCK_WEIGHT.fullmatch(name)
        if match:
            blocks_by_member[int(match[1])].append(int(match[2]))
    members = sorted(blocks_by_member)
    if not members or members != list(range(len(members))):
        raise ValueError(f"non-contiguous ResNet ensemble members {members}")
    blocks = sorted(blocks_by_member[0])
    if any(sorted(blocks_by_member[m]) != blocks for m in members):
        raise ValueError("ResNet ensemble members have different block counts")

    def stack_param(suffix):
        return np.stack([initializers[f"base.members.{m}.{suffix}"] for m in members])

    # Verify identical input scalers across members.
    x_mean = initializers["base.members.0.scaler.x_mean"].reshape(num_inputs)
    x_std = initializers["base.members.0.scaler.x_std"].reshape(num_inputs)
    log10_mask = initializers["base.members.0.scaler.log10_mask"].astype(bool).reshape(num_inputs)
    for m in members[1:]:
        if not (
            np.array_equal(initializers[f"base.members.{m}.scaler.x_mean"].reshape(num_inputs), x_mean)
            and np.array_equal(initializers[f"base.members.{m}.scaler.x_std"].reshape(num_inputs), x_std)
            and np.array_equal(initializers[f"base.members.{m}.scaler.log10_mask"].astype(bool).reshape(num_inputs), log10_mask)
        ):
            raise ValueError(f"member {m} input scaler differs from member 0")

    # Read scalar constants directly from the ONNX graph nodes.
    node_by_out = {n.output[0]: n for n in graph.node if n.output}

    def const_val(tensor_name):
        return walk.constant(tensor_name)

    # Input log10 floor from scaler Clip node.
    scaler_clip = next(
        n for n in graph.node
        if n.op_type == "Clip" and n.input[0] == graph_input.name
    )
    input_log10_floor = float(const_val(scaler_clip.input[1]).item())

    # Activation: mish uses Softplus->Tanh->Mul on inp.0 output; gelu uses Erf.
    inp_gemm = next(
        n for n in graph.node
        if n.op_type == "Gemm" and "base.members.0.model.inp.0.weight" in n.input
    )
    inp_consumers = [n.op_type for n in walk.consumers(inp_gemm.output[0])]
    if "Softplus" in inp_consumers:
        activation = "mish"
    elif "Div" in inp_consumers:
        activation = "gelu"
    else:
        raise ValueError(f"unsupported activation consumers after inp.0: {inp_consumers}")

    # Member variance epsilon: Softplus -> Squeeze -> Add(eps) in member 0 var head.
    var_softplus = next(
        n for n in graph.node
        if n.op_type == "Softplus" and "/base/members.0/model/Softplus" in n.output[0]
    )
    var_squeeze = walk.next_node(var_softplus.output[0], "Squeeze")
    var_add = walk.next_node(var_squeeze.output[0], "Add")
    variance_epsilon = float(const_val(var_add.input[1]).item())

    # Member output scaling: mu * y_std + y_mean, then asinh_scale * sinh(clip(z, -15, 15)).
    m0_mul = node_by_out["/base/members.0/Mul_output_0"]
    m0_add = node_by_out["/base/members.0/Add_output_0"]
    m0_mul4 = node_by_out["/base/members.0/Mul_4_output_0"]
    output_std = float(const_val(m0_mul.input[1]).item())
    output_mean = float(const_val(m0_add.input[1]).item())
    asinh_scale = float(const_val(m0_mul4.input[1]).item())

    # Mean clamping (heat fluxes clamp >= 0; particle flux is unclamped).
    mean_node = node_by_out["mean"]
    mean_src = node_by_out[mean_node.input[0]]
    if mean_src.op_type == "Clip":
        clamp_mean_min = float(const_val(mean_src.input[1]).item())
    else:
        clamp_mean_min = float("nan")

    # Calibration wrapper feature indices (rmin_idx, betae_idx) from Gather on input.
    input_gathers = [
        n for n in graph.node if n.op_type == "Gather" and n.input[0] == graph_input.name
    ]
    if len(input_gathers) != 2:
        raise ValueError(f"expected 2 Gather nodes on input for calibration, got {len(input_gathers)}")
    rmin_idx = int(const_val(input_gathers[0].input[1]).item())
    betae_idx = int(const_val(input_gathers[1].input[1]).item())

    block_params = []
    for b in blocks:
        block_params.append({
            "norm_weight": stack_param(f"model.blocks.{b}.net.0.weight"),
            "norm_bias": stack_param(f"model.blocks.{b}.net.0.bias"),
            "linear1_weight": stack_param(f"model.blocks.{b}.net.1.weight"),
            "linear1_bias": stack_param(f"model.blocks.{b}.net.1.bias"),
            "linear2_weight": stack_param(f"model.blocks.{b}.net.4.weight"),
            "linear2_bias": stack_param(f"model.blocks.{b}.net.4.bias"),
        })

    return {
        "architecture": "resnet_v42",
        "num_estimators": len(members),
        "num_blocks": len(blocks),
        "hidden_size": int(initializers["base.members.0.model.inp.0.weight"].shape[0]),
        "activation": activation,
        "input_mean": x_mean.astype(np.float64),
        "input_std": x_std.astype(np.float64),
        "input_log10_mask": log10_mask,
        "input_log10_floor": np.float32(input_log10_floor),
        "output_mean": output_mean,
        "output_std": output_std,
        "asinh_scale": np.float32(asinh_scale),
        "clamp_mean_min": np.float32(clamp_mean_min),
        "variance_epsilon": np.float32(variance_epsilon),
        "inp_weight": stack_param("model.inp.0.weight"),
        "inp_bias": stack_param("model.inp.0.bias"),
        "blocks": block_params,
        "norm_weight": stack_param("model.norm.weight"),
        "norm_bias": stack_param("model.norm.bias"),
        "mean_head_0_weight": stack_param("model.mean.0.weight"),
        "mean_head_0_bias": stack_param("model.mean.0.bias"),
        "mean_head_2_weight": stack_param("model.mean.2.weight"),
        "mean_head_2_bias": stack_param("model.mean.2.bias"),
        "var_head_0_weight": stack_param("model.var.0.weight"),
        "var_head_0_bias": stack_param("model.var.0.bias"),
        "var_head_2_weight": stack_param("model.var.2.weight"),
        "var_head_2_bias": stack_param("model.var.2.bias"),
        "cal_theta": initializers["theta"].astype(np.float32),
        "cal_aux_mean": initializers["aux_mean"].astype(np.float32).reshape(2),
        "cal_aux_std": initializers["aux_std"].astype(np.float32).reshape(2),
        "cal_rmin_idx": np.int64(rmin_idx),
        "cal_betae_idx": np.int64(betae_idx),
    }


def parse_onnx(path):
    """Reads one flux's regressor, after checking the graph structure strictly.

    Supports both legacy ParallelDeepEnsemble MLP graphs and STEP EM 5.0.x
    V42PhysicalVariance(V42Ensemble(ResNetMLP)) graphs.
    """
    graph = onnx.load(str(path)).graph
    init_names = {i.name for i in graph.initializer}
    if "base.members.0.scaler.x_mean" in init_names and "theta" in init_names:
        try:
            return parse_onnx_resnet_v42(graph)
        except ValueError as e:
            raise ValueError(f"{path.name}: {e}") from e

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
        "architecture": "mlp",
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
    arch = networks[fluxes[0]]["architecture"]
    if any(networks[f]["architecture"] != arch for f in fluxes):
        raise ValueError("mixed network architectures across fluxes")

    def per_flux(key, dtype=None):
        return np.array([networks[f][key] for f in fluxes], dtype=dtype)

    if arch == "resnet_v42":
        arrays = {
            "version": np.array(version),
            "model_name": np.array(model_dir.name),
            "architecture": np.array("resnet_v42"),
            "input_labels": np.array(input_labels),
            "output_labels": np.array(fluxes),
            "input_mean": per_flux("input_mean", np.float64),
            "input_std": per_flux("input_std", np.float64),
            "input_log10_mask": per_flux("input_log10_mask", bool),
            "input_log10_floor": per_flux("input_log10_floor", np.float32),
            "output_mean": per_flux("output_mean", np.float64),
            "output_std": per_flux("output_std", np.float64),
            "asinh_scale": per_flux("asinh_scale", np.float32),
            "clamp_mean_min": per_flux("clamp_mean_min", np.float32),
            "variance_epsilon": per_flux("variance_epsilon", np.float32),
            "num_estimators": per_flux("num_estimators", np.int64),
            "num_blocks": per_flux("num_blocks", np.int64),
            "hidden_size": per_flux("hidden_size", np.int64),
            "activation": np.array([networks[f]["activation"] for f in fluxes]),
            "onnx_sha256": np.array(
                [hashlib.sha256(onnx_paths[f].read_bytes()).hexdigest() for f in fluxes]
            ),
        }
        for flux, net in networks.items():
            arrays[f"{flux}/inp/weight"] = net["inp_weight"]
            arrays[f"{flux}/inp/bias"] = net["inp_bias"]
            for b, blk in enumerate(net["blocks"]):
                arrays[f"{flux}/block_{b}/norm/weight"] = blk["norm_weight"]
                arrays[f"{flux}/block_{b}/norm/bias"] = blk["norm_bias"]
                arrays[f"{flux}/block_{b}/linear1/weight"] = blk["linear1_weight"]
                arrays[f"{flux}/block_{b}/linear1/bias"] = blk["linear1_bias"]
                arrays[f"{flux}/block_{b}/linear2/weight"] = blk["linear2_weight"]
                arrays[f"{flux}/block_{b}/linear2/bias"] = blk["linear2_bias"]
            arrays[f"{flux}/norm/weight"] = net["norm_weight"]
            arrays[f"{flux}/norm/bias"] = net["norm_bias"]
            for head in ("mean", "var"):
                for idx in (0, 2):
                    arrays[f"{flux}/{head}_head_{idx}/weight"] = net[f"{head}_head_{idx}_weight"]
                    arrays[f"{flux}/{head}_head_{idx}/bias"] = net[f"{head}_head_{idx}_bias"]
            arrays[f"{flux}/cal/theta"] = net["cal_theta"]
            arrays[f"{flux}/cal/aux_mean"] = net["cal_aux_mean"]
            arrays[f"{flux}/cal/aux_std"] = net["cal_aux_std"]
            arrays[f"{flux}/cal/rmin_idx"] = net["cal_rmin_idx"]
            arrays[f"{flux}/cal/betae_idx"] = net["cal_betae_idx"]
        return arrays

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


_ERF_UFUNC = np.frompyfunc(math.erf, 1, 1)


def _apply_activation(x, name):
    if name == "mish":
        return x * np.tanh(np.logaddexp(np.float32(0.0), x))
    if name == "gelu":
        inv_sqrt2 = np.float32(1.0 / math.sqrt(2.0))
        erf_val = _ERF_UFUNC(x * inv_sqrt2).astype(np.float32)
        return np.float32(0.5) * x * (np.float32(1.0) + erf_val)
    raise ValueError(f"unsupported activation: {name}")


def _layer_norm(x, weight, bias, eps=np.float32(1e-5)):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * weight[:, None, :] + bias[:, None, :]


def _forward_resnet_v42(data, x):
    x32 = np.asarray(x, dtype=np.float32)
    inv_ln10 = np.float32(1.0 / math.log(10.0))
    knots = np.array(
        [0.0, math.log1p(1.0), math.log1p(10.0), math.log1p(100.0), math.log1p(1000.0)],
        dtype=np.float32,
    )
    out = {}
    for i, flux in enumerate(data["output_labels"]):
        flux = str(flux)
        floor = np.float32(data["input_log10_floor"][i])
        log10_mask = np.asarray(data["input_log10_mask"][i], dtype=bool)
        x_mean = data["input_mean"][i].astype(np.float32)
        x_std = data["input_std"][i].astype(np.float32)

        log_x = np.log(np.maximum(x32, floor)) * inv_ln10
        xt = np.where(log10_mask[None, :], log_x, x32)
        z_in = (xt - x_mean[None, :]) / x_std[None, :]

        act = str(data["activation"][i])
        h = np.einsum("ebi,eoi->ebo", z_in[None], data[f"{flux}/inp/weight"]) + data[f"{flux}/inp/bias"][:, None, :]
        h = _apply_activation(h, act)
        for b in range(int(data["num_blocks"][i])):
            ln = _layer_norm(h, data[f"{flux}/block_{b}/norm/weight"], data[f"{flux}/block_{b}/norm/bias"])
            u = np.einsum("ebi,eoi->ebo", ln, data[f"{flux}/block_{b}/linear1/weight"]) + data[f"{flux}/block_{b}/linear1/bias"][:, None, :]
            u = _apply_activation(u, act)
            u = np.einsum("ebi,eoi->ebo", u, data[f"{flux}/block_{b}/linear2/weight"]) + data[f"{flux}/block_{b}/linear2/bias"][:, None, :]
            h = h + u
        z = _layer_norm(h, data[f"{flux}/norm/weight"], data[f"{flux}/norm/bias"])

        mh = _apply_activation(
            np.einsum("ebi,eoi->ebo", z, data[f"{flux}/mean_head_0/weight"]) + data[f"{flux}/mean_head_0/bias"][:, None, :],
            act,
        )
        mu_raw = (np.einsum("ebi,eoi->ebo", mh, data[f"{flux}/mean_head_2/weight"]) + data[f"{flux}/mean_head_2/bias"][:, None, :])[..., 0]

        vh = _apply_activation(
            np.einsum("ebi,eoi->ebo", z, data[f"{flux}/var_head_0/weight"]) + data[f"{flux}/var_head_0/bias"][:, None, :],
            act,
        )
        v_logit = (np.einsum("ebi,eoi->ebo", vh, data[f"{flux}/var_head_2/weight"]) + data[f"{flux}/var_head_2/bias"][:, None, :])[..., 0]
        v_raw_head = np.logaddexp(np.float32(0.0), v_logit) + np.float32(data["variance_epsilon"][i])

        y_mean = np.float32(data["output_mean"][i])
        y_std = np.float32(data["output_std"][i])
        asinh_scale = np.float32(data["asinh_scale"][i])
        z_y = mu_raw * y_std + y_mean
        v_y = v_raw_head * (y_std ** 2)
        zc = np.clip(z_y, np.float32(-15.0), np.float32(15.0))
        ep, em = np.exp(zc), np.exp(-zc)
        sinh_z = np.float32(0.5) * (ep - em)
        cosh_z = np.float32(0.5) * (ep + em)
        m_phys = asinh_scale * sinh_z
        v_phys = (asinh_scale * asinh_scale) * v_y * (cosh_z * cosh_z)

        mu = m_phys.mean(axis=0)
        var = v_phys.mean(axis=0) + ((m_phys - mu[None, :]) ** 2).mean(axis=0)
        clamp_min = float(data["clamp_mean_min"][i])
        if math.isfinite(clamp_min):
            mu = np.maximum(mu, np.float32(clamp_min))
        var = np.maximum(var, np.float32(0.0))

        # Post-hoc physical variance calibration (V42PhysicalVariance).
        v0 = np.maximum(var, np.float32(1e-12))
        z_c = np.clip(z_in, np.float32(-6.0), np.float32(6.0))
        aux_raw = np.stack([np.log(np.maximum(np.abs(mu), np.float32(1e-6))), np.log(v0)], axis=1)
        aux = np.clip((aux_raw - data[f"{flux}/cal/aux_mean"][None, :]) / data[f"{flux}/cal/aux_std"][None, :], np.float32(-6.0), np.float32(6.0))
        q = np.log1p(np.abs(mu) / np.float32(0.1))
        r0 = np.clip((q - knots[0]) / (knots[1] - knots[0]), np.float32(0.0), np.float32(1.0))
        r1 = np.clip((q - knots[1]) / (knots[2] - knots[1]), np.float32(0.0), np.float32(1.0))
        r2 = np.clip((q - knots[2]) / (knots[3] - knots[2]), np.float32(0.0), np.float32(1.0))
        r3 = np.clip((q - knots[3]) / (knots[4] - knots[3]), np.float32(0.0), np.float32(1.0))
        b_mag = np.stack([np.float32(1.0) - r0, r0 - r1, r1 - r2, r2 - r3, r3], axis=1)
        near_zero = np.exp(-np.abs(mu) / np.float32(0.1))[:, None]
        neg_in = np.clip(-mu / np.float32(0.05), np.float32(-30.0), np.float32(30.0))
        neg = np.float32(1.0) / (np.float32(1.0) + np.exp(-neg_in))
        neg_feat = np.stack([neg, neg * q], axis=1)

        rmin = x32[:, int(data[f"{flux}/cal/rmin_idx"])]
        log10_betae = np.log(np.maximum(x32[:, int(data[f"{flux}/cal/betae_idx"])], np.float32(1e-8))) * inv_ln10
        oc70 = np.clip((rmin - np.float32(0.70)) / np.float32(0.10), np.float32(0.0), np.float32(1.0))[:, None]
        oc85 = np.clip((rmin - np.float32(0.85)) / np.float32(0.10), np.float32(0.0), np.float32(1.0))[:, None]
        lowb_oc = (
            np.clip((rmin - np.float32(0.65)) / np.float32(0.10), np.float32(0.0), np.float32(1.0))
            * np.clip((np.float32(-2.8) - log10_betae) / np.float32(0.60), np.float32(0.0), np.float32(1.0))
        )[:, None]
        feats = np.concatenate(
            [np.ones((len(mu), 1), dtype=np.float32), z_c, aux, b_mag, near_zero, neg_feat, oc70, oc85, lowb_oc],
            axis=1,
        )
        eta = np.clip(feats @ data[f"{flux}/cal/theta"], np.float32(-25.0), np.float32(25.0))
        v_cal = np.clip(v0 * np.exp(eta), np.float32(1e-12), np.float32(1e12))
        out[flux] = (mu.astype(np.float64), v_cal.astype(np.float64))
    return out


def forward(data, x):
    """NumPy forward pass of an exported npz, mirroring the ONNX graph.

    Returns {flux: (mean, variance)}.
    """
    if "architecture" in data and str(data["architecture"]) == "resnet_v42":
        return _forward_resnet_v42(data, x)

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

    is_resnet = "architecture" in data and str(data["architecture"]) == "resnet_v42"
    n_samples = min(num_samples, 1_000) if is_resnet else num_samples
    rng = np.random.default_rng(seed)
    passed = True
    for i, (flux, path) in enumerate(onnx_paths.items()):
        # Inputs within +/- 2 standard deviations of this flux's normalisation.
        input_mean, input_std = data["input_mean"][i], data["input_std"][i]
        x = input_mean + input_std * rng.uniform(-2, 2, (n_samples, len(input_mean)))
        if "input_log10_mask" in data:
            mask = np.asarray(data["input_log10_mask"][i], dtype=bool)
            x[:, mask] = 10.0 ** x[:, mask]
        x = x.astype(np.float32)

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        onnx_mean, onnx_var = (
            o.ravel() for o in session.run(None, {session.get_inputs()[0].name: x})
        )
        npz_mean, npz_var = forward(data, x)[flux]
        if is_resnet:
            mean_error = np.max(np.abs(npz_mean - onnx_mean) / np.maximum(np.abs(onnx_mean), 1.0))
            var_error = np.max(np.abs(npz_var - onnx_var) / np.maximum(np.abs(onnx_var), 2e-3))
            print(
                f"  verify {flux}: max rel |d mean| = {mean_error:.1e}, "
                f"max rel |d var| = {var_error:.1e}"
            )
            passed &= mean_error < 1e-4 and var_error < 1e-2
        else:
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
        if not path.exists():
            path = model_dir / f"regressor_{flux}.onnx"
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
