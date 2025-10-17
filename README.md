# tglfnn-ukaea
Neural network surrogate models of the [TGLF](https://gafusion.github.io/doc/tglf.html) quasilinear plasma turbulent transport simulator in various parameter spaces.

## Paper for acknowledgment

If you use these models within your work, we request you cite the following paper:

* L. Zanisi et al., [Data efficent digital twinning strategies and surrogate models of quasilinear turbulence in JET and STEP](https://conferences.iaea.org/event/392/contributions/36059/), International Atomic Energy Agency - Fusion Energy Conference, Chengdu, China, 2025

# Usage

Various different methods exist for using the models:

1. Loading from PyTorch checkpoint
2. Loading traced ONNX model
3. Loading traced TorchScript model
4. Loading the parameters directly into pure Python

Loading the traced TorchScript model allows the model to be used in Fortran (see below).
Loading the parameters directly is a minimal-dependency method designed for use with other machine learning frameworks.

## 1. Loading from PyTorch checkpoint

```python
import torch

# Load the model
efe_gb_model = torch.load('MultiMachineHyper_1Aug25/regressor_efe_gb.pt')

# Call the model
input_tensor = torch.tensor([[...]], dtype=torch.float32)  # Replace with appropriate input
output_tensor = efe_gb_model(input_tensor)
```

## 2. Loading traced ONNX model

```python
import onnxruntime as ort

# Load the model
ort_session = ort.InferenceSession('MultiMachineHyper_1Aug25/regressor_efe_gb.onnx')

# Call the model
input_tensor = np.array([[...]], dtype=np.float32)  # Replace with appropriate input
outputs = ort_session.run(None, {'input': input_tensor})
```

## 3. Loading traced TorchScript model

```python
import torch

# Load the model
torchscript_model = torch.jit.load('MultiMachineHyper_1Aug25/regressor_efe_gb_torchscript.pt')

# Call the model
input_tensor = torch.tensor([[...]], dtype=torch.float32)  # Replace with appropriate input
output_tensor = torchscript_model(input_tensor)
```

### Using the traced TorchScript models in Fortran

The traced PyTorch models can be used in Fortran with [FTorch](https://github.com/Cambridge-ICCS/FTorch), which provides Fortran bindings for LibTorch (the C++ backend of PyTorch).  Please [cite the Ftorch publication](https://github.com/Cambridge-ICCS/FTorch#authors-and-acknowledgment) if using these models from Fortran.

Further details on the FTorch Implementation of these networks can be found in a [related project](https://github.com/ProjectTorreyPines/TurbulentTransport.jl/blob/master/utilities/README_onnx_to_pytorch_fortran.md).

#### Prerequisites

- **LibTorch**: Download the appropriate version (CPU or GPU) from the [PyTorch website](https://pytorch.org/get-started/locally/) and ensure it is accessible in your environment. CPU versions of the `LibTorch` and `Pip` packages have been tested. The `LibTorch` version requires no Python to install or run. It is suggested to look at the `FTorch` instructions below first.
- **FTorch**: Install the FTorch library following the instructions in the [FTorch repository](https://github.com/Cambridge-ICCS/FTorch). This also provides a compiler specific module (`ftorch.mod`).
- **Fortran Compiler**: Use a modern Fortran compiler (e.g., `gfortran` or `ifort`) compatible with FTorch.
- **CMake**: Version >= 3.1 required to build FTorch. Not essential, but helpful for building final Fortran code. 

## 4. Loading the parameters directly into pure Python

The parameters are distributed using the `tglfnn_ukaea` python package.

Usage:

```
$ pip install tglfnn_ukaea
$ python
>>> import tglfnn_ukaea
>>> tglfnn_ukaea.loader.load("multimachine")
{
    "stats":  {...}, # Used for normalisation
    "config": {...}, # Network architecture
    "input_labels": (...), # Input feature names (ordered)
    "params": { # Weights and biases
        "efe_gb": {...},
        "efi_gb": {...},
        "pfi_gb": {...},
    },
}
```

The returned dictionary contains all the information needed to implement the neural network in any framework. See [google-deepmind/fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates) for an example.
