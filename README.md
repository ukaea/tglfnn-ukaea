# tglfnn-ukaea
Neural networks as surrogate models that emulate the TGLF quasilinear plasma turbulent transport simulator in various parameter spaces. Each folder includes both ONNX and Pytorch models traced via TorchScript. 

## Using the traced PyTorch models in Fortran

The traced PyTorch models can be used in Fortran with [FTorch](https://github.com/torch/FTorch), which provides Fortran bindings for LibTorch (the C++ backend of PyTorch).

### Prerequisites

- **LibTorch**: Download the appropriate version (CPU or GPU) from the [PyTorch website](https://pytorch.org/get-started/locally/) and ensure it is accessible in your environment. CPU versions of the `LibTorch` and `Pip` packages have been tested. The `LibTorch` version requires no Python to install or run. It is suggested to look at the `FTorch` instructions below first.
- **FTorch**: Install the FTorch library following the instructions in the [FTorch repository](https://github.com/torch/FTorch). This also provides a compiler specific module (`ftorch.mod`).
- **Fortran Compiler**: Use a modern Fortran compiler (e.g., `gfortran` or `ifort`) compatible with FTorch.
- **CMake**: Check for version required. Required to build FTorch. Not essential, but helpful for building final Fortran code. 
