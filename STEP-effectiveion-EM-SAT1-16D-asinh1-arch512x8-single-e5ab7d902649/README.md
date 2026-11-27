# STEP EM SAT1 (16D, Deuterium Mass Units) — 5.0.2

Light single-member (`512×8`) 16D model per flux for STEP electromagnetic ramp-up and flat-top transport
modelling (selecting the `5.0.1` member per flux with the lowest validation Balanced MAE).
The relative `5.0.2` symlink selects this content-hashed bundle directory. Also distributed as pure-NumPy
`.npz` weights in the `tglfnn_ukaea` Python package (`tglfnn_ukaea.load("step", "5.0.2")`).

This bundle shares its 16-input interface, 30M training/validation/test splits, physics settings, uncertainty
calibration method, and advisory domain guard with the rest of the `5.0.x` family. **See the
[shared 5.0.x README](../STEP-effectiveion-EM-SAT1-16D-asinh1-large-ensemble5-c2cdba9dc173/README.md) for full
documentation** and the shared
[input distributions](../STEP-effectiveion-EM-SAT1-16D-asinh1-large-ensemble5-c2cdba9dc173/plots/input_distributions.png)
and [flux distributions](../STEP-effectiveion-EM-SAT1-16D-asinh1-large-ensemble5-c2cdba9dc173/plots/flux_distributions.png).

## Architecture

| Flux | Width × residual blocks, activation | Members | Completed epochs/member | Selected epochs |
| --- | --- | ---: | ---: | --- |
| efe_gb | 512 × 8 mish | 1 | 25 | 23 |
| efi_gb | 512 × 8 mish | 1 | 30 | 27 |
| pfi_gb | 512 × 8 gelu | 1 | 60 | 52 |

## Held-out test-set scores and coverage

Evaluated across the full 292,910-row held-out test split (no regions or high-flux tails filtered out;
[full metrics](production_metrics.json)):

| Flux | R² | RMSE [GB] | MAE [GB] | Empirical 68% coverage | Empirical 95% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| efe_gb | 0.8210 | 28.7 | 9.377 | 68.0% | 95.3% |
| efi_gb | 0.7851 | 15.25 | 2.576 | 68.2% | 94.4% |
| pfi_gb | 0.8094 | 3.79 | 0.6934 | 67.9% | 94.7% |

## Parity figures and checksums

![Electron heat test parity](plots/parity_efe_gb.png)
![Ion heat test parity](plots/parity_efi_gb.png)
![Particle test parity](plots/parity_pfi_gb.png)

[Source and detailed provenance, builder tag `5.0.2`](https://gitlab.ukaea.uk/pr5739/tglf-surrogate-builder/-/tree/5.0.2/surrogate_models/v42-5.0-series-20261117/5.0.2). Verify LFS payloads via `sha256sum -c SHA256SUMS`.
