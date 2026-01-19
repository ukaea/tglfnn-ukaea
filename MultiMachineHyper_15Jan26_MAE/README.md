This folder contains NNs trained using Mean Absolute Error as loss function on a hypercube that is applicable to all devices with aspect ratio R/a>3 and only for pure deuterium plasmas.
Quick facts:
- The following convention is used to name the folders: [Heat flux of electrons](efe): `efe`, [Heat flux of ions](efi): `efi`, [Particle flux](pfi): `pfi`.
- 10M data points have been generated with TGLF-SAT2 using the options shown in [common.tglf](common.tglf). 
- Data has been preprocessed separately for every NN by removing fluxes above 100GB. 
- Plots showing how the data cut affects the distribution at the level of each flux are shown in each folder.
- Validation plots can also be found in each folder.
- The ordering of the input variables must be as follows: `RLNS\_1`,`RLTS\_1`,`RLTS\_2`,`TAUS\_2`,`RMIN\_LOC`,`DRMAJDX\_LOC`,`Q\_LOC`,`SHAT`,`XNUE`,`KAPPA\_LOC`,`DELTA\_LOC`,`ZEFF`,`VEXB\_SHEAR`, `BETAE`
