This folder contains NNs trained using Mean Squared Error as loss function on a hypercube that is applicable to all devices with aspect ratio R/a>3 and for a DT  plasmas.
Quick facts:
- The following convention is used to name the folders: [Heat flux of electrons](efe): `efe`, [Total heat flux of ions](efi): `efi`, [Particle flux of deuterium ions](pfi): `pfi1`, [Particle flux of tritium ions](pfi2): `pfi2`.
- The space is based on `Ideal_MHD_stability_and_downstream_analysis.xlsx` with priority<=4. This file is available only to STEP/UKIFS employees/contractors.
- 10M data points have been generated with TGLF-SAT2 using the options shown in [common.tglf](common_STEP_ramp_flat_top.tglf). 
- Data has been preprocessed separately for every NN by removing fluxes above 100GB. 
- Validation plots can be found in each folder.
- The ordering of the input variables must be as follows: `AS_3`,`RLNS_1`,`RLNS_2``RLTS_1`,`RLTS_2`,`TAUS_2`,`RMIN_LOC`,`DRMAJDX_LOC`,`Q_LOC`,`SHAT`,`XNUE`,`KAPPA_LOC`,`S_KAPPA_LOC`,`DELTA_LOC`,`S_DELTA_LOC`,`ZEFF`,`VEXB_SHEAR`,`BETAE`
