# SPR45 ramp-up / Bohm/gyro-Bohm space
- The input space is obtained from JETTO simulations of the STEP ramp up (SPR45) which were run with Bohm/gyro-Bohm transport. Further details on the TGLF settings can be found in [common.tglf](common.tglf) 
- The variables *must* be provided as input of the surrogate models in the following order
`RLNS_1, RLTS_1, RLTS_2, TAUS_2, RMIN_LOC, DRMAJDX_LOC, Q_LOC, SHAT, XNUE, KAPPA_LOC, S_KAPPA_LOC, DELTA_LOC, S_DELTA_LOC, BETAE, ZEFF`       
- Quasineutrality is enforced across the dataset
- `S_HAT` was used for practical reasons, but TGLF would require Q_PRIME_LOC. The following conversion formula can be used: `Q_PRIME_LOC= S_HAT * (Q_LOC/ RMIN_LOC)**2`
- The performance of the surrogate models is shown below:
    ![](validation.png)
- The following naming convention for the NN checkpoints is used: `efe_gb`=$`q_e \ [GB]`$, `efi_gb`=$`q_i \ [GB]`$, `pfi_gb`=$`\Gamma_i \ [GB]`$
- None of the inputs and outputs require any scaling typical of ML models. This is handled internally in the NN checkpoints.

