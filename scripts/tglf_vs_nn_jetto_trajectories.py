import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.size"] = 15


def compute_shat(q_prime_loc: np.ndarray, rmin_loc: np.ndarray, q_loc: np.ndarray) -> np.ndarray:
    """
    Compute the magnetic shear parameter shat.

    Parameters
    ----------
    q_prime_loc : array-like
        Radial derivative of q.
    rmin_loc : array-like
        Minor radius.
    q_loc : array-like
        Safety factor.

    Returns
    -------
    np.ndarray
        Magnetic shear values.
    """
    return q_prime_loc * (rmin_loc / q_loc) ** 2


def _extract_radial_trajectory(values: np.ndarray, radial_index: int, n_radial: int) -> np.ndarray:
    """
    Extract the time trajectory for a specific radial location.

    Data is stored interleaving radial positions:
        r0_t0, r1_t0, ... rN_t0, r0_t1, r1_t1, ...

    Parameters
    ----------
    values : np.ndarray
        Full vector of values.
    radial_index : int
        Radial location index.
    n_radial : int
        Number of radial locations.

    Returns
    -------
    np.ndarray
        Time trajectory at the chosen radius.
    """
    return values[radial_index::n_radial]


def plot_diff(
    nn_input_path: Path,
    nn_flux_paths: dict,
    tglf_input_path: Path,
    tglf_flux_paths: dict,
    input_names: list,
    n_radial: int = 10,
):
    """
    Plot time traces comparing NN predictions against TGLF results.

    For each input variable and each flux, the function plots the
    temporal trajectory at multiple radial locations.

    Parameters
    ----------
    nn_input_path : Path
        File containing NN inputs.
    nn_flux_paths : dict[str, Path]
        Mapping of flux label -> NN flux file.
    tglf_input_path : Path
        CSV file containing TGLF inputs.
    tglf_flux_paths : dict[str, Path]
        Mapping of flux label -> TGLF flux file.
    input_names : list[str]
        Names of input variables to plot.
    n_radial : int, default=10
        Number of radial locations stored in the dataset.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
    """

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, n_radial))

    # -----------------------
    # Load input data
    # -----------------------

    nn_inputs = np.genfromtxt(nn_input_path)
    nn_df = pd.DataFrame(nn_inputs, columns=input_names)

    tglf_df = pd.read_csv(tglf_input_path)
    tglf_df["shat"] = compute_shat(
        tglf_df["tglf_q_prime_loc_in"],
        tglf_df["tglf_rmin_loc_in"],
        tglf_df["tglf_q_loc_in"],
    )

    tglf_df = tglf_df[input_names]

    # -----------------------
    # Determine time dimension
    # -----------------------

    total_rows = len(nn_df)
    n_timesteps = total_rows // n_radial
    timesteps = np.arange(n_timesteps)

    variables = input_names + list(tglf_flux_paths.keys())

    n_vars = len(variables)
    n_cols = math.ceil(math.sqrt(n_vars))
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    # -----------------------
    # Plot input trajectories
    # -----------------------

    for var_idx, var in enumerate(input_names):
        ax = axes[var_idx]

        for r in range(n_radial):
            tglf_traj = _extract_radial_trajectory(tglf_df[var].values, r, n_radial)
            nn_traj = _extract_radial_trajectory(nn_df[var].values, r, n_radial)

            ax.plot(timesteps, tglf_traj, color=colors[r], label=f'radial pos: {r}')
            ax.plot(timesteps, nn_traj, linestyle=":", color=colors[r])

        ax.set_ylabel(var.replace("tglf_", "").replace("_in", ""))
        ax.set_xlabel("t")
    
    
    # dummy lines to create legend entries
    legend_ax = axes[len(input_names) - 1]

    # store limits before adding dummy lines
    xlim = legend_ax.get_xlim()
    ylim = legend_ax.get_ylim()

    dummy = np.linspace(-1, -2, 10)

    legend_ax.plot(dummy, dummy, "-", color="black", label="TGLF")
    legend_ax.plot(dummy, dummy, ":", color="black", label="NN")

    # restore limits
    legend_ax.set_xlim(xlim)
    legend_ax.set_ylim(ylim)

    handles, labels = legend_ax.get_legend_handles_labels()
    # -----------------------
    # Plot flux trajectories
    # -----------------------

    start_idx = len(input_names)

    for flux_i, label in enumerate(nn_flux_paths.keys()):

        nn_flux = np.genfromtxt(nn_flux_paths[label])
        tglf_flux = np.genfromtxt(tglf_flux_paths[label])

        if label != r"$q_e \ [GB]$":
            nn_flux = nn_flux[:, 0]
            tglf_flux = tglf_flux[:, 0]

        assert len(nn_flux) == len(tglf_flux)
        assert len(nn_df) == len(tglf_df)

        ax = axes[start_idx + flux_i]

        for r in range(n_radial):
            ax.plot(
                timesteps,
                _extract_radial_trajectory(tglf_flux, r, n_radial),
                color=colors[r], label=f'radial pos: {r}'
            )
            ax.plot(
                timesteps,
                _extract_radial_trajectory(nn_flux, r, n_radial),
                linestyle=":",
                color=colors[r],
            )

        ax.set_ylabel(label)
        ax.set_xlabel("t")

    # -----------------------
    # Global legend
    # -----------------------

    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.97, 0.5),
        ncol=1
    )

    fig.subplots_adjust(right=0.85)    
    plt.tight_layout(rect=[0,0,0.85,1])

    return fig, axes


if __name__ == "__main__":

    basepath = Path(
        "/common/cmg/gm7685/jetto/runs/run97781/"
        "feb2626seq2_Zeff1_3p2p1_smalladdtransp_jintracdevel_latest"
    )

    input_names = [
        "TGLF_RLNS_in_1",
        "TGLF_RLTS_in_1",
        "TGLF_RLTS_in_2",
        "TGLF_TAUS_in_2",
        "TGLF_RMIN_LOC_IN",
        "TGLF_DRMAJDX_LOC_IN",
        "TGLF_Q_LOC_IN",
        "shat",
        "TGLF_XNUE_IN",
        "TGLF_KAPPA_LOC_IN",
        "TGLF_DELTA_LOC_IN",
        "TGLF_ZEFF_IN",
        "TGLF_VEXB_SHEAR_in",
    ]

    input_names = [name.lower() for name in input_names]

    n_radial_locations = 19

    nn_input_path = basepath / "0000NN_inputs.txt"
    tglf_input_path = basepath / "0000TGLF_input.csv"

    nn_flux_paths = {
        r"$q_e \ [GB]$": basepath / "0000TGLFNN_elec_eflux_out.txt",
        r"$q_i \ [GB]$": basepath / "0000TGLFNN_ion_eflux_out.txt",
        r"$\Gamma_i \ [GB]$": basepath / "0000TGLFNN_ion_pflux_out.txt",
    }

    tglf_flux_paths = {
        r"$q_e \ [GB]$": basepath / "0000TGLF_elec_eflux_out.txt",
        r"$q_i \ [GB]$": basepath / "0000TGLF_ion_eflux_out.txt",
        r"$\Gamma_i \ [GB]$": basepath / "0000TGLF_ion_pflux_out.txt",
    }

    fig, axes = plot_diff(
        nn_input_path,
        nn_flux_paths,
        tglf_input_path,
        tglf_flux_paths,
        input_names=input_names,
        n_radial=n_radial_locations,
    )

    fig.savefig("traces_comparison.png")
    plt.close(fig)