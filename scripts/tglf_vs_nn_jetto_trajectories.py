import pandas as pd
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np
import math

plt.rcParams['font.size'] = 15

def _get_shat(qprimeloc, rminloc, qloc):
	return qprimeloc*(rminloc/qloc)**2


def plot_diff(nn_input_path, nn_flux_paths, tglf_input_path, tglf_flux_paths, input_names=None, N=10):
    cmap = plt.get_cmap("viridis")
    positions = np.linspace(0,1,N)
    colors = cmap(positions)
    data = np.genfromtxt(nn_input_path)       
    nn_df = pd.DataFrame(data=data, columns=input_names) 
    tglf_df = pd.read_csv(tglf_input_path)
    tglf_df['shat']= _get_shat(tglf_df['tglf_q_prime_loc_in'], tglf_df['tglf_rmin_loc_in'], tglf_df['tglf_q_loc_in'])
    tglf_df = tglf_df[input_names]

    all_vars = input_names+list(tglf_flux_paths.keys())
    num_vars = len(all_vars)
    total_rows = len(nn_df)
    T = total_rows // N  # Number of timesteps
    timesteps = np.arange(T)
    cols = math.ceil(math.sqrt(num_vars))
    rows = math.ceil(num_vars / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()


    for idx, col in enumerate(input_names):
        ax = axes[idx]
        for pos_idx in range(N):
            # Extract values for position 'pos_idx' at rows pos_idx, pos_idx+N, ...
            
            trajectory = tglf_df[col].iloc[pos_idx::N].values
            ax.plot(timesteps, trajectory, label=f'radial pos: {pos_idx}', color=colors[pos_idx])
            nn_trajectory = nn_df[col].iloc[pos_idx::N].values
            ax.plot(timesteps, nn_trajectory, color=colors[pos_idx])
          
        ax.set_ylabel(col.strip('tglf_').strip('_in'))
        ax.set_xlabel('t')
    xlim = axes[idx].get_xlim()
    ylim = axes[idx].get_ylim()
    xx = np.linspace(-100010,-100001, 10)
    axes[idx].plot(xx,xx, ls='-', color='black', label='TGLF')
    axes[idx].plot(xx,xx, ls=':', color='black', label='NN')
    axes[idx].set_xlim(xlim)
    axes[idx].set_ylim(ylim)
    handles, labels = axes[idx].get_legend_handles_labels()

    for i, key in enumerate(nn_flux_paths.keys()):
        nn_flux = np.genfromtxt(nn_flux_paths[key])
        tglf_flux = np.genfromtxt(tglf_flux_paths[key])
        if key!=r'$q_e \ [GB]$':
            tglf_flux = tglf_flux[:,0]
            nn_flux = nn_flux[:,0]

        assert len(nn_flux) == len(tglf_flux)
        assert len(tglf_df)==len(nn_df)        
        for pos_idx in range(N):
            axes[idx+i+1].plot(timesteps, tglf_flux[pos_idx::N], color=colors[pos_idx])
            axes[idx+i+1].plot(timesteps, nn_flux[pos_idx::N], ls=':', color=colors[pos_idx])
        axes[idx+i+1].set_ylabel(key)
        axes[idx+i+1].set_xlabel('t')

    # try:
    #     axes[idx+i+2].legend(handles=handles, labels=labels, ncol=2)
    # except:
    #     axes[idx].legend(ncol=2)
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(0.97, 0.5),
        ncol=1
    )

    fig.subplots_adjust(right=0.85)    
    # # # for a in axes[-(len(axes)-num_vars):]:
    # #     a.axis('off')
    plt.tight_layout(rect=[0,0,0.85,1])

    return fig, axes

    




if __name__=='__main__':
    basepath = Path('/common/cmg/gm7685/jetto/runs/run97781/feb2626seq2_Zeff1_3p2p1_smalladdtransp_jintracdevel_latest')
    input_names = ['TGLF_RLNS_in_1','TGLF_RLTS_in_1','TGLF_RLTS_in_2','TGLF_TAUS_in_2','TGLF_RMIN_LOC_IN','TGLF_DRMAJDX_LOC_IN','TGLF_Q_LOC_IN','shat','TGLF_XNUE_IN','TGLF_KAPPA_LOC_IN','TGLF_DELTA_LOC_IN','TGLF_ZEFF_IN','TGLF_VEXB_SHEAR_in']
    input_names = [name.lower() for name in input_names]
    n_radial_locations = 19

    nn_input_path = basepath / '0000NN_inputs.txt'
    tglf_input_path = basepath / '0000TGLF_input.csv'    

    nn_flux_paths = {
        r'$q_e \ [GB]$': basepath / '0000TGLFNN_elec_eflux_out.txt',
        r'$q_i \ [GB]$': basepath / '0000TGLFNN_ion_eflux_out.txt',
        r'$\Gamma_i \ [GB]$': basepath / '0000TGLFNN_ion_pflux_out.txt'
    }
    tglf_flux_paths = {
        r'$q_e \ [GB]$': basepath / '0000TGLF_elec_eflux_out.txt',
        r'$q_i \ [GB]$': basepath / '0000TGLF_ion_eflux_out.txt',
        r'$\Gamma_i \ [GB]$': basepath / '0000TGLF_ion_pflux_out.txt'
    }    
    
    fig, axes = plot_diff(nn_input_path, nn_flux_paths, tglf_input_path, tglf_flux_paths, input_names=input_names, N=n_radial_locations)
    #fig.tight_layout()
    fig.savefig('traces_comparison.png')
    fig.clf()