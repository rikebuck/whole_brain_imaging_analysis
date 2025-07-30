import numpy as np
import copy

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec


import sys
sys.path.append("/Users/friederikebuck/")

sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')
sys.path.append("/ru-auth/local/home/fbuck/scratch/test_rslds_params/")

from rslds_utils.load_data_utils import load_all_data_but_pretend_its_all_one_worm, load_all_data
from rslds_utils.rslds_plotting_utils import plot_2d_continuous_states, plot_most_likely_dynamics_new
# from rslds_utils.subsample_neurons import *


def input_slds(slds, input_list_dynamics,neural_labels,  input_list_emissions=[]):
    new_slds = copy.deepcopy(slds)

    new_slds.M += len(input_list_dynamics) + len(input_list_emissions)
    new_slds.emissions.M = new_slds.M
    new_slds.dynamics.M = new_slds.M
    new_slds.transitions.M = new_slds.M

    # Inititalize new input-driven properties
    Vs = np.zeros((new_slds.K, new_slds.D, new_slds.M))
    Fs = np.zeros((1, new_slds.N, new_slds.M))
    Ws = np.zeros((new_slds.K, new_slds.M))

    if slds.M > 0: #if old slds already supported inputs
        Vs[:,:,:slds.M] = slds.dynamics.Vs
        Ws[:,:slds.M] = slds.transitions.Ws
        Fs[:,:,:slds.M] = slds.emissions.Fs

    # Here I'm only changing Vs but maybe its worth changing Ws at some point
    input_dict = {} # neuron name -> input index
    for i in range(len(input_list_dynamics)):
        input_dict[input_list_dynamics[i]] = slds.M+i
        idx = np.argwhere(neural_labels == f"dF - {input_list_dynamics[i]}")[0][0]
        latent_vec = new_slds.emissions.Cs[0][idx]
        for k in range(new_slds.K):
            Vs[k, :,slds.M+i] = latent_vec

    for i in range(len(input_list_emissions)):
        input_dict["e"+input_list_emissions[i]] = slds.M + i + len(input_list_dynamics)
   
        idx = np.argwhere(neural_labels == f"dF - {input_list_emissions[i]}")[0][0]
        Fs[0,idx,slds.M+i+ len(input_list_dynamics)] = 10
        idx = np.argwhere(neural_labels == f"F - {input_list_emissions[i]}")[0][0]
        Fs[0,idx,slds.M+i+ len(input_list_dynamics)] = 10
                
    new_slds.dynamics.Vs = Vs
    new_slds.transitions.Ws = Ws
    new_slds.emissions.Fs = Fs
    return new_slds, input_dict


def inhibit_rim(slds, q_x, filestr, tag, neural_labels):
    input_list = ["RIM"]
    # new_slds, input_dict = input_slds(slds, input_list, neural_labels)
    try:
        # new_slds, input_dict = input_slds(slds, input_list, input_list)
        new_slds, input_dict = input_slds(slds, input_list, neural_labels)
    except:
        print("error: you threw away RIM")
        print(neural_labels)
        return

    new_slds.D = int(new_slds.D)

    T=1000
    pca_x = q_x
    inputs = np.zeros((T,new_slds.M))
    input_id = input_dict["RIM"]
    input_str = -3
    inputs[500:,input_id] = input_str

    # Create the figure and the gridspec layout
    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 5])  # 2 rows, 3 columns
    ax = fig.add_subplot(gs[0, :])
    ax.plot(inputs[:, input_id])
    ax.set_xlabel("time")
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2])


    test_z, test_x, test_y = new_slds.sample(T=T, input=inputs, with_noise=True)
    junk, lim = plot_most_likely_dynamics_new(new_slds, test_x, test_z, pca_x, input_id=input_id, input_str=0, ax=ax1)
    plot_most_likely_dynamics_new(new_slds, test_x[0:500], test_z[0:500], pca_x, input_id=input_id, input_str=0, ax=ax2, lim = lim, inds=(0,1));
    plot_most_likely_dynamics_new(new_slds, test_x[500:], test_z[500:], pca_x, input_id=input_id, input_str=input_str, ax=ax3, lim = lim, pc3=-7);

    ax.set_title("RIM Stimulation")
    ax1.set_title("full trajectory")
    ax2.set_title("pre-stim trajectory")
    ax3.set_title("post-stim trajectory")
    fig.tight_layout()
    plt.savefig(filestr + "/saved_figs/RIM_"+tag+".png")
    plt.close()