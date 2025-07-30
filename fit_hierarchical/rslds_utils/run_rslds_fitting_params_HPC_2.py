#data format library
# import h5py
#numpy
import numpy as np
# import numpy.ma as ma
# from sklearn.cluster import KMeans
import sys
import os

import argparse
# %%
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# %matplotlib widget
import pickle
# import matplotlib.gridspec as gridspec

# from sklearn.decomposition import PCA
# from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
# import seaborn as sns
# import ipywidgets as widgets
# from IPython.display import display
# from ipywidgets import interact, IntSlider
import sys
import glob
import os
from datetime import datetime
sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append("/Users/friederikebuck/")
sys.path.append("/Users/friederikebuck/WholeBrainImagingAnalysis/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')
sys.path.append('/ru-auth/local/home/fbuck/scratch/test_rslds_params/')


from rslds_utils.load_data_utils import load_all_data_but_pretend_its_all_one_worm, load_all_data
from rslds_utils.rslds_plotting_utils import *
from rslds_utils.subsample_neurons import subsample_neurons_donut_neurons, subsample_neurons_common_neurons , subsample_neurons_var_explained
from rslds_utils.train_rslds import train_and_eval_vanilla_model, train_and_eval_hierarchical_model
import functools

# %%

from matplotlib.colors import LinearSegmentedColormap

#color palette for plotting, colors as in make_behavior_ethogram
palette = ["coral",     # forward
           "lightblue", # reverse
           "darkgreen", # turn
           "purple","red", "yellow", "black", "pink", "grey", "cyan"]    # pause
cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))




def read_thresh_file(thresh_txt):
    """
    Reads in a .txt file containing threshold, latent_dim, discrete_dim combinations.

    Parameters:
        thresh_txt (str): Path to the text file.

    Returns:
        np.ndarray: Array of shape (n_combinations, 3) with dtype float.
    """
    return np.loadtxt(thresh_txt, delimiter=",")
    
def main(argv):
    # explained_var_folder =  "/Users/friederikebuck/rslds_utils/"
    explained_var_folder = "/ru-auth/local/home/fbuck/scratch/test_rslds_params/rslds_utils/"
    print("in re_rslds_fillrin_prams")
    np.random.seed(0)
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-subsample_id', '--subsample_id', help="subsample_id", default="common_neurons", type=str)
    parser.add_argument('-thresh_i', '--thresh_i', help="thresh_i", default=0, type=str)
    parser.add_argument('-params_i', '--params_i', help="params_i", default=0, type=str)
    parser.add_argument('-rslds_type', '--rslds_type', help="rslds_type", default="vanilla", type=str)
    parser.add_argument('-save_dir_parent', '--save_dir_parent', help="save_dir_parent", default=".", type=str)
    args=parser.parse_args()
    subsample_id = args.subsample_id
    save_dir_parent = args.save_dir_parent
   

    #get thresh from params 
    params_i = int(args.params_i)
    rslds_type = args.rslds_type
    # subsample_id_to_threshold_txt = {
    #     "donut": "donut_thresh.txt",
    #     "common_neurons": "common_neurons_thresh.txt", 
    #     "explained_var":  "explained_var_thresh.txt"

    # }
    subsample_id_to_threshold_txt = {
        "donut": "donut_thresh_2.txt",
        "common_neurons": "common_neurons_thresh_2.txt", 
        "explained_var":  "explained_var_thresh_2.txt"

    }
    thresh_params_txt = os.path.join(explained_var_folder, subsample_id_to_threshold_txt[subsample_id])
    print("thresh_params_txt", thresh_params_txt)
    #load txt;  #get thresh from txt and thresh_i 
    # thresh = load_txt(thresh_txt)[thresh_i]

    #get   n_disc_states = 3 n_latent_dim = 9# D from params text 
    thresh, n_latent_dim, n_disc_states  = read_thresh_file(thresh_params_txt)[params_i,:]
    thresh = float(thresh)
    n_disc_states = int(n_disc_states)
    n_latent_dim = int(n_latent_dim)
    print("thresh, n_disc_states, n_latent_dim", thresh, n_disc_states, n_latent_dim, subsample_id)

    T = 1599
    full_traces, full_neural_labels, full_beh_classification, full_beh_data = load_all_data(combine_classes=True, hpc = True)
    print("full_traces", full_traces)
    print("full_neural_labels", full_neural_labels)
    for i in range(len(full_beh_classification)):
        print("Worm {} number of neurons: {}, timesteps: {}".format(i, full_traces[i].shape[1]/2, full_traces[i].shape[0])) #dividing neurons by 2 to include both traces and dtraces

    traces_all, neural_labels_all, behavior_classification, mask_all = load_all_data_but_pretend_its_all_one_worm(combine_classes=True, hpc = True)
    print("traces_all", traces_all)
    print("neural_labels_all", neural_labels_all)
    subsample_id_to_fnc = {
        "common_neurons": subsample_neurons_common_neurons,
        "donut": functools.partial(subsample_neurons_donut_neurons, donut_neurons_save_dir = explained_var_folder),
        "explained_var":  functools.partial(subsample_neurons_var_explained, foldername = explained_var_folder)
    }

    subsample_neurons = subsample_id_to_fnc[subsample_id]
    
    traces, neural_labels, mask = subsample_neurons(traces_all, neural_labels_all, mask_all, 
                                        thresh = thresh)
    
    save_dir = os.path.join(save_dir_parent, f"{subsample_id}_{thresh}_{n_disc_states}_{n_latent_dim}_{rslds_type}")

    # %%
    Y = np.copy(traces)
    Y[np.isnan(Y)] = 0
    Y = [Y[i*1599:1599*(i+1)] for i in range(len(full_traces))] #convert to list where each element is the time series of a different worm
    mask=mask.astype(bool)
    masks = [mask[i*1599:1599*(i+1)] for i in range(len(full_traces))] #likewise for mask
    tags = [None]*len(masks)
    print("len(Y)", len(Y))
    print("Y[0].shape", Y[0].shape)

    # set up a single behavioral state array ("z" in the language of scott): 
    #   0: is_fwd
    #   1: is_rev
    #   2: is_turn
    z = behavior_classification["is_fwd"]+ 2*behavior_classification["is_rev"]+ 4*behavior_classification["is_pause"]+3*behavior_classification["is_turn"]-1
    z = z.astype(int)



    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    filestr = os.path.join(save_dir,  formatted_datetime)
    os.makedirs(filestr, exist_ok = True)
    np.save( os.path.join(save_dir, formatted_datetime, "neurons.npy"), neural_labels)
    with open(os.path.join(save_dir, formatted_datetime, "Y.npy"), 'wb') as handle:
        pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(os.path.join(save_dir, formatted_datetime, "Y.npy"))
    # %%

    emissions_dim = neural_labels.shape[0]

    transition = "recurrent_only"
    dynamic = "diagonal_gaussian"
    emission = "gaussian_orthog"

    if rslds_type == "vanilla":
        print( "{subsample_id}_{thresh}_{n_disc_states}_{n_latent_dim}_{rslds_type}", f"{subsample_id}_{thresh}_{n_disc_states}_{n_latent_dim}_{rslds_type}")
        var_explained = train_and_eval_vanilla_model(Y, z,tags, masks, neural_labels, emissions_dim,
                                                     n_disc_states,n_latent_dim , transition, dynamic,
                                                     emission,  palette, cmap, formatted_datetime, 
                                                     save_dir = save_dir,  filestr_supp = "_vanilla")
    else: 
        var_explained = train_and_eval_hierarchical_model(Y, z,tags, masks, neural_labels,
                                                          emissions_dim, n_disc_states,n_latent_dim,  
                                                          transition, dynamic, emission,  palette, cmap,
                                                          formatted_datetime, save_dir = save_dir, filestr_supp = "_hierch"
                                       )

if __name__ == "__main__":
    main(sys.argv)