# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# %matplotlib widget
import pickle

import sys

import os
from datetime import datetime
sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append("/Users/friederikebuck/")
sys.path.append("/Users/friederikebuck/WholeBrainImagingAnalysis/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')


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

save_dir_parent = "/Users/friederikebuck/eval_plots/"


# %% [markdown]
# 

# %% [markdown]
# # Load all worms
# Look at shapes

# %%
np.random.seed(0)


T = 1599
full_traces, full_neural_labels, full_beh_classification, full_beh_data = load_all_data(combine_classes=True)
for i in range(len(full_beh_classification)):
    print("Worm {} number of neurons: {}, timesteps: {}".format(i, full_traces[i].shape[1]/2, full_traces[i].shape[0])) #dividing neurons by 2 to include both traces and dtraces

traces_all, neural_labels_all, behavior_classification, mask_all = load_all_data_but_pretend_its_all_one_worm(combine_classes=True)
subsample_id_to_fnc = {
    "donut": subsample_neurons_donut_neurons,
    "common_neurons": subsample_neurons_common_neurons, 
    "explained_var":  functools.partial(subsample_neurons_var_explained, foldername = "/Users/friederikebuck/Downloads/(New!) model selection/for_neural_var/")

}
subsample_id ="common_neurons" # subsample_fncs = [subsample_neurons_donut_neurons, subsample_neurons_common_neurons , functools.partial(subsample_neurons_var_explained, foldername = "/Users/friederikebuck/Downloads/(New!) model selection/for_neural_var/")]
thresh = 15
subsample_neurons = subsample_id_to_fnc[subsample_id]
traces, neural_labels, mask = subsample_neurons(traces_all, neural_labels_all, mask_all, 
                                    thresh = thresh)

save_dir = os.path.join(save_dir_parent, f"{subsample_id}_{thresh}")

# %%
Y = np.copy(traces)
Y[np.isnan(Y)] = 0
Y = [Y[i*1599:1599*(i+1)] for i in range(len(full_traces))] #convert to list where each element is the time series of a different worm
mask=mask.astype(bool)
masks = [mask[i*1599:1599*(i+1)] for i in range(len(full_traces))] #likewise for mask
print(len(Y))
print(Y[5].shape)
tags = [None]*len(masks)

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
# %%
n_disc_states = 3#4#K #Ks = np.arange(4,5,1)
n_latent_dim = 9# D # Ds = np.arange(3,7,1)
emissions_dim = neural_labels.shape[0]

transition = "recurrent_only"
#transition = "sticky_recurrent_only"
dynamic = "diagonal_gaussian"
emission = "gaussian_orthog"



var_explained = train_and_eval_vanilla_model(Y, z,tags, masks, neural_labels, emissions_dim, n_disc_states,n_latent_dim , 
                                       transition, dynamic, emission,  palette, cmap, formatted_datetime, save_dir = save_dir,  filestr_supp = "_vanilla")
        
        



# var_explained = train_and_eval_hierarchical_model(Y, z,tags, masks, neural_labels, emissions_dim, n_disc_states,n_latent_dim , 
#                                        transition, dynamic, emission,  palette, cmap, formatted_datetime, save_dir = save_dir, filestr_supp = "_hierch"
#                                        )


