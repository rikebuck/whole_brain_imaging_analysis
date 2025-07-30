import numpy as np
import os

donut_threshes = np.arange(0.1, 0.6, 0.1)

var_explained_threshes = np.arange(0.3, 0.8, 0.05) #fewer is modr 

common_neurons_threshes = np.arange(13, 17, 1)

txt_to_threshes = {
           "donut_thresh_2.txt": donut_threshes, 
            "common_neurons_thresh_2.txt": common_neurons_threshes, 
            "explained_var_thresh_2.txt": var_explained_threshes
}


latent_dims =  np.arange(4,5,1) #np.arange(5,9,1) #np.arange(3,9,1)
discrete_dims = np.arange(3,5,1)


save_dir = "/Users/friederikebuck/rslds_utils/"  # Change to desired save directory if needed

for txt, threshes in txt_to_threshes.items(): 
    combos = []
    for thresh in threshes:
        for ld in latent_dims:
            for dd in discrete_dims:
                if dd <= ld:
                    combos.append([thresh, ld, dd])
    combos_array = np.array(combos)
    np.savetxt(os.path.join(save_dir, txt), combos_array, fmt="%.5f", delimiter=",")
    print("txt", txt)
    print(combos_array.shape)

def load_in_params(thresh_txt, param_i):
    """
    Read in the text file and return the parameter combination at row param_i.

    Parameters:
        thresh_txt (str): Filename of the threshold combinations (e.g., "donut_thresh.txt")
        param_i (int): Row index of the parameter combination to return

    Returns:
        tuple: (thresh, latent_dim, discrete_dim)
    """
    data = np.loadtxt(thresh_txt, delimiter=",")
    return tuple(data[param_i])


