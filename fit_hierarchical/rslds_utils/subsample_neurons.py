
import numpy as np

import math

import os



# removes neurons that were only recorded in fewer than [threshold] experiments
def remove_rare_neurons(traces, neural_labels, mask, thresh=10): 
    n_occurances = np.zeros(traces.shape[1]) #count the number of occurances for each neuron
    print(traces.shape[1])
    for n in np.arange(0, traces.shape[1]):
        n_occurances[n] = sum(~np.isnan(traces[np.arange(100, traces.shape[0], 1599),n]))
    print("Removing {} neurons that are not recorded in more than {} experiments".format(sum(n_occurances<=thresh)/2, thresh))
    print("{} neurons remaining".format(sum(n_occurances>thresh)/2))
    
    traces = traces[:, n_occurances>thresh]
    neural_labels = neural_labels[n_occurances>thresh]
    mask = mask[:, n_occurances>thresh]
    return traces, neural_labels, mask

def keep_from_list(traces, neural_labels, mask, to_keep):
    n_occurances = np.zeros(traces.shape[1], dtype=bool) #see if each neuron is in to_keep
    print("traces.shape[1]", traces.shape[1])
    for n in np.arange(0, traces.shape[1]):
        n_occurances[n] = neural_labels[n] in to_keep
        print(n_occurances[n])

    print("Removing {} neurons".format((traces.shape[1]-sum(n_occurances))/2))
    print("{} neurons remaining".format(len(to_keep)/2))
    
    traces = traces[:, n_occurances]
    neural_labels = neural_labels[n_occurances]
    mask = mask[:, n_occurances]
    return traces, neural_labels, mask

######
######
######
######
def get_donut_neurons_ranked(): #TODO*** 
    pass



######
######
######
######

def subsample_neurons_donut_neurons(traces_all, neural_labels_all, mask_all, 
                            # to_keep
                            donut_neurons_save_dir = "",
                            thresh = 0.5, 
                            ):
    with open(os.path.join(donut_neurons_save_dir, "sorted_donut_neurons.txt"), "r") as f:
        sorted_neurons = [line.strip() for line in f]
    
    #### OPTION 1: donut neurons
    # to_keep = ["RIM", "AVA","AVE", "RIB", "AVB", "AIB", "RID", "RME", "ASG", "SIB", "RIV", "VB02", "BAG", "AUA", "AVL", "URY", "AQR", "AIA", "AIM", "IL1"] # top 20 donuty neurons
    
    donut_neurons_ranked = sorted_neurons
    n_neurons = len(donut_neurons_ranked)
    n_neurons_to_keep = math.floor(n_neurons*thresh)
    to_keep = donut_neurons_ranked[n_neurons_to_keep:]
    print("to_keep donut neurons", to_keep)
    neurons_to_keep = ["F - " + neuron for neuron in to_keep] + ["dF - " + neuron for neuron in to_keep]
    traces, neural_labels, mask = keep_from_list(traces_all, neural_labels_all, mask_all, neurons_to_keep)
    return  traces, neural_labels, mask


def subsample_neurons_common_neurons(traces_all, neural_labels_all, mask_all, 
                                    
                            thresh = 2
                            ):


    #### OPTION 2: keep neurons based on how often theyre recorded (threshold = minimum number of worms the neurons are present in)
    traces, neural_labels, mask = remove_rare_neurons(traces_all, neural_labels_all, mask_all, thresh=thresh)
    return traces, neural_labels, mask

def subsample_neurons_var_explained(traces_all, neural_labels_all, mask_all, 
                                    thresh = 0.3, foldername = "/Users/friederikebuck/Downloads/(New!) model selection/for_neural_var/"):
    # #### OPTION 3: keep neurons based on how well the model can predict their activity
    # foldername = "for_neural_var/"
    # items = np.array(os.listdir(foldername))
    # folders = np.array(["." not in directory for directory in items], dtype=bool)
    # folders = items[folders]
    # folder = folders[0]

    # neural_var_explained = np.load(foldername+folder+"/neural_var_explained.npy", allow_pickle=True)
    neural_var_explained = np.load(foldername+"/neural_var_explained.npy", allow_pickle=True)
    to_remove = np.nanmean(neural_var_explained.T, axis=0)<thresh #var explained threshold
    classes = [neuron.split(' - ')[1] for neuron in neural_labels_all[to_remove] ]
    classes = np.unique(classes)
    Fs = ["F - " + classs for classs in classes]
    dFs = ["dF - " + classs for classs in classes]
    neurons_to_remove = Fs + dFs
    neurons_to_keep = list(set(neural_labels_all) - set(neurons_to_remove))
    traces, neural_labels, mask = keep_from_list(traces_all, neural_labels_all, mask_all, neurons_to_keep)
    print(neurons_to_keep)# oh doe sthis no keep dF and F togetoth? 
    return traces, neural_labels, mask
