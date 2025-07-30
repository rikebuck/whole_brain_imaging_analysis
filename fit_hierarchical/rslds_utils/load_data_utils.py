import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import pickle
# import matplotlib.gridspec as gridspec

# from sklearn.decomposition import PCA
# from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
# import seaborn as sns
# import ipywidgets as widgets
# from IPython.display import display
# from ipywidgets import interact, IntSlider
import sys
import glob
sys.path.append("/Users/friederikebuck/")

# sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')
sys.path.append("/Users/friederikebuck/WholeBrainImagingAnalysis/")
sys.path.append("/ru-auth/local/home/fbuck/scratch/test_rslds_params/")
sys.path.append("/ru-auth/local/home/fbuck/scratch/test_rslds_params/WholeBrainImagingAnalysis/")
from scipy.ndimage import gaussian_filter1d
from get_data.read_neuro_beh_data import get_exp_features
from get_data.process_neural_data import get_derivative_of_neural_activity, get_neural_activity_from_labels, get_neural_activity_and_labels
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification
# from beh_classification.beh_classification_utils import get_start_end_is_of_ones_in_binary_array
import socket
import sklearn
from sklearn.cross_decomposition import CCA
from sklearn.utils import shuffle
import os
import copy


def smooth_trace(raw_trace, sigma = 2):
    denoised_data = gaussian_filter1d(raw_trace, sigma=sigma)
    return denoised_data

def shift_array_by_one(array,  T = 1599):
    _, n_dim = array.shape
    tmp = np.ones((T, n_dim))*np.nan
    tmp[1:] = array
    array = tmp
    return array


def load_all_data(combine_classes = False, hpc = False):
    # this just returns lists of traces, neural labels, and behavior for each worm
    # Also includes dtrace
    # be warned that all elements of the list have different shapes
    combine_classes = np.mod(1+int(combine_classes),2)
    if not hpc: 
        host = socket.gethostname()[0:6]
        if host == 'Bennet' or host == 'bennet' or 'northwestern' in socket.gethostname():
            json_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/Neuropal_no_heat/"
            h5_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/processed_h5/"
            beh_data_dir_parent = "/Users/bennetsakelaris/Desktop/behavior/outputs_sparse/"
        else:
            json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
            h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
            beh_data_dir_parent = "/Users/friederikebuck/Desktop/MBL/project/outputs/"
    else: 
        ##########
        json_dir = "/ru-auth/local/home/fbuck/scratch/flavell_data/Neuropal_no_heat/"
        h5_dir = "/ru-auth/local/home/fbuck/scratch/flavell_data/processed_h5/"
        beh_data_dir_parent = "/ru-auth/local/home/fbuck/scratch/flavell_data/outputs/"
 ##########
    directories = glob.glob(json_dir+"*.json") #get filename/paths of all the json files
    exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates
    for i in range(len(exp_dates)):
        print("Worm {}, date: {}".format(i, exp_dates[i]))

    full_traces = []
    full_neural_labels = []
    full_beh_classification = []
    full_beh_data = []
    print("expDates", exp_dates)
    for exp_date in exp_dates:
        dt, T, beh_data, neural_data, neuroID_to_key = get_exp_features(exp_date, 
                                                                    json_dir = json_dir, 
                                                                    h5_dir = h5_dir)
        behavior_features = get_behavior_features(beh_data)
        beh_data.update(behavior_features)
        behavior_classification = get_behavior_classification(beh_data)

        beh_data_dir = os.path.join(beh_data_dir_parent, str(exp_date), "csvs1")
        csv_name = os.path.join(beh_data_dir, "beh_confocal.csv")
        beh_confocal = np.loadtxt(csv_name, delimiter=",")
        z_beh = beh_confocal
        
        behavior_classification = {
        "is_fwd": beh_confocal==0, 
        "is_rev": beh_confocal==1, 
        "is_turn": beh_confocal==2, 
        "is_pause":  beh_confocal==3,
        }

        neural_labels_orig, ftrace = get_neural_activity_and_labels(neural_data,  denoised = True)

        dtrace = get_derivative_of_neural_activity(smooth_trace(ftrace.T,sigma=5).T, dframes = 1)
        dtrace -= np.mean(dtrace,axis=0)
        dtrace /= np.std(dtrace,axis=0)
        traces = np.concatenate([ftrace[1:,: ],dtrace ], axis =1)
        neural_labels = np.array(["F - "+ lbl[combine_classes] for lbl in neural_labels_orig]+
                                ["dF - "+ lbl[combine_classes] for lbl in neural_labels_orig])
        
        full_traces.append(traces)
        full_neural_labels.append(neural_labels)
        full_beh_classification.append(behavior_classification)
        full_beh_data.append(beh_data)
        
        # print("traces.shape",traces.shape)

    return full_traces, full_neural_labels, full_beh_classification, full_beh_data
        


def load_all_data_but_pretend_its_all_one_worm(combine_classes = False,  T = 1599, hpc = False):
    # This function concatenates everything into one worm and also returns a mask that tells you which neurons are and arent present
    # Thing to ponder: adding columns of NaNs between worms would probably help the rSLDS learn better
    full_traces, full_neural_labels, full_beh_classification, full_beh_data = load_all_data(combine_classes, hpc = hpc)

    #get list of all neurons recorded in at least one trial
    neural_labels_set = set()
    for nl in full_neural_labels:
        neural_labels_set = neural_labels_set.union(set(nl))
    neural_labels = np.sort(list(neural_labels_set))

    #now make the traces array, where the different recordings are concatenated and matched by neuron
    #note: a couple of worms have 15 extra timesteps. I am truncating those to make things easier down the line
   
    traces = np.zeros((T*len(full_traces), neural_labels.shape[0]))*np.nan #initialize traces to nan 
    #build the traces matrix
    w=0
    for tr, labels, in zip(full_traces, full_neural_labels): #for each worm, w
        for i in range(tr.shape[1]): #for each neuron, i
            label = labels[i] #get neuron name
            idx = np.where(neural_labels==label)[0][0] #get index in full array
            traces[w*T:(w+1)*T, idx] = tr[0:T,i] #put the neuron's activity in the appropriate spot
        w+=1 #update worm index

    

    #set up behavior classification dict
    behavior_classification = dict()
    behavior_classification["is_turn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_pause"] = np.zeros(T*len(full_traces))
    behavior_classification["is_rev"] = np.zeros(T*len(full_traces))
    behavior_classification["is_fwd"] = np.zeros(T*len(full_traces))
    behavior_classification["is_revturn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_purerev"] = np.zeros(T*len(full_traces))
    behavior_classification["is_pureturn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_rev_of_rev_turn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_turn_of_rev_turn"] = np.zeros(T*len(full_traces))

    #fill it in
    w=0
    for bc in full_beh_classification:
        for key in bc.keys():
            behavior_classification[key][w*T:(w+1)*T] = bc[key][1:(T+1)] # shifting by one bc of the trace
        w+=1 #update worm index

    #return mask of nan data
    mask =  (~np.isnan(traces)).astype(int)  # 1 where not naan (where exsts )
    return traces, neural_labels, behavior_classification, mask

# removes neurons that were only recorded in fewer than [threshold] experiments
# def remove_rare_neurons(traces, neural_labels, mask, threshold=10): 
#     n_occurances = np.zeros(traces.shape[1]) #count the number of occurances for each neuron
#     for n in np.arange(0, traces.shape[1]):
#         n_occurances[n] = sum(~np.isnan(traces[np.arange(100, traces.shape[0], 1599),n]))
#     print("Removing {} neurons that are not recorded in more than {} experiments".format(sum(n_occurances<=threshold)/2, threshold))
#     print("{} neurons remaining".format(sum(n_occurances>threshold)/2))
    
#     traces = traces[:, n_occurances>threshold]
#     neural_labels = neural_labels[n_occurances>threshold]
#     mask = mask[:, n_occurances>threshold]
#     return traces, neural_labels, mask

# def keep_from_list(traces, neural_labels, mask, to_keep):
#     n_occurances = np.zeros(traces.shape[1], dtype=bool) #see if each neuron is in to_keep
#     for n in np.arange(0, traces.shape[1]):
#         n_occurances[n] = neural_labels[n] in to_keep
#         print(n_occurances[n])

#     print("Removing {} neurons".format((traces.shape[1]-sum(n_occurances))/2))
#     print("{} neurons remaining".format(len(to_keep)/2))
    
#     traces = traces[:, n_occurances]
#     neural_labels = neural_labels[n_occurances]
#     mask = mask[:, n_occurances]
#     return traces, neural_labels, mask


