import os
import glob
import copy
import sys
import pickle 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 


from rslds_visualization_utils import plot_states_and_neurons, plot_states_and_neurons_state_list
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap

sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')

sys.path.append("/Users/friederikebuck/Downloads/worm notes/worm_code/")


from get_data.read_neuro_beh_data import get_exp_features, load_json
from get_data.process_neural_data import get_derivative_of_neural_activity, get_neural_activity_from_labels, get_neural_activity_and_labels
from beh_classification.behavior_features import get_behavior_features
from get_data.read_neuro_beh_data import get_exp_features, load_json

sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from beh_classification.process_midlines.midline_utils import save_as_csv, load_from_csv, convert_coordinates_to_angles
from rslds_motor_state_class_fncs import get_motor_state_start_end_is_rslds_start_end_specific

from load_data_fncs import load_all_data, get_exp_dates



def find_runs(x):
    """Find start indices, end indices, and values of runs in a 1D array."""
    n = len(x)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=x.dtype)

    change_idx = np.diff(x, prepend=x[0]-1).nonzero()[0]
    start_idx = change_idx
    end_idx = np.append(change_idx[1:], n)
    values = x[start_idx]
    return start_idx, end_idx, values


def make_masks(labels):
    labels = np.asarray(labels)
    start_idx, end_idx, values = find_runs(labels)

    mask_1_to_2 = np.zeros_like(labels, dtype=int)
    mask_1_to_0 = np.zeros_like(labels, dtype=int)
    mask_2_after_1 = np.zeros_like(labels, dtype=int)
    mask_2_after_0 = np.zeros_like(labels, dtype=int)

    for i in range(len(values)):
        val = values[i]
        s, e = start_idx[i], end_idx[i]

        if val == 1:
            if i + 1 < len(values) and values[i + 1] == 2:
                mask_1_to_2[s:e] = 1
            elif i + 1 < len(values) and values[i + 1] == 0:
                mask_1_to_0[s:e] = 1

        elif val == 2:
            if i - 1 >= 0 and values[i - 1] == 1:
                mask_2_after_1[s:e] = 1
            elif i - 1 >= 0 and values[i - 1] == 0:
                mask_2_after_0[s:e] = 1
                
    # ev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(z[i])

    return mask_1_to_2, mask_1_to_0, mask_2_after_1, mask_2_after_0

def get_date_to_state_bin(date_to_start_ends, n_frames):
    date_to_state_bin = {}
    all_state_bin =[]
    for date, (starts, ends) in date_to_start_ends.items():
        state_bin = np.zeros(n_frames)
        for start, end in zip(starts, ends):
            state_bin[start:end] = 1
        date_to_state_bin[date] = state_bin
        all_state_bin.append(state_bin)
    return date_to_state_bin, all_state_bin
        

def get_all_states_bin(exp_dates,  z, q_z):
    n_dates = len(exp_dates)
    n_frames = len(z[0])
    all_states = np.zeros((n_dates, n_frames))*np.nan
    pair_to_i = {}
    motor_states = [0,1,2]
    for motor_state in motor_states:#[0,1,2]:
        rslds_state_to_exp_date_to_beh_start_end_is =  get_motor_state_start_end_is_rslds_start_end_specific(motor_state, exp_dates, z, q_z,  rslds_states = [0,1,2])
    
        # motor_state_to_rslds_state_to_exp_date_to_beh_start_end_is[motor_state] = rslds_state_to_exp_date_to_beh_start_end_is
        for i, (pair, exp_date_to_beh_start_end_is) in enumerate(rslds_state_to_exp_date_to_beh_start_end_is.items()):
            pair_to_i[(motor_state, pair)] = i
            for j, (date, start_ends) in enumerate(exp_date_to_beh_start_end_is.items()):
                starts, ends = start_ends
                # print("start_ends", starts, ends )
                for start, end in zip(starts, ends):
                    # print(start)
                    all_states[j, start:end] = i
    return pair_to_i, all_states


def get_transitions(stateseq):
    ''' copied from keypoint moseq github: keypoint-moseq/keypoint_moseq/util.py'''
    transitions = np.nonzero(stateseq[1:] != stateseq[:-1])[0] + 1
    starts = np.insert(transitions, 0, 0)
    ends = np.append(transitions, len(stateseq))
    return  transitions, starts, ends



    
def get_state_start_ends(state, stateseq,  starts= None, ends = None):
    if starts is None: 
        transitions, starts, ends = get_transitions(stateseq)
    # for state1, state2 in state_pairs:
    state_start_is = np.argwhere(state==stateseq[starts[:-1]]).flatten()
    
    state_starts = starts[state_start_is]
    state_ends = ends[state_start_is] ##check 
    
    return state_starts, state_ends

def get_reor_mat(z):
    all_reors = []
    for z_w in z:
        rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(z_w)
        # fig, axs = plt.subplots(5,1, figsize = (24,15))
        # for i, (states, cmap, vmax) in enumerate(zip([z_w, rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin],
        #                                        [beh_cmap,state_cmap, state_cmap,state_cmap, state_cmap ], 
        #                                        [len(palette1), 2,2,2,2]
        #                                        )):
        #     axs[i].imshow(states[:, None].T, cmap = cmap, vmin =0 , vmax =vmax)
        #     axs[i].set_aspect('auto')
            
        
        reor_types = np.zeros(1599)
        reor_types[pure_rev_bin.astype('bool')] = 3
        reor_types[rev_of_rev_turn_bin.astype('bool')] = 4
        reor_types[turn_of_rev_turn_bin.astype('bool')] = 5
        reor_types[pure_turn_bin.astype('bool')] = 6
        all_reors.append(reor_types[:,None])
    return np.concatenate(all_reors, axis =1 ).T



def get_zero_crossing_indices(arr):
    arr = np.asarray(arr)
    signs = np.sign(arr)
    # Replace zeros with previous nonzero sign to avoid false crossings
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i-1]

    crossings = np.where(np.diff(signs) != 0)[0]
    return crossings  # indices *before* crossing


def count_zero_crossing(body_angle, starts, ends):
    total_crossings = []
    crossings_is = []
    if len(starts)==0:
        return [], np.array([])
    for start, end in zip(starts, ends):
        crossings_i = get_zero_crossing_indices(body_angle[start:end])
        crossings_is.append(crossings_i+start)
        total_crossings.append(len(crossings_i))
        
    # print(total_crossings,crossings_is )
    return total_crossings, np.concatenate(crossings_is)
