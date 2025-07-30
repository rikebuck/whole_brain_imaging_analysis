from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as colors
import cv2
import copy
from functools import partial

####figure out how bennet sampled everything.. 
import numpy as np
from scipy.ndimage import gaussian_filter1d
import copy

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from itertools import permutations
import sys

def interpolate_feature_time_series(
                                    feature_time_series, 
                                    original_time, new_time,
                                    ):
    f_interp_imgs = interp1d(original_time, feature_time_series, axis=0, kind='linear', fill_value="extrapolate")
    # Get the interpolated values for the new frames
    feature_time_series_interpol = f_interp_imgs(new_time)
    return feature_time_series_interpol

def evenly_sample_feature(feature_timeseries, 
                          nir_to_confocal,
                          time_bins=10, ):

    indices = nir_to_confocal#.astype('int32')[1:]

    # Initialize frame counter for saving files
    frame_counter = 0
    feature_resampled = []
    # Loop through each unique index value
    for idx in np.unique(indices):
        if idx == 0: 
            continue
        frames = np.argwhere(indices == idx).flatten()

        # Define the original time points (0 to len(current_features)-1)
        original_time = np.arange(frames.shape[0])  # shape: (n_frames,)

        # Define the new time points (interpolated)
        new_time = np.linspace(0, frames.shape[0]- 1, time_bins) 
        
        interpolated_feature = interpolate_feature_time_series(
                            feature_timeseries[frames], 
                            original_time, new_time,
                            )

        feature_resampled.append(interpolated_feature)#[None, :])
    feature_resampled = np.concatenate(feature_resampled, axis =0 )
    return feature_resampled

def resample_binary(values, original_fps=6, target_fps=1.6):
    values = np.array(values)
    indices = np.arange(len(values))
    new_indices = np.linspace(0, len(values) - 1, int(len(values) * (target_fps / original_fps)))
    
    # Linear interpolation
    interpolated_values = np.interp(new_indices, indices, values)

    
    return interpolated_values

def get_acceleration(signed_speed, time_interval, sigma = 2 ): # this works..
     # Standard deviation for Gaussian kernel
    denoised_data = gaussian_filter1d(signed_speed, sigma=sigma)

    # Step 2: Compute the derivative of the denoised data
    # derivative = np.diff(denoised_data )
    return denoised_data, (denoised_data[time_interval:]- denoised_data[:-time_interval])/time_interval



def get_body_angle_mag_from_midline(midlines):
    body_angle_mag_all = []
    body_angles = []
    for midline in midlines: 
        
        mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
        
        body_angle_mags = np.linalg.norm(mean_subtracted_angles)#, axis = 1)
        body_angle_mag_all.append(body_angle_mags)
        body_angles.append(mean_subtracted_angles)
    
    return np.array(body_angle_mag_all), np.array(body_angles)
       
def convert_coordinates_to_angles(midline):

    diffs = midline[1:,:]-midline[0:-1,:]
    angles = np.arccos(diffs[:,1]/np.sqrt(np.sum(diffs**2,axis=1)))*np.sign(diffs[:,0])
    angles = np.unwrap(angles)
    mean_angle = np.mean(angles)
    mean_subtracted_angle = angles-np.mean(angles)
    return mean_subtracted_angle, mean_angle


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



def plot_states_and_neurons(state,
                            # traces, labels, 
                            body_angle, 
                         fig = None, axs = None, palette = None, cmap = None):
    
    if cmap is None:

        #color palette for plotting, colors as in make_behavior_ethogram
        palette = ["coral",     # forward
                "lightblue", # reverse
                "darkgreen", # turn
                "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
    fig, axs = plt.subplots(2,1,sharex = True, figsize=(18, 6))
    ax = axs[0]
    ax_img = ax.twinx()        
    ax_img.imshow(state[None,:], aspect="auto", cmap=cmap, alpha=0.3, 
                    vmin=0, vmax=len(palette)
                    , extent=[0, len(state), 0, 1]
                    )
    

    ax_img.set_yticks([])
    ax_img.set_ylim(0, 1)  # Keep the background thin and constant

    # Hide twin axis spines
    for spine in ax_img.spines.values():
        spine.set_visible(False)

    ax.set_yticks([])
    # if i == 0:
    #     ax.set_title(f"{date}; Beh")
    # else:
    #     ax.set_title("Inferred by rSLDS")
    #     ax.set_xticks([])
    head_i = 3
    ax.plot(np.arange(body_angle[:, head_i].shape[0]), body_angle[:, head_i], c ="red", label = "head")
    body_i = 20
    ax.plot(np.arange(body_angle[:, body_i].shape[0]), body_angle[:, body_i], c ="orange", label = "body")
    axs[0].legend(loc="upper left")
    ax.axhline(0)
    
    
    axs[1].imshow(body_angle.T)
    return fig, axs, ax_img 


def plot_states_and_neurons_1(beh_state, rslds_state,
                            # traces, labels, 
                            body_angle, 
                         fig = None, axs = None, palette = None, cmap = None):
    
    if cmap is None:

        #color palette for plotting, colors as in make_behavior_ethogram
        palette = ["coral",     # forward
                "lightblue", # reverse
                "darkgreen", # turn
                "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
    fig, axs = plt.subplots(2,1,sharex = True, figsize=(18, 6))
    ax = axs[0]
    ax_img = ax.twinx()        
    ax_img.imshow(beh_state[None,:], aspect="auto", cmap=cmap, alpha=0.3, 
                    vmin=0, vmax=len(palette)
                    , extent=[0, len(beh_state), 0, 1]
                    )
    

    ax_img.set_yticks([])
    ax_img.set_ylim(0, 1)  # Keep the background thin and constant

    # Hide twin axis spines
    for spine in ax_img.spines.values():
        spine.set_visible(False)

    ax.set_yticks([])
    # if i == 0:
    #     ax.set_title(f"{date}; Beh")
    # else:
    #     ax.set_title("Inferred by rSLDS")
    #     ax.set_xticks([])
    head_i = 3
    ax.plot(np.arange(body_angle[:, head_i].shape[0]), body_angle[:, head_i], c ="red", label = "head")
    body_i = 20
    ax.plot(np.arange(body_angle[:, body_i].shape[0]), body_angle[:, body_i], c ="orange", label = "body")
    ax.legend(loc="upper left")
    ax.axhline(0)
    
    
    ax = axs[1]
    ax_img1 = ax.twinx()        
    ax_img1.imshow(rslds_state[None,:], aspect="auto", cmap=cmap, alpha=0.3, 
                    vmin=0, vmax=len(palette)
                    , extent=[0, len(rslds_state), 0, 1]
                    )
    

    ax_img1.set_yticks([])
    ax_img1.set_ylim(0, 1)  # Keep the background thin and constant

    # Hide twin axis spines
    for spine in ax_img1.spines.values():
        spine.set_visible(False)

    ax.set_yticks([])
    # if i == 0:
    #     ax.set_title(f"{date}; Beh")
    # else:
    #     ax.set_title("Inferred by rSLDS")
    #     ax.set_xticks([])
    head_i = 3
    ax.plot(np.arange(body_angle[:, head_i].shape[0]), body_angle[:, head_i], c ="red", label = "head")
    body_i = 20
    ax.plot(np.arange(body_angle[:, body_i].shape[0]), body_angle[:, body_i], c ="orange", label = "body")
    ax.legend(loc="upper left")
    ax.axhline(0)
    
    
    # axs[-1].imshow(body_angle.T)
    return fig, axs, ax_img , ax_img1




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



###############################

def get_motor_state_start_end_is_rslds_start_end_specific(motor_state, exp_dates, z, q_z,  rslds_states = [0,1,2], start = None, end = None, plot = False):
    # rslds_pairs = permutations(rslds_states)
    rslds_pairs = [(x,y) for x in rslds_states for y in rslds_states]
    if start is None: 
        start = 0 
        end = z[0].shape[0]
    rslds_state_to_exp_date_to_beh_start_end_is ={pair:{} for pair in rslds_pairs}# {pair:([],[]) for pair in rslds_pairs}
    for rslds_start_state, rslds_end_state in rslds_pairs: 
        
        for i, (date, z_w, q_z_w) in enumerate(zip(exp_dates, z, q_z)):# (exp_date, behseq) in exp_date_to_beh.items():
            
            beh_starts, beh_ends = get_state_start_ends(motor_state, z_w[ start:end],  starts= None, ends = None)
            '''get where start in rslds state '''
            rev_is = np.argwhere(
                np.logical_and(
                rslds_start_state==q_z_w[beh_starts]
                                    , 
                                    rslds_end_state==q_z_w[beh_ends-1]
                                    
                                    )
                                ).flatten()
            # total_behs+=rev_is.shape[0]
            starts, ends = beh_starts[rev_is], beh_ends[rev_is]


            rslds_state_to_exp_date_to_beh_start_end_is[(rslds_start_state, rslds_end_state)][date] = [starts, ends]

        if plot: #indented here so just for one date
            _, body_angles = get_body_angle_mag_from_midline(exp_date_to_midlines[date])
            fig, axs, ax_img, ax_img1 = plot_states_and_neurons_1(z_w[start:end], q_z_w[start:end],
                                # traces, labels, 
                                body_angles[ start:end,:],
                            fig = None, axs = None, palette = None, cmap = None)
            ax_img.set_title(f"beh: {motor_state} rslds: {rslds_start_state} { rslds_end_state}; date")
            
            for rev_i in rev_is:
                ax_img.axvline(beh_starts[rev_i], color = "green")
                ax_img.axvline(beh_ends[rev_i], color = "red")


    return rslds_state_to_exp_date_to_beh_start_end_is

def get_features_from_start_end_is(feature_to_dict, rslds_state_to_exp_date_to_beh_start_end_is ):
    feature_rslds_state_to_feature_vals = {feature: {} for feature in feature_to_dict.keys()}
    for feature, feature_params in feature_to_dict.items(): 
        print("feature", feature)
        if  feature == "duration (s)": 
            continue
        # feature_rslds_state_to_feature_vals[feature] = {pair: [] for pair in rslds_pairs}
        exp_date_to_feature =  feature_params["exp_date_to_feature"]

        # bins = feature_params["bins"]
        feature_fnc = feature_params["feature_fnc"]
        # ylim = feature_params["ylim"]
        
        for rslds_start_state, rslds_end_state in rslds_pairs: 
            # fig, ax = plt.subplots()
            
            # cdf_fig, cdf_ax = plt.subplots()
            # rslds_state_to_feature = {}
        # for rslds_start_state in  rslds_start_states:#range(n_rslds_states+1): 
            '''get number of reversals that: 
                - start in rslds state and end in other rslds state 
                - start in rslds state and stay in rslds state 
                - start in differnt state and end in rslds state 
            ...based on this decide which version to plot.. 
            '''

            feature_vals = []
            # beh_states = []
        
            for exp_date, featureseq in exp_date_to_feature.items():
                # print("!!!!min", np.min(featureseq))
                
                # rsldsseq = date_to_discrete_rslds_states[exp_date]
                # behseq = exp_date_to_beh[exp_date]
                
                if  feature =="speed (um/s)": 
                    featureseq = copy.deepcopy(exp_date_to_feature[exp_date])*1000
                

                # state_starts, state_ends = get_state_start_ends(rev_state , behseq,  starts= None, ends = None)
                
                # rsldsseq = exp_date_to_rslds_states[exp_date]
                
                # rev_starts, rev_ends = get_state_start_ends(rev_state , behseq,  starts= None, ends = None)
                starts, ends = rslds_state_to_exp_date_to_beh_start_end_is[(rslds_start_state, rslds_end_state)][exp_date]
                
                '''get where start in rslds state '''
                print("featureseq", exp_date, featureseq)

                for start, end in zip(starts, ends):

                    feature_vals.append(feature_fnc(featureseq[start: end]))

            feature_rslds_state_to_feature_vals[feature][(rslds_start_state, rslds_end_state )] = feature_vals
   
    return feature_rslds_state_to_feature_vals

def remove_nans(vals):
    vals = np.array(vals)  
    return vals[~np.isnan(vals)] 

def visualize_transition_matrix(transition_matrix, title = "Transition Matrix",  labels=None):
    """
    Visualizes the transition matrix using a heatmap.

    Parameters:
        transition_matrix (np.ndarray): 2D numpy array representing the transition matrix.
        labels (list or None): List of state labels. If None, indices are used.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    magma_cmap = plt.cm.magma
    # Create a truncated version of the colormap
    # Use the Normalize function to scale the data between 0 and 0.75
    magma_cmap = mcolors.LinearSegmentedColormap.from_list(
    'truncated_magma', magma_cmap(np.linspace(0.1, 0.75, 256))
    )

    
    
    cax = ax.matshow(transition_matrix, cmap=magma_cmap)
    plt.colorbar(cax, ax=ax)

    # Set labels
    if labels is None:
        labels = [str(i) for i in range(transition_matrix.shape[0])]
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)


    ax.set_title(title, fontsize=14)
    ax.set_xlabel("To State", fontsize=12)
    ax.set_ylabel("From State", fontsize=12)

    # Add text annotations
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            value = transition_matrix[i, j]
            if value > 0:
                # ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black")
                # ax.text(j, i, f"{value:.2f}", ha="center", va="center", color= "black")
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color= "white")

    

    return fig, ax


def interpolates_discrete(labels, n_target = 100 ):

    # # Original array of class labels
    # labels = np.array([1, 1, 2, 2, 3, 3, 3, 2, 1])  # example input
    n_original = len(labels)

    # Target number of frames
    n_target = 100

    # Create interpolation function using nearest-neighbor
    interp_func = interp1d(np.linspace(0, 1, n_original),
                        labels,
                        kind='nearest')

    # Interpolate to 100 frames
    interpolated_labels = interp_func(np.linspace(0, 1, n_target)).astype(int)
    return interpolated_labels


def get_pair_wise_avgs(pair1_vals, pair2_vals, max_comparisons = 1000, random_state=None):
    rng = np.random.default_rng(random_state)
    # n_samples = min(len(pair1_vals), len(pair2_vals))
    # n_samples = min(n_samples, max_comparisons)
    n_samples = 1000
    idx1 = rng.integers(0, len(pair1_vals), size=n_samples)
    idx2 = rng.integers(0, len(pair2_vals), size=n_samples)

    diffs = pair1_vals[idx1] - pair2_vals[idx2]

    return diffs


def plot_neural_activity_heatmap(columns, neural_labels_all, col_titles, neural_labels_to_plot = None, border_neurons = None):
    
    if neural_labels_to_plot is None: 
        neural_labels_to_plot  = neural_labels_all
    neural_labels_to_plot = np.array(neural_labels_to_plot)
    indices = [np.where(neural_labels_all == val)[0][0] for val in neural_labels_to_plot]
    
    
    
    columns = np.array(columns)
    n_rows, n_cols = columns.shape
    fig, axs = plt.subplots( 1, n_cols ,figsize=(12, 30))
    
    
    for i in range(n_cols):
        col, title = columns[:, i], col_titles[i]
        axs[i].imshow(col[indices][:,None], aspect="auto", cmap="coolwarm", vmin = -2, vmax=2)
        axs[i].set_title(title)
    

    ax = axs[0]
    ax.set_yticks(np.arange(len(neural_labels_to_plot)), labels=neural_labels_to_plot, #rotation = 45, 
                  fontsize=10)
    

    if border_neurons is not None:
            for neuron in border_neurons: 
                i = np.argwhere(neural_labels_to_plot == neuron)[0]
                for ax in axs: 
                    ax.axhline(i-0.5, color = "black")
    