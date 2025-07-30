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


from plotting_utils import plot_probabilities_hist, plot_hist_across_conditins_and_get_KS_stats
from categorize_reor_fncs import get_state_start_ends


def get_motor_state_start_end_is_rslds_start_end_specific_1(motor_states, rslds_states, z, q_z):

    state_to_track_start_end_is = {}
    state_to_lbl = {}
    rslds_pairs = [(x,y) for x in rslds_states for y in rslds_states]
    state = 0
    for motor_state in motor_states:#[0,1,2]:
        for rslds_start_state, rslds_end_state in rslds_pairs: 
            
            state_lbl = (motor_state,(rslds_start_state, rslds_end_state) )
            state_to_lbl[state] = state_lbl
            
            all_tracks = []
            all_starts = []
            all_ends = []
            for track, (z_w, q_z_w) in enumerate(zip( z, q_z)):# (exp_date, behseq) in exp_date_to_beh.items():
                beh_starts, beh_ends = get_state_start_ends(motor_state, z_w,  starts= None, ends = None)
                
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

                all_tracks.append(np.ones(starts.shape[0])*track)
                all_starts.append(starts)
                all_ends.append(ends)
            state_to_track_start_end_is[state] = np.concatenate([
                                                    np.concatenate(all_tracks)[:,None],
                                                    np.concatenate(all_starts)[:,None],
                                                    np.concatenate(all_ends)[:,None]
                                                ], axis = 1).astype('int')
            state+=1

    return state_to_lbl,  state_to_track_start_end_is

def get_features_from_track_start_end_is(feature_to_dict, state_to_track_start_end_is, features = None):
    feature_state_to_feature_vals = {feature: {} for feature in feature_to_dict.keys()}
    # states = state_to_start_end_is.keys()
    if features is None: 
        features = feature_to_dict.keys()
    
    for feature in features: 
        feature_params = feature_to_dict[feature]
        feature_mat = feature_params["feature_mat"]
        feature_func = feature_params["feature_fnc"]
        print("feature", feature)
        if  feature == "duration (s)": 
            for state, track_start_end_is in state_to_track_start_end_is.items():
                feature_state_to_feature_vals["duration (s)"][state] = track_start_end_is[:,2] - track_start_end_is[:, 1]
            continue
        else:
            for state, track_start_end_is in state_to_track_start_end_is.items():
                feature_vals = []
                
                # print("track_start_end_is", track_start_end_is)
                for track, start, end in track_start_end_is:
                    feature_vals.append(feature_func(feature_mat[track, start:end]))

                feature_state_to_feature_vals[feature][state] = np.array(feature_vals)


    return feature_state_to_feature_vals
        
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
    ax.tick_params(axis='x', labelsize=20) 
    ax.tick_params(axis='y', labelsize=20) 

    # Add text annotations
    for i in range(transition_matrix.shape[0]):
        for j in range(transition_matrix.shape[1]):
            value = transition_matrix[i, j]
            if value > 0:
                # ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.5 else "black")
                # ax.text(j, i, f"{value:.2f}", ha="center", va="center", color= "black")
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color= "white", fontsize=20 )

    

    return fig, ax


def get_rslds_pair_counts(state_to_track_start_end_is, state_to_label, motor_states, rslds_states, plot= True,rslds_state_num_to_name = None ):
    label_to_state = {val:key for key, val in state_to_label.items()}
    rslds_pairs = [(x,y) for x in rslds_states for y in rslds_states]
    motor_state_to_counts ={}
    for motor_state in motor_states:#[0,1,2]:
        counts = np.zeros((len(rslds_states),  len(rslds_states)))
        for start, end in rslds_pairs:
            state = label_to_state[(motor_state, (start, end))]
            counts[start, end] = state_to_track_start_end_is[state].shape[0]
        motor_state_to_counts[motor_state] = counts
        if plot:
            visualize_transition_matrix(counts, title = f"start, end rslds state of each {motor_state}",   labels= [rslds_state_num_to_name[i] for i in range(3)] )    
    return motor_state_to_counts

def plot_feature_hists_motor_states(feature_state_to_feature_vals,feature_to_dict, rslds_state_to_color,  state_to_label, motor_states, rslds_states,
                                    motor_state_num_to_name, 
                                    rslds_state_num_to_name,
                                    
                                    features = None):
    label_to_state = {val:key for key, val in state_to_label.items()}
    
    # rslds_pairs = [(x,y) for x in rslds_states for y in rslds_states]
    if features is None: 
        features = feature_state_to_feature_vals.keys()
        
    for feature in features: 
        state_to_feature_vals  = feature_state_to_feature_vals[feature]
        
        for motor_state in motor_states:#[0,1,2]:
            for rslds_start_state in rslds_states:
                fig1, ax1 = plt.subplots()
                for rslds_end_state in rslds_states:
                    state = label_to_state[(motor_state, (rslds_start_state, rslds_end_state ))]
                    feature_vals = state_to_feature_vals[state]

                    ax1.hist(np.array(feature_vals),
                                            # bins= feature_to_dict[feature]["bins"], 
                                            bins = feature_to_dict[feature]["bins"], # np.arange(0, 15, 1)
                                            color = rslds_state_to_color[rslds_end_state], 
                                            alpha = 0.3)
        
                    ax1.set_title(f"{feature}; \nmotor state: {motor_state_num_to_name[motor_state]};  \n rslds start:{rslds_state_num_to_name[rslds_start_state]}")
            
            
                rslds_state_to_feature = {rslds_end_state: remove_nans( state_to_feature_vals[label_to_state[(motor_state, (rslds_start_state, rslds_end_state ))]]) for  rslds_end_state in rslds_states}
                fig1, ax1 = plt.subplots()
                if np.all(np.array([feature_val.size >0 for feature_val in rslds_state_to_feature.values()])):
                    plot_hist_across_conditins_and_get_KS_stats(rslds_state_to_feature,  rslds_state_to_color,
                                                            title = f"{feature} \n motor state: {motor_state_num_to_name[motor_state]}; \n rslds start:{rslds_state_num_to_name[rslds_start_state]}", 
                                                            ylabel = feature_to_dict[feature]["ylabel"],
                                                            xlabel = f"{feature}",
                                                            stat_color = "tomato" ,
                                                            bins = feature_to_dict[feature]["bins"], # np.arange(0, 15, 1)
                                                            ylim = feature_to_dict[feature]["ylim"], 
                                                            alpha = 0.3, 
                                                        #    linewidth=2, 
                                                            fill =  True,
                                                            fig = fig1, 
                                                            ax = ax1 , 
                                                            print_sig = False,
                                                            print_effect_size = False, 
                                                            legend = False,
                                                            plot_medians = True
                                                            )
                    
                    
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
    


def plot_neural_activity(neural_feature_to_dict,state_to_track_start_end_is, label_to_state, rslds_states, motor_states, motor_state_num_to_name, neural_labels, neural_labels_cropped):
    # pairs = list(pair_to_vals_dict.keys())
    state_to_neural_activity_means = get_features_from_track_start_end_is(neural_feature_to_dict, state_to_track_start_end_is, features = ["neural_activity_mean"] )["neural_activity_mean"]
    pairs = [(x,y) for x in rslds_states for y in rslds_states]
    for motor_state in motor_states:
    
            
        

        for i in range(len(pairs)):
            
            pair1 = pairs[i]
            state1 = label_to_state[(motor_state, pair1)]
            pair1_vals = np.array(state_to_neural_activity_means[state1])
            n_pair1_instances = len(pair1_vals)
            if n_pair1_instances<15:
                continue
            print(pair1 , n_pair1_instances)
            pair_specific_mean_diffs = []

            col_titles = []

            for j in range(len(pairs)):
                # ax = axs[k]
                if i ==j:
                    continue
                pair2 = pairs[j]
                state2 = label_to_state[(motor_state, pair2)]
                pair2_vals = np.array(state_to_neural_activity_means[state2])
                n_pair1_instances = len(pair1_vals)
                n_pair2_instances = len(pair2_vals)
                if n_pair2_instances<15:
                    print("skipping, ", pair2, n_pair2_instances)
                    continue
                diffs  = get_pair_wise_avgs(pair1_vals, pair2_vals, max_comparisons = 1000, random_state=None)
                mean_diffs = np.nanmean(diffs, axis = 0 )[:, None]
                ##plot heat map of diffs with neaul activity as labels 
            
                # im = ax.imshow(mean_diffs, aspect="auto", cmap="coolwarm", vmin = -0.3, vmax=0.3)
                pair_specific_mean_diffs.append(mean_diffs)
                col_titles.append(f"{pair1}, {pair2}, {motor_state_num_to_name[motor_state]}")

            pair_specific_mean_diffs = np.concatenate(pair_specific_mean_diffs, axis = 1)
            plot_neural_activity_heatmap(pair_specific_mean_diffs, neural_labels, col_titles, neural_labels_to_plot = neural_labels_cropped, border_neurons = ['dF - AVE'
        , "dF - AVE", "F - OLL"])
