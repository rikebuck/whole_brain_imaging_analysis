# %%
import copy
import socket
import os
import pickle 
# import ssm #need to/ keep 
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')

from scipy.ndimage import gaussian_filter1d
from get_data.read_neuro_beh_data import get_exp_features, load_json
from get_data.process_neural_data import get_derivative_of_neural_activity, get_neural_activity_from_labels, get_neural_activity_and_labels
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification
# from beh_classification.beh_classification_utils import get_start_end_is_of_ones_in_binary_array
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import NearestNeighbors

from load_data_fncs import load_all_data, get_exp_dates
from analyse_rslds_utils import make_behavior_decoder, input_slds

from spont_rev_prediction_fncs import plot_mean_line_graph_with_individ_traces, align_rev_to_stim_start_from_start_end_is, filter_start_end_is_by_start_time

# %%
# import numpy as np
# import warnings
# from sklearn.exceptions import ConvergenceWarning
# from predict_latency_prob_from_features_donut import prep_FB_inputs_select_features_only_no_onset, fit_eval_linearSVC, visualize_model_classification
# from predict_latency_prob_from_features_donut import fit_eval_lasso, visualize_model_regression

# def mark_future_events(arr, start=1, end=10):
#     arr = np.asarray(arr)
#     out = np.zeros_like(arr, dtype=float)

#     # Set NaNs where arr is 1
#     out[arr == 1] = np.nan

#     n = len(arr)
#     for i in range(n):
#         if arr[i] == 0:
#             # Ensure window stays in bounds
#             window = arr[i + start : min(i + end + 1, n)]
#             if np.any(window == 1):
#                 out[i] = 1.0

#     return out
# def mark_future_events_with_distance(arr, start=1, end=10):
#     arr = np.asarray(arr)
#     out_dist = np.zeros_like(arr, dtype=float)

#     # Set NaNs where arr is 1
#     out_dist[arr == 1] = np.nan

#     n = len(arr)
#     for i in range(n):
#         if not arr[i]:
#             for offset in range(start, end + 1):
#                 j = i + offset
#                 if j < n and arr[j] == 1:
#                     out_dist[i] = offset
#                     break  # take the first (nearest) future 1
#             # If no 1 found, stays 0

#     return out_dist
# def get_upcoming_revs(rev_bin, win_start = 1, win_end = 10):
#     n_tracks, n_frames = rev_bin.shape
#     all_upcoming_revs = []
#     # features = []
#     for track in range(n_tracks):
        
#         rev_bin_track = rev_bin[track, :]
#         upcoming_revs = mark_future_events(rev_bin_track, start=win_start, end=win_end)
#         all_upcoming_revs.append(upcoming_revs[:, None])

#     return np.concatenate(all_upcoming_revs, axis = 1).T
# def get_time_to_rev(rev_bin, win_start = 1, win_end = 10):
#     n_tracks, n_frames = rev_bin.shape
#     all_time_to_revs = []
#     # features = []
#     for track in range(n_tracks):
        
#         rev_bin_track = rev_bin[track, :]
       
#         time_to_rev = mark_future_events_with_distance(rev_bin_track, start=win_start, end=win_end)
        
#         all_time_to_revs.append(time_to_rev[:,None])
        
#         # fig, ax = plt.subplots(3,1, figsize = (25,5))
#         # rev_bin_track1 = rev_bin_track[~np.isnan(rev_bin_track)]
#         # upcoming_revs = mark_future_events(rev_bin_track, start=win_start, end=win_end)
#         # upcoming_revs1 = upcoming_revs[~np.isnan(rev_bin_track)]
#         # time_to_rev1 = time_to_rev[~np.isnan(rev_bin_track)]
#         # ax[0].imshow(rev_bin_track1[:, None].T, aspect='auto')
#         # ax[1].imshow(upcoming_revs1[:, None].T, aspect='auto')
#         # ax[1].imshow(time_to_rev1[:, None].T, aspect='auto')
        
#         # if track == 6: 
#         #     break
#     return np.concatenate(all_time_to_revs, axis = 1).T
    
    
# def prep_data_spont_rev_prob(features_list, feature_labels, rev_bin, n_timesteps = 3, win_start = 1, win_end = 10, fps = 1, flatten = True):
#     n_tracks, n_frames = rev_bin.shape
#     all_upcoming_revs = get_upcoming_revs(rev_bin, win_start = win_start, win_end = win_end)
#     Y_time_to_revs = get_time_to_rev(rev_bin, win_start = win_start,  win_end = win_end)
    
#     X_features,feature_names =  prep_FB_inputs_select_features_only_no_onset(
    
#                 features_list,
#                 feature_labels,     
#                 n_timesteps,
#                 fps = fps
#                 )
#     Y_time_to_revs = Y_time_to_revs[:, n_timesteps:]
#     Y_upcoming_revs = all_upcoming_revs[:, n_timesteps:]
#     if flatten: 
#         Y_upcoming_revs_flat = Y_upcoming_revs.reshape(-1) 
#         Y_time_to_revs_flat = Y_time_to_revs.reshape(-1)
#         X_features_flat = X_features.reshape(-1, X_features.shape[-1])
#         non_nan_is = np.logical_and(np.logical_not(np.isnan(Y_upcoming_revs_flat)), np.all(np.logical_not(np.isnan(X_features_flat)), axis = 1)).astype('bool')
#         return Y_upcoming_revs_flat[non_nan_is], X_features_flat[non_nan_is, :], Y_time_to_revs_flat[non_nan_is], feature_names
#     else: 
#         return Y_upcoming_revs, X_features, Y_time_to_revs_flat, feature_names 
    
    
# def prep_data_spont_rev_latency(features_list, feature_labels, rev_bin, n_timesteps = 3, win_start = 1, win_end = 10, fps = 1, flatten = True):
#     Y_upcoming_revs, X_features, Y_time_to_revs_flat, feature_names  = prep_data_spont_rev_prob(features_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = flatten)
#     will_rev_is =(Y_upcoming_revs==1).astype('bool')
#     print("CHECK TRUE!:" , np.all(Y_time_to_revs_flat[will_rev_is] >0))
#     # return Y_upcoming_revs[will_rev_is], X_features[will_rev_is], Y_time_to_revs_flat[will_rev_is], feature_names
#     return Y_time_to_revs_flat[will_rev_is], X_features[will_rev_is], Y_time_to_revs_flat[will_rev_is], feature_names


# def downsample_majority_np(X, y, majority_label=0, random_state=42):
#     rng = np.random.default_rng(random_state)

#     # Boolean masks
#     majority_mask = y == majority_label
#     minority_mask = ~majority_mask

#     # Split arrays
#     X_majority = X[majority_mask]
#     y_majority = y[majority_mask]
#     X_minority = X[minority_mask]
#     y_minority = y[minority_mask]

#     # Downsample majority
#     n_minority = len(y_minority)
#     indices = rng.choice(len(X_majority), size=n_minority, replace=False)
#     X_majority_down = X_majority[indices]
#     y_majority_down = y_majority[indices]

#     # Combine and shuffle
#     X_bal = np.vstack([X_majority_down, X_minority])
#     y_bal = np.concatenate([y_majority_down, y_minority])
    
#     perm = rng.permutation(len(y_bal))
#     return X_bal[perm], y_bal[perm]


# def balanced_train_test_split(X, y, n_train, n_test, random_state=42):
#     rng = np.random.default_rng(random_state)
#     classes = np.unique(y)
#     X_train_list, y_train_list = [], []
#     X_test_list, y_test_list = [], []

#     for cat in classes:
#         # Find indices for this class
#         cls_indices = np.where(y == cat)[0]
#         if len(cls_indices) < n_train + n_test:
#             raise ValueError(f"Not enough samples of class {cat}: needed {n_train + n_test}, got {len(cls_indices)}")

#         # Shuffle and split
#         cls_indices = rng.permutation(cls_indices)
#         train_idx = cls_indices[:n_train]
#         test_idx = cls_indices[n_train:n_train + n_test]

#         # Append data
#         X_train_list.append(X[train_idx])
#         y_train_list.append(y[train_idx])
#         X_test_list.append(X[test_idx])
#         y_test_list.append(y[test_idx])

#     # Combine and shuffle
#     X_train = np.vstack(X_train_list)
#     y_train = np.concatenate(y_train_list)
#     X_test = np.vstack(X_test_list)
#     y_test = np.concatenate(y_test_list)

#     # Final shuffle
#     train_perm = rng.permutation(len(y_train))
#     test_perm = rng.permutation(len(y_test))

#     return (
#         X_train[train_perm], X_test[test_perm],
#         y_train[train_perm], y_test[test_perm]
#     )
    
# # def downsample_all_classes_np_multiclass(X, y,classes_timeseries,  random_state=42):
# #     rng = np.random.default_rng(random_state)
# #     X = np.asarray(X)
# #     y = np.asarray(y)

# #     classes, counts = np.unique(classes_timeseries, return_counts=True)
# #     min_count = np.min(counts)

# #     X_bal_list = []
# #     y_bal_list = []

# #     for cls in classes:
# #         cls_indices = np.where(classes_timeseries == cls)[0]
# #         sampled_indices = rng.choice(cls_indices, size=min_count, replace=False)
# #         X_bal_list.append(X[sampled_indices])
# #         y_bal_list.append(y[sampled_indices])

# #     # Combine and shuffle
# #     X_bal = np.vstack(X_bal_list)
# #     y_bal = np.concatenate(y_bal_list)

# #     perm = rng.permutation(len(y_bal))
# #     return X_bal[perm], y_bal[perm]



# # def downsample_all_classes_and_split(arrays, label_index, test_size=0.2, random_state=42):
# #     """
# #     Parameters:
# #     - arrays: list of np.arrays, all with same first dimension
# #     - label_index: index of the array in `arrays` that contains the class labels (1D array)
# #     - test_size: proportion of data to allocate to test set
# #     - random_state: seed for reproducibility

# #     Returns:
# #     - train_arrays: list of arrays for training
# #     - test_arrays: list of arrays for testing
# #     """

# #     rng = np.random.default_rng(random_state)
# #     arrays = [np.asarray(arr) for arr in arrays]
# #     y = arrays[label_index]
    
# #     classes, counts = np.unique(y, return_counts=True)
# #     min_count = np.min(counts)

# #     # Collect balanced indices
# #     balanced_indices = []
# #     for cls in classes:
# #         cls_indices = np.where(y == cls)[0]
# #         sampled_indices = rng.choice(cls_indices, size=min_count, replace=False)
# #         balanced_indices.extend(sampled_indices)

# #     # Shuffle balanced indices
# #     balanced_indices = rng.permutation(balanced_indices)

# #     # Extract balanced data for all arrays
# #     balanced_arrays = [arr[balanced_indices] for arr in arrays]
# #     y_bal = balanced_arrays[label_index]

# #     # Get train/test indices with stratification
# #     train_idx, test_idx = train_test_split(
# #         np.arange(len(y_bal)), test_size=test_size, random_state=random_state, stratify=y_bal
# #     )

# #     # Split all arrays using same indices
# #     train_arrays = [arr[train_idx] for arr in balanced_arrays]
# #     test_arrays = [arr[test_idx] for arr in balanced_arrays]

# #     return train_arrays, test_arrays

# def downsample_all_classes_and_split_1(arrays, label_index, test_size=0.2, random_state=42, max_per_class=1000):
#     """
#     Parameters:
#     - arrays: list of np.arrays, all with same first dimension
#     - label_index: index of the array in `arrays` that contains the class labels (1D array)
#     - test_size: proportion of data to allocate to test set
#     - random_state: seed for reproducibility
#     - max_per_class: maximum number of samples to retain per class

#     Returns:
#     - train_arrays: list of arrays for training
#     - test_arrays: list of arrays for testing
#     """
#     from sklearn.model_selection import train_test_split
#     import numpy as np

#     rng = np.random.default_rng(random_state)
#     arrays = [np.asarray(arr) for arr in arrays]
#     y = arrays[label_index]
    
#     classes, counts = np.unique(y, return_counts=True)
#     min_count = min(np.min(counts), max_per_class)

#     # Collect balanced indices
#     balanced_indices = []
#     for cls in classes:
#         cls_indices = np.where(y == cls)[0]
#         sampled_count = min(len(cls_indices), max_per_class)
#         sampled_indices = rng.choice(cls_indices, size=sampled_count, replace=False)
#         balanced_indices.extend(sampled_indices)

#     # Shuffle balanced indices
#     balanced_indices = rng.permutation(balanced_indices)

#     # Extract balanced data for all arrays
#     balanced_arrays = [arr[balanced_indices] for arr in arrays]
#     y_bal = balanced_arrays[label_index]

#     # Get train/test indices with stratification
#     train_idx, test_idx = train_test_split(
#         np.arange(len(y_bal)), test_size=test_size, random_state=random_state, stratify=y_bal
#     )

#     # Split all arrays using same indices
#     train_arrays = [arr[train_idx] for arr in balanced_arrays]
#     test_arrays = [arr[test_idx] for arr in balanced_arrays]

#     return train_arrays, test_arrays
    
# def balanced_train_test_split_1(X, y,Y_time_to_revs_flat,  n_train, n_test, random_state=42):
#     rng = np.random.default_rng(random_state)
#     classes = np.unique(y)
#     X_train_list, y_train_list = [], []
#     X_test_list, y_test_list = [], []
#     Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = [], []

#     for cat in classes:
#         # Find indices for this class
#         cls_indices = np.where(y == cat)[0]
#         if len(cls_indices) < n_train + n_test:
#             raise ValueError(f"Not enough samples of class {cat}: needed {n_train + n_test}, got {len(cls_indices)}")

#         # Shuffle and split
#         cls_indices = rng.permutation(cls_indices)
#         train_idx = cls_indices[:n_train]
#         test_idx = cls_indices[n_train:n_train + n_test]

#         # Append data
#         X_train_list.append(X[train_idx])
#         y_train_list.append(y[train_idx])
#         Y_time_to_revs_flat_train.append(Y_time_to_revs_flat[train_idx])
        
        
#         X_test_list.append(X[test_idx])
#         y_test_list.append(y[test_idx])
#         Y_time_to_revs_flat_test.append(Y_time_to_revs_flat[test_idx])

#     # Combine and shuffle
#     X_train = np.vstack(X_train_list)
#     y_train = np.concatenate(y_train_list)
#     X_test = np.vstack(X_test_list)
#     y_test = np.concatenate(y_test_list)
    
#     Y_time_to_revs_flat_train = np.concatenate(Y_time_to_revs_flat_train)
#     Y_time_to_revs_flat_test = np.concatenate(Y_time_to_revs_flat_test)

#     # Final shuffle
#     train_perm = rng.permutation(len(y_train))
#     test_perm = rng.permutation(len(y_test))

#     return (
#         X_train[train_perm], X_test[test_perm],
#         y_train[train_perm], y_test[test_perm], 
#         Y_time_to_revs_flat_train[train_perm], Y_time_to_revs_flat_test[test_perm]
#     )


# def predict_spont_revs_from_features(   feature_list, 
#                                      feature_labels, rev_bin, fps, 
#                                      win_start, win_end,
#                                      model_fit_fnc, score_label,
#                                      result_visualization_fnc,
#                                      prep_data_fnc, 
#                                      title_label = "spont revs", 
#                                      all_timesteps = None, 

#                                     n_splits = 10, 
#                                     test_size = 0.3, 
#                                     # n_train = 1000, n_test = 300, 

#                                     single_feature = False, 
#                                     n_pre_start = 0 , 
#                                     max_per_class = 1000, 
            

#                                     ):

#     i = -1 #spont
#     with warnings.catch_warnings():
#         warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
#         fig, ax = plt.subplots()
#         best_models = {} 
#         all_scores_per_i = {}
#         results_dict = {}


#         score_means = []
#         score_stds = []
#         n_timesteps_list = []
#         n_frames = rev_bin.shape[1]
#         if all_timesteps is None: 
#             all_timesteps = range(1, 15, 3)

#         for n_timesteps in all_timesteps:
#             print("i, n_timesteps, win_end",i,  n_timesteps, win_end)
#             scores = []
#             models = []

#             # y,  X, Y_time_to_revs_flat, feature_names = prep_data_spont_rev_prob(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)

#             y,  X, Y_time_to_revs_flat, feature_names = prep_data_fnc(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)

#             # y = y[np.logical_not(np.isnan(Y_time_to_revs_flat))]
#             # X = X[np.logical_not(np.isnan(Y_time_to_revs_flat))]
#             # Y_time_to_revs_flat = Y_time_to_revs_flat[np.logical_not(np.isnan(Y_time_to_revs_flat))]
            
#             unique, counts = np.unique(Y_time_to_revs_flat, return_counts = True)
#             for un, cnt in zip( unique, counts):
#                     print("Y_time_to_revs_flat all unqiuecounts", un, cnt)
#             ###if want to see how far back of a window cna be used) 
#             y = y[n_pre_start:]
#             Y_time_to_revs_flat = Y_time_to_revs_flat[n_pre_start:]
#             X = X[:(X.shape[0]-n_pre_start)]
#             # X_bal, y_bal = downsample_majority_np(X, y, majority_label=0, random_state=42)
            
#             for split_idx in range(n_splits):
#                 # X_train, X_test, y_train, y_test = train_test_split(
#                 #     X, Y, test_size=test_size, random_state=split_idx)
                
#                 # X_train, X_test, y_train, y_test = train_test_split(
#                 #     X_bal, y_bal, test_size=test_size, random_state=split_idx)
                
#                 # X_train, X_test, y_train, y_test = balanced_train_test_split(X, y, n_train, n_test, random_state=split_idx)
#                 # train_arrays, test_arrays = downsample_all_classes_and_split([X, y, Y_time_to_revs_flat], 2, test_size=0.2, random_state=split_idx)
#                 print("Y_time_to_revs_flat all unqiuecounts","split_idx", i,  n_timesteps, win_end, split_idx)
#                 train_arrays, test_arrays = downsample_all_classes_and_split_1([X, y, Y_time_to_revs_flat], 1, test_size=test_size, random_state=split_idx, max_per_class=max_per_class)
#                 X_train, y_train, Y_time_to_revs_flat_train = train_arrays
#                 X_test, y_test, Y_time_to_revs_flat_test =  test_arrays
                
#                 # unique, counts = np.unique(Y_time_to_revs_flat_train, return_counts = True)
#                 # for un, cnt in zip( unique, counts):
#                 #     print("Y_time_to_revs_flat_train all unqiuecounts", un, cnt)
                
#                 # unique, counts = np.unique(Y_time_to_revs_flat_test, return_counts = True)
#                 # for un, cnt in zip( unique, counts):
#                 #     print("Y_time_to_revs_flat_test all unqiuecounts", un, cnt)
#                 # X_train, X_test, y_train, y_test, Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = balanced_train_test_split_1(X, y,Y_time_to_revs_flat,  n_train, n_test, random_state=split_idx)
                
                
#                 best_model, y_pred, score = model_fit_fnc(X_train, X_test, y_train, y_test )
#                 scores.append(score)
#                 models.append((best_model,X_train, X_test, y_train, y_test, Y_time_to_revs_flat_train, Y_time_to_revs_flat_test))

#             scores_mean = np.mean(scores)
#             scores_std = np.std(scores)
#             best_model_idx = np.argmax(scores)
#             best_model_overall = models[best_model_idx]

#             # Save results
#             n_timesteps_list.append(n_timesteps)
#             # all_r2_means.append(r2s)
#             all_scores_per_i[(i, n_timesteps)] = scores
#             score_means.append(scores_mean)
#             score_stds.append(scores_std)
#             results_dict[(i, n_timesteps)] = (scores_mean, scores_std)
#             best_models[(i, n_timesteps)] = best_model_overall
            

#         # cmap1 = cm.get_cmap('tab10', len(resampled_onsets))
#         ###visualiza
#         n_timesteps = all_timesteps[-2]
#         ax.errorbar(np.array(n_timesteps_list)/fps, score_means, yerr=score_stds, #color=cmap1(i),
#                     fmt='-o', capsize=5, label = f"stim {i}")

#         ax.set_title( f"{title_label}; stim{i}; win start {win_start}; win end {win_end}" )
#         ax.set_xlabel("max time delay (sec)")
#         ax.set_ylabel(f"{score_label}")
#         ax.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         model_label= f"{title_label}; stim{i}; win start {win_start}; win end {win_end}"#; lasso_a{np.round(alpha,2)}"
#         best_model, X_train, X_test, y_train, y_test,  Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = best_models[(i, n_timesteps)]



        
#         # best_model,X_train, X_test, y_train, y_test,  Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = best_models[(i, n_timesteps)]
        
#         # if single_feature: 
#         #     feature_lbl = feature_labels[0]
#         #     feature_names = np.array([[f"{feature_lbl}_t-{np.round((n_timesteps - t - 1)/fps, 2)}"]
#         #                         for t in range(n_timesteps)]).flatten()
#         #     result_visualization_fnc(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False, coeffs = model.coef_[0])#, xlim = [-2,1])
            
#         # else: ]
#         _,  _,_,  feature_names = prep_data_spont_rev_prob(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)
        
        
#         fig, ax = plt.subplots()
#         y_pred = best_model.predict(X_test)
#         if prep_data_fnc == prep_data_spont_rev_prob:
#             print("prep_data_fnc == prep_data_spont_rev_prob")
#             ax.hist(Y_time_to_revs_flat_test[(y_test==y_pred).astype('bool')],alpha = 0.3, density = True,  label = "hit")
#             unique, counts = np.unique(Y_time_to_revs_flat_test[(y_test==y_pred).astype('bool')], return_counts = True)
#             for un, cnt in zip( unique, counts):
#                     print("hit unqiuecounts", un, cnt)
            
#             ax.hist(Y_time_to_revs_flat_test[(y_test!=y_pred).astype('bool')],alpha = 0.3, density = True,  color = "purple", label = "miss")
            
#             unique, counts = np.unique(Y_time_to_revs_flat_test[(y_test!=y_pred).astype('bool')], return_counts = True)
#             for un, cnt in zip( unique, counts):
#                     print("miss unqiuecounts", un, cnt)
#             ax.legend()
#             # ax.scatter(Y_time_to_revs_flat_test, err )
#             ax.set_xlabel("time_to_rev")
#             ax.set_ylabel("prob")
#             ax.set_title(model_label)
#         result_visualization_fnc(best_model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False)#, xlim = [-2,1])
        
        
#         fig, ax = plt.subplots()
#         ax.hist(Y_time_to_revs_flat_test)
#         ax.set_title("Y_time_to_revs_flat_test")

#         print("feature_names_all",feature_names)
            
#         return best_models
    
            
# def predict_spont_revs_from_features_single_feature(   
#                                                     feature_list, 
#                                      feature_labels, rev_bin, fps, 
#                                      win_start, win_end,
#                                      model_fit_fnc, score_label,
#                                      result_visualization_fnc,
#                                      prep_data_fnc, 
#                                      title_label = "spont revs", 
#                                      all_timesteps = None, 

#                                     n_splits = 10, 
#                                     test_size = 0.3, 
#                                     # n_train = 1000, n_test = 300, 

#                                     single_feature = False, 
#                                     n_pre_start = 0 , 
            

#                                     ):
#     feature_to_best_models = {}
#     for feature, feature_lbl in zip(feature_list, feature_labels):
#         print(feature_lbl)
#         best_models = predict_spont_revs_from_features(   [feature], 
#                                         [feature_labels],
#                                         rev_bin, fps, 
#                                         win_start, win_end,
#                                         model_fit_fnc, score_label,
#                                         result_visualization_fnc,
#                                         prep_data_fnc, 
#                                         title_label = f"{feature_lbl}; {title_label}", 
#                                         all_timesteps = all_timesteps, 

#                                         n_splits = n_splits, 
#                                         test_size = test_size, 
#                                         # n_train = 1000, n_test = 300, 

#                                         single_feature = single_feature, 
#                                         n_pre_start = n_pre_start
                

#                                         )
#         feature_to_best_models[feature_lbl] = best_models
#     return feature_to_best_models

# %%

# def get_transitions(stateseq):
#     ''' copied from keypoint moseq github: keypoint-moseq/keypoint_moseq/util.py'''
#     transitions = np.nonzero(stateseq[1:] != stateseq[:-1])[0] + 1
#     starts = np.insert(transitions, 0, 0)
#     ends = np.append(transitions, len(stateseq))
#     return  transitions, starts, ends



    
# def get_state_start_ends(state, stateseq,  starts= None, ends = None):
#     if starts is None: 
#         transitions, starts, ends = get_transitions(stateseq)
#     # for state1, state2 in state_pairs:
#     state_start_is = np.argwhere(state==stateseq[starts[:-1]]).flatten()
    
#     state_starts = starts[state_start_is]
#     state_ends = ends[state_start_is] ##check 
    
#     return state_starts, state_ends

# def plot_mean_line_graph_with_individ_traces(data, n_pre_frames, fps = 6,  fig = None, ax = None, seconds = None,vline = None, title = "", 
#                                              xlabel ="seconds post-stim start", 
#                                              ylabel ="speed (um/s)", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                              max_n_tracks = 30,line_width=1.0
                                             
#                                              ):
#     if fig is None:# types.NoneType 
#         fig, ax = plt.subplots()
#     if data.shape[0] ==0:
#         return fig, ax
    
#     n_tracks, n_frames = data.shape
#     if n_tracks>max_n_tracks:
#         track_is = np.arange(n_tracks)
#         np.random.shuffle(track_is)
#         sampled_track_is = track_is[:max_n_tracks]
#         sampled_data = data[sampled_track_is,:]
#     else:
#         sampled_data = data
#     average_row = np.nanmean(sampled_data, axis=0)
   
#     # Plot each row with grayed lines
#     if seconds is None:
#         n_frames = sampled_data.shape[1]
#         # seconds = np.arange(n_frames*params.n_stim_types)/params.fps
#         seconds = np.arange(n_frames)/fps - n_pre_frames/fps
#         vline = 0
#     for row in sampled_data:
#         ax.plot(seconds, row, color='gray', alpha=0.5, linewidth=line_width)

#     # Plot the average from BehaviorAnalyzer.PlottingFunctions.get_event_speed_line_plots row with a more prominent line
#     ax.plot(seconds, average_row, color=color, label='Average', linewidth=2)

#     # Customize labels and legend
#     if xlabel is not None: 
#         ax.set_xlabel(xlabel)
#     if ylabel is not None: 
#         ax.set_ylabel(ylabel, fontsize = ylabel_font_size)
#     if title is not None: 
#         ax.set_title(title)
#     if ylim is not None:
#         ax.set_ylim(ylim)
#     if vline is not None:
#         ax.axvline(x=vline, color='black', linestyle='--')

#     return fig, ax


# # def align_feature_to_rev_start( exp_date_to_beh, exp_date_to_feature, n_pre_frames = 60, max_plotting_length_frames = 500, 
# #                                rev_duration_min_frames = None, rev_duration_max_frames = None ,  rev_state = 1
# #                                ):
   
# #     buffered_event_features = []
# #     for exp_date, featureseq in exp_date_to_feature.items():
# #         # rsldsseq = date_to_discrete_rslds_states[exp_date]
# #         behseq = exp_date_to_beh[exp_date]
# #         featureseq = exp_date_to_feature[exp_date]
# #         state_starts, state_ends = get_state_start_ends(rev_state , behseq,  starts= None, ends = None)
# #         for rev_start_i,rev_end_i in zip( state_starts, state_ends):
# #             rev_duration_frames = rev_end_i-rev_start_i
# #             if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
# #                 continue
# #             if rev_duration_max_frames is not None and rev_duration_frames>rev_duration_max_frames:
# #                 continue
# #             frame_start = max(rev_start_i-n_pre_frames, 0)
# #             frame_end = min(rev_start_i+max_plotting_length_frames, rev_end_i)
        
# #             n_rev_frames = frame_end-rev_start_i
# #             n_fwd_frames = rev_start_i-frame_start
# #             buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
# #             buffered_event_feature[:] = np.nan
# #             buffered_event_feature[n_pre_frames+-1*n_fwd_frames:n_pre_frames+n_rev_frames] = featureseq[frame_start:frame_end]
# #             buffered_event_features.append(buffered_event_feature[:, None])

# #     if len(buffered_event_features)==0: 
# #         return np.array([])

# #     # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
# #     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
# #     return buffered_event_features


def align_feature_to_rev_start_from_start_end_is(feature_val, event_start_end_is,  n_pre_frames = 60, max_plotting_length_frames = 500, 
                               rev_duration_min_frames = None, rev_duration_max_frames = None ,  rev_state = 1, starts =None, ends = None, 
                            #    earliest_start_frame = None, earliest_end_frame = None
                               
                               ):
    
    buffered_event_features = []
    durations = event_start_end_is[:,2]-event_start_end_is[:,1]
    dur_bool = np.logical_and(durations>rev_duration_min_frames, durations<=rev_duration_max_frames).astype('bool')
    event_start_end_is_duration = event_start_end_is[dur_bool]
    for track, rev_start_i, rev_end_i in event_start_end_is_duration:
        
    # # for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
    #     # rsldsseq = date_to_discrete_rslds_states[exp_date]
        
    #     # featureseq = exp_date_to_feature[exp_date]
    #     # state_starts, state_ends = get_state_start_ends(
    #     #                                                 rev_state,
    #     #                                                 z_w,  
    #     #                                                 starts= None, 
    #     #                                                 ends = None
    #     #                                                 )
    #     # for rev_start_i,rev_end_i in zip( state_starts, state_ends):
        
    #         rev_duration_frames = rev_end_i-rev_start_i
    #         if rev_duration_frames>=rev_duration_min_frames and rev_duration_frames<rev_duration_max_frames:

            frame_start = max(rev_start_i-n_pre_frames, 0)
            frame_end = min(rev_start_i+max_plotting_length_frames, rev_end_i)
        
            n_rev_frames = frame_end-rev_start_i
            n_fwd_frames = rev_start_i-frame_start
            buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
            buffered_event_feature[:] = np.nan
            buffered_event_feature[int(n_pre_frames+-1*n_fwd_frames):int(n_pre_frames+n_rev_frames)] = feature_val[track, frame_start:frame_end]
            buffered_event_features.append(buffered_event_feature[:, None])

    if len(buffered_event_features)==0: 
        return np.array([])

    # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
    buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
    return buffered_event_features


# def align_feature_to_rev_end_from_start_end_is(feature_val, event_start_end_is,  
#                             n_pre_frames = 90, max_plotting_length_frames = 150, 
#                             rev_duration_min_frames = -1, rev_duration_max_frames = np.inf,  rev_state = 1
#     ):

#     n_frames = feature_val.shape[1]
#     buffered_event_features = []
#     durations = event_start_end_is[:,2]-event_start_end_is[:,1]
#     dur_bool = np.logical_and(durations>rev_duration_min_frames, durations<=rev_duration_max_frames).astype('bool')
#     event_start_end_is_duration = event_start_end_is[dur_bool]
#     for track, rev_start_i, rev_end_i in event_start_end_is_duration:
    
#     # for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
       
#     #     # rsldsseq = date_to_discrete_rslds_states[exp_date]
#     #     # behseq = exp_date_to_beh[exp_date]
#     #     # featureseq = exp_date_to_feature[exp_date]
#     #     state_starts, state_ends = get_state_start_ends(rev_state , z_w,  starts= None, ends = None)
#         # for rev_start_i,rev_end_i in zip( state_starts, state_ends):
#         #     rev_duration_frames = rev_end_i-rev_start_i
#         #     if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
#         #         continue
#         #     if rev_duration_max_frames is not None and rev_duration_min_frames>rev_duration_max_frames:
#         #         continue
#             frame_start = max(0, rev_end_i-n_pre_frames) # which frame to reference in freature mat as start of window to plot  
#             frame_end = min(rev_end_i+max_plotting_length_frames, n_frames)# which frame to reference in freature mat as end of window to plot
#             frame_window = frame_end-frame_start # full elngth to plot 
#             max_window_size = max_plotting_length_frames+n_pre_frames
#             if n_frames< rev_end_i+max_plotting_length_frames:
#                 # n_pre_frames = rev_end_i+max_plotting_length_frames-n_frames
#                 # window_start = max_window_size-frame_window #+rev_end_i+max_plotting_length_frames-n_frames
#                 diff = rev_end_i+max_plotting_length_frames-n_frames
#                 window_start = max_window_size-frame_window
#                 window_start = 0
#             else: 
#                 window_start = max_window_size-frame_window#n_pre_frames
#                 diff = 0
                

#             buffered_event_feature = np.zeros(max_window_size)
#             buffered_event_feature[:] = np.nan
#             buffered_event_feature[window_start:n_pre_frames+max_plotting_length_frames-diff] = feature_val[track, frame_start:frame_end]
#             buffered_event_features.append(buffered_event_feature[:, None])
                
            
#             # n_rev_frames = frame_end-rev_start_i
#             # n_fwd_frames = rev_start_i-frame_start
#             # buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
#             # buffered_event_feature[:] = np.nan
#             # buffered_event_feature[n_pre_frames+-1*n_fwd_frames:n_pre_frames+n_rev_frames] = featureseq[frame_start:frame_end]
#             # buffered_event_features.append(buffered_event_feature[:, None])
  
#     if len(buffered_event_features)==0: 
#         return np.array([])

#     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
#     return buffered_event_features

# def align_rev_to_stim_start_from_start_end_is(feature_val, event_start_end_is,  stim_start_frames, 
#                             n_pre_frames = 60, max_plotting_length_frames = 500, 
#                                rev_duration_min_frames = -1, rev_duration_max_frames = np.inf ,  rev_state = 1
#                                ):
   
#     n_frames = feature_val.shape[1]
#     buffered_event_features = []
#     durations = event_start_end_is[:,2]-event_start_end_is[:,1]
#     dur_bool = np.logical_and(durations>rev_duration_min_frames, durations<=rev_duration_max_frames).astype('bool')
#     event_start_end_is_duration = event_start_end_is[dur_bool]
#     for track, rev_start_i, rev_end_i in event_start_end_is_duration:

#             # print("n_pre_frames", n_pre_frames)
#             # print("stim_start_frames", stim_start_frames)
#             frame_start = max(stim_start_frames-n_pre_frames, 0)
#             frame_end = min(stim_start_frames+max_plotting_length_frames, n_frames)
#             # print("frame_start, frame_end", frame_start, frame_end)
#             n_poststim_frames = frame_end-stim_start_frames# n_pre_frames
#             n_prestim_frames = stim_start_frames-frame_start
#             # print("n_prestim_frames", n_prestim_frames, "n_poststim_frames", n_poststim_frames)
#             buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
#             # print("n_pre_frames+max_plotting_length_frames", n_pre_frames+max_plotting_length_frames)
#             # print("n_pre_frames+n_poststim_frames", n_pre_frames+n_poststim_frames )
#             # print("n_pre_frames-n_prestim_frames", n_pre_frames-n_prestim_frames)
#             buffered_event_feature[:] = np.nan
#             buffered_event_feature[(n_pre_frames-n_prestim_frames):(n_pre_frames+n_poststim_frames)] = feature_val[track, frame_start:frame_end]
#             buffered_event_features.append(buffered_event_feature[:, None])

#     if len(buffered_event_features)==0: 
#         return np.array([])

#     # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
#     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
#     return buffered_event_features
# def align_feature_to_rev_start( z, feature_val, n_pre_frames = 60, max_plotting_length_frames = 500, 
#                                rev_duration_min_frames = -1, rev_duration_max_frames = np.inf ,  rev_state = 1
#                                ):

        
#     buffered_event_features = []
#     for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
#         # rsldsseq = date_to_discrete_rslds_states[exp_date]
        
#         # featureseq = exp_date_to_feature[exp_date]
#         state_starts, state_ends = get_state_start_ends(
#                                                         rev_state,
#                                                         z_w,  
#                                                         starts= None, 
#                                                         ends = None
#                                                         )
#         for rev_start_i,rev_end_i in zip( state_starts, state_ends):
#             rev_duration_frames = rev_end_i-rev_start_i
#             if rev_duration_frames>=rev_duration_min_frames and rev_duration_frames<rev_duration_max_frames:

#                 frame_start = max(rev_start_i-n_pre_frames, 0)
#                 frame_end = min(rev_start_i+max_plotting_length_frames, rev_end_i)
            
#                 n_rev_frames = frame_end-rev_start_i
#                 n_fwd_frames = rev_start_i-frame_start
#                 buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
#                 buffered_event_feature[:] = np.nan
#                 buffered_event_feature[(n_pre_frames+-1*n_fwd_frames):(n_pre_frames+n_rev_frames)] = featureseq[frame_start:frame_end]
#                 buffered_event_features.append(buffered_event_feature[:, None])

#     if len(buffered_event_features)==0: 
#         return np.array([])

#     # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
#     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
#     return buffered_event_features


# # def align_feature_to_rev_end(exp_date_to_beh, exp_date_to_feature,
# #                             n_pre_frames = 90, max_plotting_length_frames = 150, 
# #                             rev_duration_min_frames = None, rev_duration_max_frames = None,  rev_state = 1
# # ):

    
# #     buffered_event_features = []    
# #     for exp_date, featureseq in exp_date_to_feature.items():
# #         n_frames = featureseq.shape[0]
# #         # rsldsseq = date_to_discrete_rslds_states[exp_date]
# #         behseq = exp_date_to_beh[exp_date]
# #         featureseq = exp_date_to_feature[exp_date]
# #         state_starts, state_ends = get_state_start_ends(rev_state , behseq,  starts= None, ends = None)
# #         for rev_start_i,rev_end_i in zip( state_starts, state_ends):
# #             rev_duration_frames = rev_end_i-rev_start_i
# #             if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
# #                 continue
# #             if rev_duration_max_frames is not None and rev_duration_min_frames>rev_duration_max_frames:
# #                 continue
# #             frame_start = max(0, rev_end_i-n_pre_frames) # which frame to reference in freature mat as start of window to plot  
# #             frame_end = min(rev_end_i+max_plotting_length_frames, n_frames)# which frame to reference in freature mat as end of window to plot
# #             frame_window = frame_end-frame_start # full elngth to plot 
# #             max_window_size = max_plotting_length_frames+n_pre_frames
# #             if n_frames< rev_end_i+max_plotting_length_frames:
# #                 # n_pre_frames = rev_end_i+max_plotting_length_frames-n_frames
# #                 # window_start = max_window_size-frame_window #+rev_end_i+max_plotting_length_frames-n_frames
# #                 diff = rev_end_i+max_plotting_length_frames-n_frames
# #                 window_start = max_window_size-frame_window
# #                 window_start = 0
# #             else: 
# #                 window_start = max_window_size-frame_window#n_pre_frames
# #                 diff = 0
                

# #             buffered_event_feature = np.zeros(max_window_size)
# #             buffered_event_feature[:] = np.nan
# #             buffered_event_feature[window_start:n_pre_frames+max_plotting_length_frames-diff] = featureseq[frame_start:frame_end]
# #             buffered_event_features.append(buffered_event_feature[:, None])
                
            
# #             # n_rev_frames = frame_end-rev_start_i
# #             # n_fwd_frames = rev_start_i-frame_start
# #             # buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
# #             # buffered_event_feature[:] = np.nan
# #             # buffered_event_feature[n_pre_frames+-1*n_fwd_frames:n_pre_frames+n_rev_frames] = featureseq[frame_start:frame_end]
# #             # buffered_event_features.append(buffered_event_feature[:, None])
  
# #     if len(buffered_event_features)==0: 
# #         return np.array([])

# #     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
# #     return buffered_event_features

# def align_feature_to_rev_end( z, feature_val,
#                             n_pre_frames = 90, max_plotting_length_frames = 150, 
#                             rev_duration_min_frames = None, rev_duration_max_frames = None,  rev_state = 1
#     ):

    
#     buffered_event_features = []    
#     # for exp_date, featureseq in exp_date_to_feature.items():
#     for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
#         n_frames = featureseq.shape[0]
#         # rsldsseq = date_to_discrete_rslds_states[exp_date]
#         # behseq = exp_date_to_beh[exp_date]
#         # featureseq = exp_date_to_feature[exp_date]
#         state_starts, state_ends = get_state_start_ends(rev_state , z_w,  starts= None, ends = None)
#         for rev_start_i,rev_end_i in zip( state_starts, state_ends):
#             rev_duration_frames = rev_end_i-rev_start_i
#             if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
#                 continue
#             if rev_duration_max_frames is not None and rev_duration_min_frames>rev_duration_max_frames:
#                 continue
#             frame_start = max(0, rev_end_i-n_pre_frames) # which frame to reference in freature mat as start of window to plot  
#             frame_end = min(rev_end_i+max_plotting_length_frames, n_frames)# which frame to reference in freature mat as end of window to plot
#             frame_window = frame_end-frame_start # full elngth to plot 
#             max_window_size = max_plotting_length_frames+n_pre_frames
#             if n_frames< rev_end_i+max_plotting_length_frames:
#                 # n_pre_frames = rev_end_i+max_plotting_length_frames-n_frames
#                 # window_start = max_window_size-frame_window #+rev_end_i+max_plotting_length_frames-n_frames
#                 diff = rev_end_i+max_plotting_length_frames-n_frames
#                 window_start = max_window_size-frame_window
#                 window_start = 0
#             else: 
#                 window_start = max_window_size-frame_window#n_pre_frames
#                 diff = 0
                

#             buffered_event_feature = np.zeros(max_window_size)
#             buffered_event_feature[:] = np.nan
#             buffered_event_feature[window_start:n_pre_frames+max_plotting_length_frames-diff] = featureseq[frame_start:frame_end]
#             buffered_event_features.append(buffered_event_feature[:, None])
                
            
#             # n_rev_frames = frame_end-rev_start_i
#             # n_fwd_frames = rev_start_i-frame_start
#             # buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
#             # buffered_event_feature[:] = np.nan
#             # buffered_event_feature[n_pre_frames+-1*n_fwd_frames:n_pre_frames+n_rev_frames] = featureseq[frame_start:frame_end]
#             # buffered_event_features.append(buffered_event_feature[:, None])
  
#     if len(buffered_event_features)==0: 
#         return np.array([])

#     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
#     return buffered_event_features

# def align_rev_to_stim_start(z, feature_val,stim_start_frames, stim_end_frames,
#                             n_pre_frames = 60, max_plotting_length_frames = 500, 
#                                rev_duration_min_frames = None, rev_duration_max_frames = None ,  rev_state = 1
#                                ):
   
#     buffered_event_features = []
#     # for exp_date, featureseq in exp_date_to_feature.items():
#     #     # rsldsseq = date_to_discrete_rslds_states[exp_date]
#     #     behseq = exp_date_to_beh[exp_date]
#     #     featureseq = exp_date_to_feature[exp_date]
#     for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
#         state_starts, state_ends = get_state_start_ends(rev_state , z_w,  starts= None, ends = None)
#         for rev_start_i,rev_end_i in zip( state_starts, state_ends):
#             rev_duration_frames = rev_end_i-rev_start_i
#             if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
#                 continue
#             if rev_duration_max_frames is not None and rev_duration_frames>rev_duration_max_frames:
#                 continue
#             frame_start = max(stim_start_frames-n_pre_frames, 0)
#             frame_end = min(stim_start_frames+max_plotting_length_frames, rev_end_i)
        
#             n_poststim_frames = frame_end-n_pre_frames
#             n_prestim_frames = stim_start_frames-frame_start
#             buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
            
#             buffered_event_feature[:] = np.nan
#             buffered_event_feature[(n_pre_frames-n_prestim_frames):n_pre_frames+n_poststim_frames] = featureseq[frame_start:frame_end]
#             buffered_event_features.append(buffered_event_feature[:, None])

#         if len(buffered_event_features)==0: 
#             return np.array([])

#     # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
#     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
#     return buffered_event_features

# %%

from matplotlib.colors import LinearSegmentedColormap

#color palette for plotting, colors as in make_behavior_ethogram
palette = ["coral",     # forward
           "lightblue", # reverse
           "darkgreen", # turn
           "purple","red", "yellow", "black", "pink"]    # pause
cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))


# %% [markdown]
# # Load data, model, behavior decoder

# %%


T = 1599
full_traces, full_neural_labels, full_beh_classification, full_beh_data = load_all_data()

velocity_og = np.array([full_beh_data[i]["velocity"][0:1599] for i in range(len(full_beh_data))])
acceleration_og = np.array([full_beh_data[i]["acceleration"][1:1600] for i in range(len(full_beh_data))])
head_curvature_og = np.array([full_beh_data[i]["head_angle"][0:1599] for i in range(len(full_beh_data))])
worm_curvature_og = np.array([full_beh_data[i]["worm_curvature"][0:1599] for i in range(len(full_beh_data))])
pumping_og = np.array([full_beh_data[i]["pumping"][0:1599] for i in range(len(full_beh_data))])
body_angle_mag_og = np.array([full_beh_data[i]["body_angle_mag"][0:1599] for i in range(len(full_beh_data))])
midlines_og =  np.array([full_beh_data[i]["midlines"][0:1599] for i in range(len(full_beh_data))])
body_angles_og =  np.array([full_beh_data[i]["body_angles"][0:1599] for i in range(len(full_beh_data))])

velocity = copy.deepcopy(velocity_og)
acceleration = copy.deepcopy(acceleration_og)
head_curvature = copy.deepcopy(head_curvature_og)
worm_curvature = copy.deepcopy(worm_curvature_og)
pumping = copy.deepcopy(pumping_og)
body_angle_mag =copy.deepcopy(body_angle_mag_og)
midlines =copy.deepcopy( midlines_og)
body_angles = copy.deepcopy(body_angles_og)


np.random.seed(0)

foldername = "/Users/friederikebuck/Downloads/worm notes/processed_checkpts/test_rslds_QM_run_061325/model_selection/top_2_combined/"
items = np.array(os.listdir(foldername))
folders = np.array(["." not in directory for directory in items], dtype=bool)
folders = items[folders]

folder = folders[0]

with open(foldername+"/Y.npy", 'rb') as handle:  
    Y = pickle.load(handle)


with open(foldername+folder+"/saved_data/prior.npy", 'rb') as handle:
    global_worm = pickle.load(handle)
    global_worm.D = int(global_worm.D)

with open(foldername+folder+"/saved_data/prior_q.npy", 'rb') as handle:
    q = pickle.load(handle)
    

slds = global_worm

neural_labels = np.load(foldername+"/neurons.npy", allow_pickle=True)

# set up a single behavioral state array ("z" in the language of scott): 
z = [beh_classification["is_fwd"][0:1599]+ 2*beh_classification["is_rev"][0:1599]+ 4*beh_classification["is_pause"][0:1599]+3*beh_classification["is_turn"][0:1599]-1 for beh_classification in full_beh_classification]

# get estimated latents, states
q_x = q.mean_continuous_states
q_z = [slds.most_likely_states(q_x[w], Y[w]) for w in range(len(q_x))]

# traces, neural_labels, behavior_classification, mask = load_all_data_but_pretend_its_all_one_worm()

# velocity = np.array([full_beh_data[i]["velocity"][0:1599] for i in range(len(full_beh_data))])
# # velocity_all = 
# acceleration = np.array([full_beh_data[i]["acceleration"][1:1600] for i in range(len(full_beh_data))])
# head_curvature = np.array([full_beh_data[i]["head_angle"][0:1599] for i in range(len(full_beh_data))])
# worm_curvature = np.array([full_beh_data[i]["worm_curvature"][0:1599] for i in range(len(full_beh_data))])
# pumping = np.array([full_beh_data[i]["pumping"][0:1599] for i in range(len(full_beh_data))])

q_z_all = np.concatenate(q_z)
z_all = np.concatenate(z)

# Input rSLDS
input_list = ["AVA", "RIB", "RIM", "AIB"]
new_slds, input_dict = input_slds(slds,neural_labels, input_list, input_list)
new_slds.D = int(new_slds.D)
input_dict

beh_decoder = make_behavior_decoder(q_x, z)



# %%

from rslds_visualization_utils import plot_states_and_neurons
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from compute_donut_utils import fix_signs, compute_phase,compute_radius, extract_rotation_angle, set_seed

import sys 

sys.path.append("/Users/friederikebuck/")
sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')
sys.path.append("/Users/friederikebuck/Downloads/worm notes/worm_code/fit_hierarchical/")

emissions_dim = max(traces.shape[1] for traces in full_traces)
model = slds

exp_dates = get_exp_dates() 




y = np.array(q_x)
pca = PCA(n_components=9, svd_solver="full")
pca.fit(np.concatenate(q_x[:-2])) #leave out two worms to test generalization
pca.components_ = fix_signs(pca.components_)
pca_x = pca.transform(np.concatenate(q_x))

#instead of training on phase, we'll train on sin and cosine to avoid the jumping between 2pi and 0
thetas = np.array([compute_phase(q_x[i], pca) for i in range(21)])
y_sin = [np.sin(y_i) for y_i in thetas]
y_cos = [np.cos(y_i) for y_i in thetas]

y = np.zeros((21, 1599, 3))
y[:,:,0] = y_sin
y[:,:,1] = y_cos
y[:,:,2] = [compute_radius(q_x[i], pca) for i in range(21)]

rotated_data = extract_rotation_angle(pca_x, n_neighbors=5)

rotated_data2d = rotated_data[:, 1:3]  # Project onto the principal plane
angles = np.arctan2(rotated_data2d[:, 1], rotated_data2d[:, 0])  

n_worms, timepoints = [21, 1599]
# angles = np.radians((270-np.degrees(angles+ np.pi))%360-180)

angles = angles.reshape(n_worms, timepoints)
rotated_data2d = rotated_data2d.reshape(n_worms, timepoints, 2)

y_sin = np.sin(angles)
y_cos = np.cos(angles)
radii = np.sqrt(rotated_data2d[..., 0]**2 + rotated_data2d[..., 1]**2)

y = np.stack([y_sin, y_cos, radii], axis=-1)
y.shape
z_norm = np.array(z, np.float64)



# %%
len(z)

# %% [markdown]
# ### This block is the main new thing compared to angles.ipynb

# %%
velocity -= velocity.mean()
velocity /= velocity.std()
acceleration -= acceleration.mean()
acceleration /= acceleration.std()
worm_curvature -= worm_curvature.mean()
worm_curvature /= worm_curvature.std()
z_norm -= z_norm.mean()
z_norm /= z_norm.std()
z_flat = np.array(z).flatten()
from angle.preprocess import prepare_data, split_and_normalize
from angle.train import train_model
from angle.plot_funcs_1 import plot_prediction, plot_held_out_worm

set_seed(42)
normalize=False
T = 5
X_all, y_all = prepare_data([velocity, acceleration, worm_curvature, z_norm], y, T=T, holdout_idxs=[20])
X_train, X_test, y_train, y_test, input_stats = split_and_normalize(X_all, y_all, normalize=normalize) # test here is held-out timepoints of held-in worms

input_dim = X_train.shape[2]
output_dim = y_train.shape[1]

# %%
####get stim algined and rev alogined plots
from spont_rev_prediction_fncs import get_state_start_ends, align_feature_to_rev_start, plot_mean_line_graph_with_individ_traces, align_feature_to_rev_end
'''get reversal features'''

# feature_to_dict = {
#     "body angle mag":(exp_date_to_body_angles,  50, lambda x: np.mean(x) ), 
#     "velocity": (exp_date_to_velocity,   50,  lambda x: np.max(np.abs(x))), 
#     "head angle": (exp_date_to_head_angle,  50,  lambda x: np.mean(x)),#np.arange(0, 6, 0.01)), , 
#     "acceleration": (exp_date_to_acceleration, 50, lambda x: np.max(np.abs(x)))
    
    
# }
feature_list = [velocity_og, acceleration_og, worm_curvature_og]
feature_labels = ["flav_vel", "flav_accel", "flav_curv"]

dt = 0.6
'''get rev start aligned '''
n_pre_frames = int(6/dt)
max_plotting_length_frames =  320


rev_duration_mins = np.arange(0, 20/dt)
print(rev_duration_mins)
# rev_diff = 0.5/dt
rev_duration_maxs = rev_duration_mins+1

all_durations = []
for i, z_w in enumerate(z):
    # rsldsseq = date_to_discrete_rslds_states[exp_date]
    
    # featureseq = exp_date_to_feature[exp_date]
    state_starts, state_ends = get_state_start_ends(
                                                    1,
                                                    z_w,  
                                                    starts= None, 
                                                    ends = None
                                                    )
    for rev_start_i,rev_end_i in zip( state_starts, state_ends):
        rev_duration_frames = rev_end_i-rev_start_i
        all_durations.append(rev_duration_frames)
        
fig, ax = plt.subplots()
ax.hist(all_durations)
        
        
for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
    all_fig, all_axs = plt.subplots(len(feature_list), 1 )
    for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
        # exp_date_to_feature, bins, feature_fnc  = feature_to_dict[feature]
        # for rslds_start_state, rslds_end_state in rslds_pairs: 
            # '''get number of reversals that: 
            #     - start in rslds state and end in other rslds state 
            #     - start in rslds state and stay in rslds state 
            #     - start in differnt state and end in rslds state 
            # ...based on this decide which version to plot.. 
            # '''
            # print(feature)
            # # fig, ax = plt.subplots()
            # feature_vals = []
            # beh_states = []
        # buffered_event_features = []
        
        rev_aligned_feature = align_feature_to_rev_start(z, feature, n_pre_frames = n_pre_frames, 
                                                         max_plotting_length_frames =  max_plotting_length_frames, 
                                                         rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
        fig, ax1 = plt.subplots()
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None,# vline = n_pre_frames*dt, 
                                                title = f"{np.round(rev_duration_min*dt,2)}", 
                                                xlabel ="seconds post-rev start", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0, 
                                                
                                                )
        
        
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax =all_axs[i] , seconds = None,# vline = n_pre_frames*dt, 
                                               title = f"{np.round(rev_duration_min*dt,2)}", 
                                                xlabel ="seconds post-rev start", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0
                                                
                                                )
        if i>0:
            all_axs[i].set_title("")
   


                                    
################plto aligned to rev end

# features = ["acceleration", "velocity", "body angle mag",  "head angle"]
max_plotting_length_frames =  int((20+6)/dt)
n_pre_frames  = int(20/dt)
for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
    all_fig, axs = plt.subplots(len(feature_list), 1 )
    for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
        # exp_date_to_feature, bins, feature_fnc  = feature_to_dict[feature]
        # n_pre_frames  = int(rev_duration_max/dt)+2
        buffered_event_features = []
        rev_aligned_feature = align_feature_to_rev_end(z, feature,
                                n_pre_frames = n_pre_frames, max_plotting_length_frames = max_plotting_length_frames, 
                                rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
        
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax = axs[i], seconds = None,title = "", 
                                                xlabel ="seconds post-rev end", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0, vline = n_pre_frames*dt)
        
        fig, ax1 = plt.subplots()
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, vline = n_pre_frames*dt, 
                                                    title = f"{np.round(rev_duration_min*dt,2)}", 
                                                    xlabel ="seconds post-rev end", 
                                                    ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                    max_n_tracks = 30,line_width=1.0, 
                                                    
                                                    )





                                             

##TODO plot aligned to stim end                            

# %% [markdown]
# 

# %%
feature_list = [velocity, acceleration, worm_curvature]
feature_labels = ["flav_vel", "flav_accel", "flav_curv"]
rev_bin = np.array([z_w ==1 for z_w in z])

all_timesteps = range(1, 21, 5) #range(1, 15, 3)
win_start = 1
# win_end = 5
for win_end in range(5, 16, 3): 
    n_splits = 10
    test_size = 0.3
    fps = 1.6

    score_label = "balanced accuracy"
    model_fit_fnc = fit_eval_linearSVC
    prep_data_fnc = prep_data_spont_rev_prob #vs prep_data_spont_rev_latency
    title_label = "FLAVELL spont revs"
    result_visualization_fnc = visualize_model_classification


    best_models = predict_spont_revs_from_features(   feature_list, 
                                        feature_labels, rev_bin,
                                        fps, 
                                        win_start, win_end,
                                        model_fit_fnc, 
                                        score_label,
                                        result_visualization_fnc, 
                                        prep_data_fnc, 
                                        
                                        title_label = title_label, 
                                        all_timesteps = all_timesteps, 
                                        test_size = test_size, 
                                        n_splits = n_splits,  
                                        # n_train = n_train,
                                        # n_test = n_test, 

                                        single_feature = False, 
                                        n_pre_start = 0 ) 




# %%
feature_list = [velocity, acceleration, worm_curvature]
feature_labels = ["flav_vel", "flav_accel", "flav_curv"]
rev_bin = np.array([z_w ==1 for z_w in z])

all_timesteps = range(1, 21, 5) #range(1, 15, 3)
win_start = 1
win_end = 5
n_splits = 10
test_size = 0.3
fps = 1.6

score_label = "balanced accuracy"
model_fit_fnc = fit_eval_linearSVC
prep_data_fnc = prep_data_spont_rev_prob #vs prep_data_spont_rev_latency
title_label = "FLAVELL spont revs"
result_visualization_fnc = visualize_model_classification


feature_to_best_models = predict_spont_revs_from_features_single_feature(   feature_list, 
                                     feature_labels, rev_bin,
                                     fps, 
                                     win_start, win_end,
                                     model_fit_fnc, 
                                     score_label,
                                     result_visualization_fnc, 
                                     prep_data_fnc, 
                                     
                                     title_label = title_label, 
                                     all_timesteps = all_timesteps, 
                                    test_size = test_size, 
                                    n_splits = n_splits,  
                                    # n_train = n_train,
                                    # n_test = n_test, 

                                    single_feature = False, 
                                    n_pre_start = 0 ) 

# %%

# def reformat_data_1(exp_vel, exp_acc, exp_curve, exp_behaviors, turns, pauses, flavell_beh, flavell_velocity, flavell_acceleration, flavell_worm_curvature, palette =None, plot=True, sep_pos_neg_resample = True, match_vel_only = True):
#     if palette is None:
#         palette = ["coral",     # forward
#             "lightblue", # reverse
#             "darkgreen", # turn
            
#             "purple","red", "yellow", "black", "pink"]  
#     print("in reformar data")
        
#     if plot:
#         fig,axs = plt.subplots(4, 3, figsize=(18,8))
#         axs[0,0].set_ylabel("Steve's data")
#         axs[1,0].set_ylabel("FB data (raw)")
#         axs[2,0].set_ylabel("FB (interpolated)")
#         axs[3,0].set_ylabel("FB (scaled n centered)")

#         bins = [np.linspace(-6, 6, 1000), np.linspace(-5,5, 1000), np.linspace(-5, 5, 1000)]

#         # variables = ["velocity", "acceleration", "total curvature"]
#         print("flavell_velocity.shahpe", flavell_velocity.shape)
#         # # for i in range(3):
#         # #     quantity= X_all[:,0,i]
#         # #     axs[0,i].hist(quantity, bins=bins[i])
#         # #     axs[0,i].set_title(variables[i])
#         #     # axs[0,0].hist(flavell_velocity, bins=bins[i])
#         # step = 10
#         # axs[0,0].hist(flavell_velocity[::step], bins=200)
#         # axs[0,0].set_title("velocity")
        
#         # # axs[0,1].hist(flavell_acceleration, bins=bins[i])
#         # axs[0,1].hist(flavell_acceleration[::step], bins=200)
#         # axs[0,1].set_title("accel")
        
#         # # axs[0,2].hist(flavell_worm_curvature, bins=bins[i])
#         # axs[0,2].hist(flavell_worm_curvature[::step], bins=200)
#         # axs[0,2].set_title("curvature")


#     all_vel = exp_vel.flatten()
#     all_acc = exp_acc.flatten()
#     all_curve = exp_curve.flatten()
#     all_rev = exp_behaviors.flatten()
#     all_pause = pauses.flatten()
#     all_turn = turns.flatten()

#     if plot:
#         axs[1,0].hist(all_vel[all_pause == 1], bins=1000, density=True, color="purple", label="pause")
#         axs[1,0].hist(all_vel[all_rev == 1], bins=1000, density=True, color=palette[1], label="rev")
#         axs[1,0].hist(all_vel[(all_rev == 0)*(all_pause==0)], bins=1000, density=True, color=palette[0], label="fwd + turn")
#         axs[1,0].hist(all_vel, bins=100, density=True, color='k', label="all")
#         axs[1,0].legend()


#         axs[1,1].hist(all_acc, bins=1000, density=True)
#         axs[1,2].hist(all_curve, bins=1000, density=True)
#     print("plotted")

#     print("exp_acc1", all_acc.shape)
#     if sep_pos_neg_resample: 
#         print("resamplsep_pos_neg_resampleing")
#         exp_acc1 = replace_outliers_neg_post_seperate_and_resample(all_acc,  threshold = 5)/1000
#         print("exp_acc1.shape)", exp_acc1.shape)
#         resampled_acc = exp_acc1
        
#         exp_curve1 = replace_outliers_neg_post_seperate_and_resample(all_curve)
#         resampled_curve = exp_curve1
        
#         exp_vel1 = replace_outliers_neg_post_seperate_and_resample(all_vel, threshold = 2)/1000
#         resampled_vel = exp_vel1
        
#         print()
#     else: 
#         resampled_vel = replace_outliers_and_resample(all_vel)/1000
#         resampled_acc = replace_outliers_and_resample(all_acc)/1000
#         resampled_curve = replace_outliers_and_resample(all_curve)
#     resampled_features = [copy.deepcopy(resampled_vel).reshape((exp_vel.shape[0],-1)), copy.deepcopy(resampled_acc).reshape((exp_vel.shape[0],-1)), copy.deepcopy(resampled_curve).reshape((exp_vel.shape[0],-1))]
#     resampled_rev = resample_binary(all_rev)#.reshape((exp_vel.shape[0],-1))
#     resampled_pause = resample_binary(all_pause)#.reshape((exp_vel.shape[0],-1))
#     print("resampled_pause", resampled_pause.shape)
#     resampled_turn = resample_binary(all_turn)#.reshape((exp_vel.shape[0],-1))
    
#     if plot:
#         palette = ["coral",     # forward
#             "lightblue", # reverse
#             "darkgreen", # turn
#         #    "purple","red", "yellow", "black", "pink"
#             ]    # pause
#         axs[2,0].hist(resampled_vel[resampled_pause == 1], bins=100, density=True, color="purple", label="pause")
#         axs[2,0].hist(resampled_vel[resampled_rev == 1], bins=100, density=True, color=palette[1], label="rev")
#         axs[2,0].hist(resampled_vel[(resampled_rev == 0)*(resampled_pause==0)], bins=100, density=True, color=palette[0], label="fwd + turn")
#         axs[2,0].hist(resampled_vel, bins=100, density=True, color='k', label="all")
#         axs[2,0].legend()

#         axs[2,1].hist(resampled_acc, bins=1000)
#         axs[2,1].set_xlim(axs[0,1].get_xlim())

#         axs[2,2].hist(resampled_curve, bins=1000)
#         axs[2,2].set_xlim(axs[0,2].get_xlim())
#     print("resampling")


#     # smooth_vel = smooth_trace(resampled_vel, 0.5).flatten() 
#     # smooth_acc = smooth_trace(resampled_acc, 1.8).flatten()
#     # smooth_curve = smooth_trace(resampled_curve, 1).flatten()
#     smooth_vel = causal_moving_average(resampled_vel, 3)
#     smooth_acc = causal_moving_average(resampled_acc, 3)
#     smooth_curve = causal_moving_average(resampled_curve, 3)
    
#     smooth_features = [copy.deepcopy(smooth_vel).reshape((exp_vel.shape[0],-1)), copy.deepcopy(smooth_acc).reshape((exp_vel.shape[0],-1)), copy.deepcopy(smooth_curve).reshape((exp_vel.shape[0],-1))]

    


#     print("matching cdfs")
#     matched_vel = copy.deepcopy(smooth_vel)
#     matched_acc = copy.deepcopy(smooth_acc)
#     matched_curve = copy.deepcopy(smooth_curve)
#     if match_vel_only:

#         criteria = (resampled_rev == 0) * (resampled_turn ==0)
#         matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh== 0])

#         criteria = (resampled_rev == 1) 
#         matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 1])

#         criteria = (resampled_turn ==1)
#         matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 2])

        
#         matched_acc = match_cdf(smooth_acc, flavell_acceleration.flatten())

#         matched_curve = match_cdf(smooth_curve, flavell_worm_curvature.flatten())
#     else: 
#         criteria = (resampled_rev == 0) * (resampled_turn ==0)
#         matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_acceleration.flatten()[flavell_beh== 0])
#         matched_acc[criteria] = match_cdf(smooth_acc[criteria] , flavell_acceleration.flatten()[flavell_beh== 0])
#         matched_curve[criteria] = match_cdf(smooth_curve[criteria] , flavell_acceleration.flatten()[flavell_beh== 0])
        
        
#         criteria = (resampled_rev == 1) 
#         matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 1])
#         matched_acc[criteria] = match_cdf(smooth_acc[criteria] , flavell_acceleration.flatten()[flavell_beh== 1])
#         matched_curve[criteria] = match_cdf(smooth_curve[criteria] , flavell_acceleration.flatten()[flavell_beh== 1])

#         criteria = (resampled_turn ==1)
#         matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 2])
#         matched_acc[criteria] = match_cdf(smooth_acc[criteria] , flavell_acceleration.flatten()[flavell_beh== 2])
#         matched_curve[criteria] = match_cdf(smooth_curve[criteria] , flavell_acceleration.flatten()[flavell_beh== 2])
#     matched_features = [copy.deepcopy(matched_vel).reshape((exp_vel.shape[0],-1)), copy.deepcopy(matched_acc).reshape((exp_vel.shape[0],-1)), copy.deepcopy(matched_curve).reshape((exp_vel.shape[0],-1))]


#     if plot:
#         axs[3,0].hist(matched_vel[resampled_rev == 1], bins=bins[0], density=True, color=palette[1], label="FB rev only")
#         axs[3,0].hist(matched_vel[(resampled_rev == 0)*(resampled_pause==0)], bins=bins[0], density=True, color=palette[0], label="FB fwd + turn only")
#         # axs[3,0].hist(resampled_vel[(resampled_pause==1)], bins=100, density=True, color="purple", label="FB pause only")

#         axs[3,0].hist(matched_vel, bins=bins[0], density=True, label="FB all")


#         axs[3,1].hist(matched_acc, bins=bins[1], density=True, label="FB")
#         axs[3,1].set_xlim(axs[0,1].get_xlim())


#         axs[3,2].hist(matched_curve, bins=bins[2], density=True, label="FB")
#         axs[3,2].set_xlim(axs[0,2].get_xlim())

#         for i in range(3):
#             quantity= X_all[:,0,i]
#             axs[3,i].hist(quantity, bins=bins[i], alpha=0.3, density=True, label="steve")
#             axs[3,i].legend()

#         fig.tight_layout()



#     matched_vel = matched_vel.reshape((exp_vel.shape[0],-1))
#     matched_acc = matched_acc.reshape((exp_vel.shape[0],-1))
#     matched_curve = matched_curve.reshape((exp_vel.shape[0],-1))
#     resampled_rev = resampled_rev.reshape((exp_vel.shape[0],-1))
#     resampled_turn = resampled_turn.reshape((exp_vel.shape[0],-1))

#     # resampled_vel = smooth_trace(resampled_vel, 0.5)
#     # resampled_acc = smooth_trace(resampled_acc, 1.8)
#     # resampled_curve = smooth_trace(resampled_curve, 1)

    

#     return  resampled_rev, resampled_turn,resampled_features,  smooth_features, matched_features


# %%
# from load_and_process_FB_data_fncs_8 import load_data, reformat_data,plot_data,  analyze_data, replace_outliers_exp, reformat_data_1
# ####get stim algined and rev alogined plots



# import pickle 
# pickle_fname = "/Users/friederikebuck/Downloads/worm notes/processed_checkpts/worm_angles_concat_neural.pickle"
# with open(pickle_fname, "rb") as handle:
#     worm_angles_concat_neural, worm_radii_concat_neural, model_neural_directed = pickle.load(handle) 
    



# pickle_fname = "/Users/friederikebuck/Downloads/worm notes/processed_checkpts/worm_angles_concat_beh.pickle"
# with open(pickle_fname, "rb") as handle:
#    worm_angles, worm_radii, model_beh_directed =  pickle.load( handle) 
    
    

# # experiment = "duty_cycle"
# # condition = "exp"
# # neuron = "AVA"
# # path = "/Users/friederikebuck/beh_analysis_results/csv/_Aim1_AVA_duty_cycle/"


# # experiment = "duty_cycle"
# # condition = "exp"
# # neuron = "AIB"
# # path = "/Users/friederikebuck/beh_analysis_results/csv/_AIBC-3_Aim1_AIB_duty_cycle/"


# experiment = "duty_cycle"
# # condition = "exp"
# condition = "control"
# neuron = "RIM"
# path = "/Users/friederikebuck/beh_analysis_results/csv/_Aim1_RIM_duty_cycle/"


# exp_data = load_data(experiment, condition, neuron, path = path)


# spont_exp_vel, spont_exp_acc, spont_exp_curve, spont_exp_rev, exp_pause, exp_turn, exp_body_angle_mag  = replace_outliers_exp(exp_data, exp_data[0].shape)

# exp_onsets = plot_data(spont_exp_vel, spont_exp_acc, spont_exp_curve, spont_exp_rev, exp_turn, experiment, fps=6)

# resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn = reformat_data(
#                                                                                              spont_exp_vel, 
#                                                                                              spont_exp_acc,
#                                                                                              spont_exp_curve,
#                                                                                              spont_exp_rev,
#                                                                                              exp_turn, 
#                                                                                              exp_pause,
#                                                                                              z_flat,
#                                                                                              velocity, 
#                                                                                              acceleration, 
#                                                                                              worm_curvature,
#                                                                                             #  X_all,  
#                                                                                              plot=False, 
                                                                                             
#                                                                                              sep_pos_neg_resample = True)

# onsets = plot_data(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, experiment, fps=1.6, orig = False)
# resampled_onsets = onsets

# ###using body angle mag;
# pickle_fname = f"/Users/friederikebuck/Downloads/worm notes/processed_checkpts/aim1_FB_beh_{neuron}_{condition}_{experiment}_phase_features_exp.pickle"
# with open(pickle_fname, "rb") as handle:
#     resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn,behaviors_all,  inferred_phases_all, inferred_rad_all = pickle.load(handle) 
    

# %%
from load_and_process_FB_data_fncs_8 import  causal_moving_average,plot_data,   reformat_data_1, replace_outliers_exp, replace_outliers_neg_post_seperate_and_resample, smooth_trace,replace_outliers_and_resample,resample_binary,   match_cdf 

from visualize_donut_time_series_and_states import plot_donut_beh_figs_1
import pickle 
pickle_fname = "/Users/friederikebuck/Downloads/worm notes/processed_checkpts/worm_angles_concat_neural.pickle"
with open(pickle_fname, "rb") as handle:
    worm_angles_concat_neural, worm_radii_concat_neural, model_neural_directed = pickle.load(handle) 
    



pickle_fname = "/Users/friederikebuck/Downloads/worm notes/processed_checkpts/worm_angles_concat_beh.pickle"
with open(pickle_fname, "rb") as handle:
   worm_angles, worm_radii, model_beh_directed =  pickle.load( handle) 
    
    
from load_and_process_FB_data_fncs_8 import load_data, reformat_data,plot_data,  analyze_data, replace_outliers_exp, reformat_data_1
# experiment = "duty_cycle"
# condition = "control"
# neuron = "AVA"
# path = "/Users/friederikebuck/beh_analysis_results/csv/_Aim1_AVA_duty_cycle/"


# experiment = "duty_cycle"
# condition = "control"
# neuron = "AIB"
# path = "/Users/friederikebuck/beh_analysis_results/csv/_AIBC-3_Aim1_AIB_duty_cycle/"


experiment = "duty_cycle"
condition = "control"
neuron = "RIM"
path = "/Users/friederikebuck/beh_analysis_results/csv/_Aim1_RIM_duty_cycle/"

from load_and_process_FB_data_fncs_8 import  load_data, causal_moving_average,plot_data,   reformat_data_1, replace_outliers_exp, replace_outliers_neg_post_seperate_and_resample, smooth_trace,replace_outliers_and_resample,resample_binary,   match_cdf 


exp_data = load_data(experiment, condition, neuron, path = path)


spont_exp_vel, spont_exp_acc, spont_exp_curve, spont_exp_rev, spont_exp_pause, spont_exp_turn, stim_exp_body_angle_mag  = replace_outliers_exp(exp_data, exp_data[0].shape)

exp_onsets = plot_data(spont_exp_vel, spont_exp_acc, spont_exp_curve, spont_exp_rev, spont_exp_pause, experiment, fps=6)

spont_resampled_rev, spont_resampled_turn,spont_resampled_features,  spont_smooth_features, spont_matched_features = reformat_data_1(spont_exp_vel, 
                                                                                             spont_exp_acc,
                                                                                             spont_exp_curve,
                                                                                             spont_exp_rev,
                                                                                             spont_exp_turn, 
                                                                                             spont_exp_pause,
                                                                                             z_flat,
                                                                                             velocity, 
                                                                                             acceleration, 
                                                                                             worm_curvature,
                                                                                             plot=True, 
                                                                                             replace_outliers = True, 
                                                                                             sep_pos_neg_resample = True, match_vel_only = True
                                                                                             )

spont_resampled_vel, spont_resampled_acc, spont_resampled_curve = spont_resampled_features
smooth_vel, smooth_acc, smooth_curve = spont_smooth_features
matched_vel, matched_acc, matched_curve = spont_matched_features


onsets = plot_data(matched_vel, matched_acc, matched_curve, spont_resampled_rev, spont_resampled_turn, experiment, fps=1.6, orig = False)
resampled_onsets = onsets

inferred_phases, final_behaviors, final_behaviors_all, inferred_phases_all, inferred_rad_all, behaviors_all, inferred_rad, initial_behaviors, initial_cts_beh, X_all_LSTM = analyze_data(matched_vel,
                                                                                                                                                                                         matched_acc, 
                                                                                                                                                                                         matched_curve, 
                                                                                                                                                                                         spont_resampled_rev, 
                                                                                                                                                                                         spont_resampled_turn, 
                                                                                                                                                                                         onsets, 
                                                                                                                                                                                         model_beh_directed,  
                                                                                                                                                                                         z_norm,input_stats,  
                                                                                                                                                                                         z,   
                                                                                                                                                                                         lag=16, 
                                                                                                                                                                                         inclusion_thresh = 2, 
                                                                                                                                                                                         model_type="ff", 
                                                                                                                                                                                         remove_revs=False
                                                                                                                                                                                         
                                                                                                                                                                                         )
beh_all_org = (behaviors_all*  np.array(z).std())+ np.array(z).mean()

plot_donut_beh_figs_1(inferred_phases_all,inferred_rad_all,  beh_all_org,

                         cmap,
                            n_bins = 10, 
                         window = 5)
    

# %%

resampled_vel, resampled_acc, resampled_curve = spont_resampled_features

# from spont_rev_prediction_fncs import get_state_start_ends, align_feature_to_rev_start, plot_mean_line_graph_with_individ_traces, align_feature_to_rev_end
# '''get reversal features'''
# ####get stim algined and rev alogined plots
# from spont_rev_prediction_fncs import get_state_start_ends, align_feature_to_rev_start, plot_mean_line_graph_with_individ_traces, align_feature_to_rev_end
'''get reversal features'''
    
# }
feature_list = [resampled_vel, resampled_acc, resampled_curve]
feature_labels = ["spont resamp vel", "spont resamp acc", "spont resamp curv"]

dt = 0.6
'''get rev start aligned '''
n_pre_frames = int(6/dt)
max_plotting_length_frames =  320


rev_duration_mins = np.arange(0, 20/dt)
print(rev_duration_mins)
# rev_diff = 0.5/dt
rev_duration_maxs = rev_duration_mins+1

all_durations = []
for i, z_w in enumerate(spont_resampled_rev):
    # rsldsseq = date_to_discrete_rslds_states[exp_date]
    
    # featureseq = exp_date_to_feature[exp_date]
    state_starts, state_ends = get_state_start_ends(
                                                    1,
                                                    z_w,  
                                                    starts= None, 
                                                    ends = None
                                                    )
    for rev_start_i,rev_end_i in zip( state_starts, state_ends):
        rev_duration_frames = rev_end_i-rev_start_i
        all_durations.append(rev_duration_frames)
        
fig, ax = plt.subplots()
ax.hist(all_durations)
        
        
for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
    all_fig, all_axs = plt.subplots(len(feature_list), 1 )
    for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
        
        rev_aligned_feature = align_feature_to_rev_start(spont_resampled_rev, feature, n_pre_frames = n_pre_frames, 
                                                         max_plotting_length_frames =  max_plotting_length_frames, 
                                                         rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
        fig, ax1 = plt.subplots()
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None,# vline = n_pre_frames*dt, 
                                                title = f"{np.round(rev_duration_min*dt,2)}", 
                                                xlabel ="seconds post-rev start", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0, 
                                                
                                                )
        
        
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax =all_axs[i] , seconds = None,# vline = n_pre_frames*dt, 
                                               title = f"{np.round(rev_duration_min*dt,2)}", 
                                                xlabel ="seconds post-rev start", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0
                                                
                                                )
        if i>0:
            all_axs[i].set_title("")
   


                                    
################plto aligned to rev end

# features = ["acceleration", "velocity", "body angle mag",  "head angle"]
max_plotting_length_frames =  int((20+6)/dt)
n_pre_frames  = int(20/dt)
for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
    all_fig, axs = plt.subplots(len(feature_list), 1 )
    for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
        # exp_date_to_feature, bins, feature_fnc  = feature_to_dict[feature]
        # n_pre_frames  = int(rev_duration_max/dt)+2
        buffered_event_features = []
        rev_aligned_feature = align_feature_to_rev_end(spont_resampled_rev, feature,
                                n_pre_frames = n_pre_frames, max_plotting_length_frames = max_plotting_length_frames, 
                                rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
        
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax = axs[i], seconds = None,title = "", 
                                                xlabel ="seconds post-rev end", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0, vline = n_pre_frames*dt)
        
        fig, ax1 = plt.subplots()
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, vline = n_pre_frames*dt, 
                                                    title = f"{np.round(rev_duration_min*dt,2)}", 
                                                    xlabel ="seconds post-rev end", 
                                                    ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                    max_n_tracks = 30,line_width=1.0, 
                                                    
                                                    )



resampled_vel, resampled_acc, resampled_curve = spont_resampled_features


####get stim algined and rev alogined plots
from spont_rev_prediction_fncs import get_state_start_ends, align_feature_to_rev_start, plot_mean_line_graph_with_individ_traces, align_feature_to_rev_end
'''get reversal features'''
    
# }
feature_list = [spont_exp_vel,  spont_exp_acc, spont_exp_curve,]

feature_labels = ["spont exp vel", "spont exp acc", "spont exp curv"]

dt = 1/6
'''get rev start aligned '''
n_pre_frames = int(6/dt)
# max_plotting_length_frames =  320
max_plotting_length_frames =  int((20+6)/dt)
rev_duration_mins = np.arange(0, 20/dt)
print(rev_duration_mins)
# rev_diff = 0.5/dt
rev_duration_maxs = rev_duration_mins+1

all_durations = []
for i, z_w in enumerate(spont_exp_rev):
    state_starts, state_ends = get_state_start_ends(
                                                    1,
                                                    z_w,  
                                                    starts= None, 
                                                    ends = None
                                                    )
    for rev_start_i,rev_end_i in zip( state_starts, state_ends):
        rev_duration_frames = rev_end_i-rev_start_i
        all_durations.append(rev_duration_frames)
        
fig, ax = plt.subplots()
ax.hist(all_durations)
        
        
for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
    all_fig, all_axs = plt.subplots(len(feature_list), 1 )
    for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):=
        
        rev_aligned_feature = align_feature_to_rev_start(spont_exp_rev, feature, n_pre_frames = n_pre_frames, 
                                                         max_plotting_length_frames =  max_plotting_length_frames, 
                                                         rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
        fig, ax1 = plt.subplots()
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None,# vline = n_pre_frames*dt, 
                                                title = f"{np.round(rev_duration_min*dt,2)}", 
                                                xlabel ="seconds post-rev start", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0, 
                                                
                                                )
        
        
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax =all_axs[i] , seconds = None,# vline = n_pre_frames*dt, 
                                               title = f"{np.round(rev_duration_min*dt,2)}", 
                                                xlabel ="seconds post-rev start", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0
                                                
                                                )
        if i>0:
            all_axs[i].set_title("")
   


                                    
################plto aligned to rev end

# features = ["acceleration", "velocity", "body angle mag",  "head angle"]
max_plotting_length_frames =  int((20+6)/dt)
n_pre_frames  = int(20/dt)
for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
    all_fig, axs = plt.subplots(len(feature_list), 1 )
    for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
        # exp_date_to_feature, bins, feature_fnc  = feature_to_dict[feature]
        # n_pre_frames  = int(rev_duration_max/dt)+2
        buffered_event_features = []
        rev_aligned_feature = align_feature_to_rev_end(spont_exp_rev, feature,
                                n_pre_frames = n_pre_frames, max_plotting_length_frames = max_plotting_length_frames, 
                                rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
        
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax = axs[i], seconds = None,title = "", 
                                                xlabel ="seconds post-rev end", 
                                                ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                max_n_tracks = 30,line_width=1.0, vline = n_pre_frames*dt)
        
        fig, ax1 = plt.subplots()
        plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, vline = n_pre_frames*dt, 
                                                    title = f"{np.round(rev_duration_min*dt,2)}", 
                                                    xlabel ="seconds post-rev end", 
                                                    ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                    max_n_tracks = 30,line_width=1.0, 
                                                    
                                                    )







# %%

fig, ax = plt.subplots()
ax.hist(resampled_acc.flatten())
ax.set_title("resampled_acc")

fig, ax = plt.subplots()
ax.hist(smooth_acc.flatten())
ax.set_title("smooth_acc")

fig, ax = plt.subplots()
ax.hist(matched_acc.flatten())
ax.set_title("matched_acc")

for i, track in enumerate(resampled_curve):
    fig, ax = plt.subplots()
    ax.plot(track[~np.isnan(track)])
    ax.plot(resampled_turn[i, ~np.isnan(track)])
    ax.plot(resampled_rev[i, ~np.isnan(track)])
    if i == 7:
        break
    

# %%
resampled_rev, resampled_turn,resampled_features,  smooth_features, matched_features = reformat_data_1(stim_exp_vel, 
                                                                                             stim_exp_acc,
                                                                                             stim_exp_curve,
                                                                                             stim_exp_rev,
                                                                                             stim_exp_turn, 
                                                                                             stim_exp_pause,
                                                                                             z_flat,
                                                                                             velocity, 
                                                                                             acceleration, 
                                                                                             worm_curvature,
                                                                                            #  X_all,  
                                                                                             plot=True, 
                                                                                             replace_outliers = True, 
                                                                                             sep_pos_neg_resample = True, match_vel_only = False
                                                                                             )

resampled_vel, resampled_acc, resampled_curve = resampled_features
smooth_vel, smooth_acc, smooth_curve = smooth_features
matched_vel, matched_acc, matched_curve = matched_features

# resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn = reformat_data(exp_vel, 
#                                                                                              exp_acc,
#                                                                                              exp_curve,
#                                                                                              exp_rev,
#                                                                                              exp_turn, 
#                                                                                              exp_pause,
#                                                                                              z_flat,
#                                                                                              velocity, 
#                                                                                              acceleration, 
#                                                                                              worm_curvature,
#                                                                                             #  X_all,  
#                                                                                              plot=False, 
#                                                                                              sep_pos_neg_resample = True)

# onsets = plot_data(stim_resampled_vel, stim_resampled_acc, stim_resampled_curve, stim_resampled_rev, stim_resampled_turn, experiment, fps=np.round(1/0.6, 3), orig = False)
# onsets = plot_data(stim_resampled_vel, stim_resampled_acc, stim_resampled_curve, stim_resampled_rev, stim_resampled_turn, experiment, fps=1.6, orig = False)
onsets = plot_data(matched_vel, matched_acc, matched_curve, resampled_rev, resampled_turn, experiment, fps=1.6, orig = False)
resampled_onsets = onsets

inferred_phases, final_behaviors, final_behaviors_all, inferred_phases_all, inferred_rad_all, behaviors_all, inferred_rad, initial_behaviors, initial_cts_beh, X_all_LSTM = analyze_data(matched_vel, matched_acc, matched_curve,  resampled_rev, resampled_turn, onsets, model_beh_directed,  z_norm,input_stats,  z,   lag=16, inclusion_thresh = 2, model_type="ff", remove_revs=False)
# plot_donuts_FB_beh(z, behaviors_all, inferred_phases_all, inferred_rad_all)
beh_all_org = (behaviors_all*  np.array(z).std())+ np.array(z).mean()
from visualize_donut_time_series_and_states import plot_donut_beh_figs_1
plot_donut_beh_figs_1(inferred_phases_all,inferred_rad_all,  beh_all_org,

                         cmap,
                            n_bins = 10, 
                         window = 5)


# %%

fig, ax = plt.subplots()
ax.hist(resampled_acc.flatten())
ax.set_title("resampled_acc")

fig, ax = plt.subplots()
ax.hist(smooth_acc.flatten())
ax.set_title("smooth_acc")

fig, ax = plt.subplots()
ax.hist(matched_acc.flatten())
ax.set_title("matched_acc")

for i, track in enumerate(resampled_curve):
    fig, ax = plt.subplots()
    ax.plot(track[~np.isnan(track)])
    ax.plot(resampled_turn[i, ~np.isnan(track)])
    ax.plot(resampled_rev[i, ~np.isnan(track)])
    if i == 7:
        break
    

# %%
####get stim algined and rev alogined plots

'''get reversal features'''
def filter_start_ends_is_by_feature(track_start_end_is, feature_mat,  feature_min, feature_max):
    valid_is = []
    for i, (track, start, end) in enumerate(track_start_end_is):
        feature_val = feature_mat[track, start:end]
        if np.logical_and(feature_val>=feature_min, feature_val<feature_max ):
            valid_is.append(i)
    return track_start_end_is[valid_is, :]

def filter_start_ends_is_by_latency(track_start_end_is, stim_start, latency_min, latency_max):
    latencies = track_start_end_is[:, 1]-stim_start
    latency_bool = np.logical_and(latencies>=latency_min, latencies<=latency_max ).astype('bool')
    return track_start_end_is[latency_bool]

def get_state_track_start_ends(state, all_stateseqs):
    all_tracks, all_starts, all_ends = [], [], []
    for i, stateseq in enumerate(all_stateseqs):
        # if starts is None:
        transitions, starts, ends = get_transitions(stateseq)
        # for state1, state2 in state_pairs:
        state_start_is = np.argwhere(state==stateseq[starts[:-1]]).flatten()
        tracks = np.ones(state_start_is.shape)*i
        state_starts = starts[state_start_is]
        state_ends = ends[state_start_is] ##check 
        
        all_tracks.append(tracks)
        all_starts.append(state_starts)
        all_ends.append(state_ends)
        
        
    return np.concatenate([
                            np.concatenate(all_tracks)[:, None],
                            np.concatenate(all_starts)[:, None],
                            np.concatenate(all_ends)[:, None]
                            
                            ], axis = 1)

def filter_start_end_is_by_start_time(event_start_end_is, min_start_time, max_start_time):
    start_bool = np.logical_and(event_start_end_is[:,1]>=min_start_time, (event_start_end_is[:,1] < max_start_time))
    return event_start_end_is[start_bool]

event_start_end_is = get_state_track_start_ends(1, stim_resampled_rev).astype('int')
feature_list = [stim_resampled_rev,
                stim_resampled_vel, stim_resampled_acc, stim_resampled_curve]
feature_labels = ["resampled_rev", 
                  "resampled_vel", "resampled_accel", "resampled_curv"]
dt = 0.6
fps = 1/dt
'''get rev start aligned '''
n_pre_frames = int(6*fps)#16*5
max_plotting_length_frames =  int(25*fps) #320


rev_duration_mins = np.arange(0, int(10*fps))#[::15]#[-5:]
rev_duration_maxs = rev_duration_mins+1

################plot aligned to rev start
# for stim_i in range(5):
#     stim_onset = resampled_onsets[stim_i]
#     stim_offset = stim_onset+5*fps
#     fig, ax = plt.subplots()
#     ax.imshow(stim_resampled_rev)
#     ax.axvline(stim_onset)
#     ax.axvline(stim_offset)
#     stim_start_end_is = filter_start_end_is_by_start_time(event_start_end_is, stim_onset, stim_offset)
#     for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
        
#         all_fig, all_axs = plt.subplots(len(feature_list), 1 )
#         # for i, feature in enumerate(features): 
#         for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):

#             rev_aligned_feature = align_feature_to_rev_start_from_start_end_is(feature, stim_start_end_is, n_pre_frames = n_pre_frames, 
#                                                             max_plotting_length_frames =  max_plotting_length_frames, 
#                                                             rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
#             # ax1 = axs[i]
#             # print(rev_aligned_feature.shape)
#             # for j in range(2,8):
#             fig, ax1 = plt.subplots()
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = fps,  fig = fig, ax =ax1 , seconds = None,
#                                                     title = f"stim: {stim_i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-rev start", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0, 
                                                    
#                                                     )
            
            
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = fps,  fig = all_fig, ax =all_axs[i] , seconds = None, 
#                                                 title = f"stim: {stim_i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-rev start", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0
                                                    
#                                                     )
#             if i >0: 
#                 all_axs[i].set_title("")


                                    
################plto aligned to rev end

# # features = ["acceleration", "velocity", "body angle mag",  "head angle"]
# # max_plotting_length_frames =  20*#120
# n_pre_frames = int(10*fps)#16*5
# max_plotting_length_frames =  int(25*fps) #320

# for stim_i in range(5):
#     stim_onset = resampled_onsets[stim_i]
#     stim_offset = stim_onset+5*fps
#     stim_start_end_is = filter_start_end_is_by_start_time(event_start_end_is, stim_onset, stim_offset)
#     for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
        
#         all_fig, axs = plt.subplots(len(feature_list), 1 )

#         for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
#             # exp_date_to_feature, bins, feature_fnc  = feature_to_dict[feature]

#             buffered_event_features = []
#             rev_aligned_feature = align_feature_to_rev_end_from_start_end_is(feature, stim_start_end_is,  
#                                     n_pre_frames = n_pre_frames, max_plotting_length_frames = max_plotting_length_frames, 
#                                     rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
            
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax = axs[i], seconds = None,
#                                                     title = f"stim: {i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-rev end", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0, vline = n_pre_frames*dt)
            
#             fig, ax1 = plt.subplots()
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, 
#                                                         title = f"stim: {i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                         xlabel ="seconds post-rev start", 
#                                                         ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                         max_n_tracks = 30,line_width=1.0, 
                                                        
#                                                         )
#             if i >0: 
#                 axs[i].set_title("")

                                                
# ################seperate by latency plot aligned to stim staet 


resampled_vel, resampled_acc, resampled_curve = resampled_features
smooth_vel, smooth_acc, smooth_curve = smooth_features
matched_vel, matched_acc, matched_curve = matched_features

feature_list = [stim_resampled_rev,
                resampled_vel , resampled_acc, resampled_curve ]
feature_list = [stim_resampled_rev,
                matched_vel , matched_acc, matched_curve ]
# resampled_feature_list = [resampled_vel , resampled_acc]
# resampled_feature_lbl = ["vel", "accel"]

# resampled_feature_list = [resampled_curve]
# resampled_feature_lbl = ["curve"]


# feature_list = [stim_resampled_rev,
#                 smooth_vel , smooth_acc, smooth_curve ]



feature_list = [stim_resampled_rev,
                matched_vel , matched_acc, matched_curve ]
feature_labels = ["resampled_rev", 
                  "resampled_vel", "resampled_accel", "resampled_curv"]
acc_vmin, acc_vmax = [-4,4]
acc_vmin, acc_vmax = [-0.2,0.2]

'''get rev start aligned '''
fps = 1.6
dt = 1/fps
n_pre_frames = int(6*fps)
max_plotting_length_frames =  int(7*fps)


latency_mins = np.arange(0, int(5*fps))#[::15]#[-5:]
print(latency_mins)
latency_maxs = latency_mins+1


for stim_i, resampled_onset in enumerate(resampled_onsets): 
    

    stim_start_frames = int(resampled_onset-5)
    stim_end_frames = int(stim_start_frames+5*fps)
    
    # fig, ax = plt.subplots(figsize =(20,30))
    # rev = feature_list[0]
    # ax.imshow(rev[:50, (stim_start_frames-10):(stim_start_frames+10)])

    # ax.axvline(10, color = "red")
    # ax.axvline(10+5*fps, color = "red")
    
    # fig, ax = plt.subplots(figsize =(20,30))
    # vel = feature_list[1]
    # ax.imshow(vel[:50, (stim_start_frames-10):(stim_start_frames+10)], vmin = -4, vmax = 4, cmap = "bwr")

    # ax.axvline(10, color = "red")
    # ax.axvline(10+5*fps, color = "red")
    
    
    # fig, ax = plt.subplots(figsize =(20,30))
    # acc = feature_list[2]
    # ax.imshow(acc[:50, (stim_start_frames-10):(stim_start_frames+10)], vmin = acc_vmin, vmax = acc_vmax, cmap = "bwr")

    # ax.axvline(10, color = "red")
    # ax.axvline(10+5*fps, color = "red")
    
    
        
    # fig, ax = plt.subplots(figsize =(20,30))
    # curv = feature_list[3]
    # ax.imshow(curv[:50, (stim_start_frames-10):(stim_start_frames+10)], vmin = -4, vmax = 4, cmap = "bwr")

    # ax.axvline(10, color = "red")
    # ax.axvline(10+5*fps, color = "red")
    
    
    stim_start_end_is = filter_start_end_is_by_start_time(event_start_end_is, stim_start_frames, stim_end_frames)
    
    for latency_min ,latency_max in list(zip(latency_mins,latency_maxs))[1:4]:
        lat_start_end_is = filter_start_ends_is_by_latency(stim_start_end_is, stim_start_frames, latency_min, latency_max)
        # lat_start_end_is = stim_start_end_is
        all_fig, axs = plt.subplots(len(feature_list), 1 )
        # for i, (feature, feature_lbl) in enumerate(zip(resampled_feature_list, resampled_feature_lbl)):
        for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
            # og_smooth = smooth_trace(feature.flatten(),  0.5).reshape(stim_resampled_rev.shape[0],-1)
            # feature_lbl_og = feature_lbl+"_smooth 0.5"
            
            
            # og_smooth_2 = smooth_trace(feature.flatten(),  1.8).reshape(stim_resampled_rev.shape[0],-1)
            # feature_lbl_og2 = feature_lbl+"_smooth1.8"
            
            # cas_1 = causal_moving_average(feature.flatten(), 1).reshape(stim_resampled_rev.shape[0],-1)
            # feature_lbl_caus1 = feature_lbl+"cause_1"
            
                        
            # cas_12 = causal_moving_average(feature.flatten(), 2).reshape(stim_resampled_rev.shape[0],-1)
            # feature_lbl_caus12 = feature_lbl+"cause_2"
            
            # cas_13 = causal_moving_average(feature.flatten(), 3).reshape(stim_resampled_rev.shape[0],-1)
            # feature_lbl_caus13 = feature_lbl+"cause_3"
            
            # cas_2 = causal_moving_average(feature.flatten(), 5).reshape(stim_resampled_rev.shape[0],-1)
            # feature_lbl_caus2 = feature_lbl+"cause_5"
            

            # for f, lbl in zip([feature, og_smooth, og_smooth_2, cas_1,cas_12, cas_13, cas_2],
            #                   [feature_lbl, feature_lbl_og, feature_lbl_og2,feature_lbl_caus1, feature_lbl_caus12, feature_lbl_caus13, feature_lbl_caus2]):
            stim_aligned_feature = align_rev_to_stim_start_from_start_end_is(feature, lat_start_end_is, stim_start_frames, n_pre_frames =n_pre_frames, 
                                                            max_plotting_length_frames =  max_plotting_length_frames, 
                                                            # rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max
                                                            )
            # ax1 = axs[i]
            # print(rev_aligned_feature.shape)
            # for j in range(2,8):
            fig, ax1 = plt.subplots()
            plot_mean_line_graph_with_individ_traces(stim_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, 
                                                #    title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}; dur: {np.round(rev_duration_min*dt,2)}", 
                                                    title = f"stim: {stim_i}; latency: {np.round(latency_min*dt,2)}", # dur: {np.round(rev_duration_min*dt,2)}", 
                                                    xlabel ="seconds post-stim start", 
                                                    ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                    max_n_tracks = 30,line_width=1.0, 
                                                    
                                                    )
            if i >0: 
                axs[i].set_title("")
                
# for resampled_onset in resampled_onsets: 
    

#     stim_start_frames = int(resampled_onset-5)
#     stim_end_frames = int(stim_start_frames+5*fps)
    
#     # fig, ax = plt.subplots(figsize =(20,30))
#     # rev = feature_list[0]
#     # ax.imshow(rev[:50, (stim_start_frames-10):(stim_start_frames+10)])

#     # ax.axvline(10, color = "red")
#     # ax.axvline(10+5*fps, color = "red")
    
#     # fig, ax = plt.subplots(figsize =(20,30))
#     # vel = feature_list[1]
#     # ax.imshow(vel[:50, (stim_start_frames-10):(stim_start_frames+10)], vmin = -4, vmax = 4, cmap = "bwr")

#     # ax.axvline(10, color = "red")
#     # ax.axvline(10+5*fps, color = "red")
    
    
#     # fig, ax = plt.subplots(figsize =(20,30))
#     # acc = feature_list[2]
#     # ax.imshow(acc[:50, (stim_start_frames-10):(stim_start_frames+10)], vmin = acc_vmin, vmax = acc_vmax, cmap = "bwr")

#     # ax.axvline(10, color = "red")
#     # ax.axvline(10+5*fps, color = "red")
    
    
        
#     # fig, ax = plt.subplots(figsize =(20,30))
#     # curv = feature_list[3]
#     # ax.imshow(curv[:50, (stim_start_frames-10):(stim_start_frames+10)], vmin = -4, vmax = 4, cmap = "bwr")

#     # ax.axvline(10, color = "red")
#     # ax.axvline(10+5*fps, color = "red")
    
    
#     stim_start_end_is = filter_start_end_is_by_start_time(event_start_end_is, stim_start_frames, stim_end_frames)
    
#     for latency_min ,latency_max in list(zip(latency_mins,latency_maxs))[1:4]:
#         lat_start_end_is = filter_start_ends_is_by_latency(stim_start_end_is, stim_start_frames, latency_min, latency_max)
#         # lat_start_end_is = stim_start_end_is
#         all_fig, axs = plt.subplots(len(feature_list), 1 )
#         for i, (feature, feature_lbl) in enumerate(zip(resampled_feature_list, resampled_feature_lbl)):
#             og_smooth = smooth_trace(feature,  0.5).flatten()
#             feature_lbl_og = feature_lbl+"_smooth"
            
            
#             buffered_event_features = []
#             stim_aligned_feature = align_rev_to_stim_start_from_start_end_is(feature, lat_start_end_is, stim_start_frames, n_pre_frames =n_pre_frames, 
#                                                             max_plotting_length_frames =  max_plotting_length_frames, 
#                                                             # rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max
#                                                             )
#             # ax1 = axs[i]
#             # print(rev_aligned_feature.shape)
#             # for j in range(2,8):
#             fig, ax1 = plt.subplots()
#             plot_mean_line_graph_with_individ_traces(stim_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, 
#                                                 #    title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}", # dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-stim start", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0, 
                                                    
#                                                     )
            
            
#             plot_mean_line_graph_with_individ_traces(stim_aligned_feature.T, n_pre_frames,  fps = 1/dt,  fig = all_fig, ax =axs[i] , seconds = None, 
#                                                    #    title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}", # dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-stim start", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0
                                                    
#                                                 )
#             if i >0: 
#                 axs[i].set_title("")

# %%
resampled_onsets

# %%

event_start_end_is = get_state_track_start_ends(1, stim_exp_rev).astype('int')
feature_list = [stim_exp_vel, stim_exp_acc, stim_exp_curve]
feature_labels = ["exp_vel", "exp_accel", "exp_curv"]

fps = 6
dt = 1/fps
'''get rev start aligned '''
n_pre_frames = int(6*fps)#16*5
max_plotting_length_frames =  int(25*fps) #320


rev_duration_mins = np.arange(0, int(10*fps))#[::15]#[-5:]
rev_duration_maxs = rev_duration_mins+1

################plot aligned to rev start
# for stim_i in range(5):
#     stim_onset = resampled_onsets[stim_i]
#     stim_offset = stim_onset+5*fps
#     fig, ax = plt.subplots()
#     ax.imshow(stim_resampled_rev)
#     ax.axvline(stim_onset)
#     ax.axvline(stim_offset)
#     stim_start_end_is = filter_start_end_is_by_start_time(event_start_end_is, stim_onset, stim_offset)
#     for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
        
#         all_fig, all_axs = plt.subplots(len(feature_list), 1 )
#         # for i, feature in enumerate(features): 
#         for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):

#             rev_aligned_feature = align_feature_to_rev_start_from_start_end_is(feature, stim_start_end_is, n_pre_frames = n_pre_frames, 
#                                                             max_plotting_length_frames =  max_plotting_length_frames, 
#                                                             rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
#             # ax1 = axs[i]
#             # print(rev_aligned_feature.shape)
#             # for j in range(2,8):
#             fig, ax1 = plt.subplots()
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = fps,  fig = fig, ax =ax1 , seconds = None,
#                                                     title = f"stim: {stim_i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-rev start", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0, 
                                                    
#                                                     )
            
            
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = fps,  fig = all_fig, ax =all_axs[i] , seconds = None, 
#                                                 title = f"stim: {stim_i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-rev start", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0
                                                    
#                                                     )
#             if i >0: 
#                 all_axs[i].set_title("")


                                    
################plto aligned to rev end

# # features = ["acceleration", "velocity", "body angle mag",  "head angle"]
# # max_plotting_length_frames =  20*#120
# n_pre_frames = int(10*fps)#16*5
# max_plotting_length_frames =  int(25*fps) #320

# for stim_i in range(5):
#     stim_onset = resampled_onsets[stim_i]
#     stim_offset = stim_onset+5*fps
#     stim_start_end_is = filter_start_end_is_by_start_time(event_start_end_is, stim_onset, stim_offset)
#     for rev_duration_min ,rev_duration_max in zip(rev_duration_mins.tolist(),rev_duration_maxs.tolist()):
        
#         all_fig, axs = plt.subplots(len(feature_list), 1 )

#         for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):
#             # exp_date_to_feature, bins, feature_fnc  = feature_to_dict[feature]

#             buffered_event_features = []
#             rev_aligned_feature = align_feature_to_rev_end_from_start_end_is(feature, stim_start_end_is,  
#                                     n_pre_frames = n_pre_frames, max_plotting_length_frames = max_plotting_length_frames, 
#                                     rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max)
            
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = all_fig, ax = axs[i], seconds = None,
#                                                     title = f"stim: {i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                     xlabel ="seconds post-rev end", 
#                                                     ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                     max_n_tracks = 30,line_width=1.0, vline = n_pre_frames*dt)
            
#             fig, ax1 = plt.subplots()
#             plot_mean_line_graph_with_individ_traces(rev_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, 
#                                                         title = f"stim: {i}; dur: {np.round(rev_duration_min*dt,2)}", 
#                                                         xlabel ="seconds post-rev start", 
#                                                         ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
#                                                         max_n_tracks = 30,line_width=1.0, 
                                                        
#                                                         )
#             if i >0: 
#                 axs[i].set_title("")

                                                
# ################seperate by latency plot aligned to stim staet 

'''get rev start aligned '''
n_pre_frames = int(6*fps)
max_plotting_length_frames =  int(7*fps)
latency_mins = np.arange(0, int(5*fps)+1, 3)#[::15]#[-5:]
print(latency_mins)

latency_maxs = latency_mins[1:]
latency_mins = latency_mins[:-1]

for onset in  exp_onsets: 
    stim_start_frames = onset
    stim_end_frames = onset+5*fps
    stim_start_end_is = filter_start_end_is_by_start_time(event_start_end_is, stim_start_frames, stim_end_frames)
    
    for latency_min ,latency_max in zip(latency_mins,latency_maxs):
        lat_start_end_is = filter_start_ends_is_by_latency(stim_start_end_is, stim_start_frames, latency_min, latency_max)
        # lat_start_end_is = stim_start_end_is
        all_fig, all_axs = plt.subplots(len(feature_list), 1 )
        for i, (feature, feature_lbl) in enumerate(zip(feature_list, feature_labels)):

            buffered_event_features = []
            stim_aligned_feature = align_rev_to_stim_start_from_start_end_is(feature, lat_start_end_is, stim_start_frames, n_pre_frames =n_pre_frames, 
                                                            max_plotting_length_frames =  max_plotting_length_frames, 
                                                            # rev_duration_min_frames = rev_duration_min, rev_duration_max_frames = rev_duration_max
                                                            )
            # ax1 = axs[i]
            # print(rev_aligned_feature.shape)
            # for j in range(2,8):
            fig, ax1 = plt.subplots()
            plot_mean_line_graph_with_individ_traces(stim_aligned_feature.T, n_pre_frames, fps = 1/dt,  fig = fig, ax =ax1 , seconds = None, 
                                                #    title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}; dur: {np.round(rev_duration_min*dt,2)}", 
                                                    title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}", # dur: {np.round(rev_duration_min*dt,2)}", 
                                                    xlabel ="seconds post-stim start", 
                                                    ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                    max_n_tracks = 30,line_width=1.0, 
                                                    
                                                    )
            
            
            plot_mean_line_graph_with_individ_traces(stim_aligned_feature.T, n_pre_frames,  fps = 1/dt,  fig = all_fig, ax =all_axs[i] , seconds = None, 
                                                   #    title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}; dur: {np.round(rev_duration_min*dt,2)}", 
                                                    title = f"stim: {i}; latency: {np.round(latency_min*dt,2)}", # dur: {np.round(rev_duration_min*dt,2)}", 
                                                    xlabel ="seconds post-stim start", 
                                                    ylabel =f"{feature_lbl}", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                                    max_n_tracks = 30,line_width=1.0
                                                    
                                                )
            if i >0: 
                axs[i].set_title("")

# %%


print(exp_onsets)
print(resampled_onsets)




# %%


# %%
rev_duration_mins = np.arange(0, 20/dt)[::15]#[-5:]
print(rev_duration_mins)
rev_diff = 0.5/dt
rev_duration_maxs = rev_duration_mins+rev_diff

print(rev_duration_maxs)
print(rev_duration_mins)

# %%

    
    

# %%

# pickle_fname = "/Users/friederikebuck/Downloads/worm notes/processed_checkpts/worm_angles_concat_beh.pickle"
# with open(pickle_fname, "rb") as handle:
#    worm_angles, worm_radii, model_beh_directed =  pickle.load( handle) 



# experiment = "duty_cycle"
# # condition = "exp"
# condition = "control"
# neuron = "RIM"
# path = "/Users/friederikebuck/beh_analysis_results/csv/_Aim1_RIM_duty_cycle/"
# exp_data = load_data(experiment, condition, neuron, path = path)


# exp_vel, exp_acc, exp_curve, exp_rev, exp_pause, exp_turn, exp_body_angle_mag  = replace_outliers_exp(exp_data, exp_data[0].shape)

# exp_onsets = plot_data(exp_vel, exp_acc, exp_curve, exp_rev, exp_turn, experiment, fps=6)
# # plot_data(exp_vel, exp_acc, exp_curve, exp_rev, exp_turn, experiment, fps=6)
# # resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn = reformat_data(exp_vel, exp_acc, exp_curve, exp_behaviors, turns, pauses, plot=True, sep_pos_neg_resample = False)
# resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn = reformat_data(exp_vel, 
#                                                                                              exp_acc,
#                                                                                              exp_curve,
#                                                                                              exp_rev,
#                                                                                              exp_turn, 
#                                                                                              exp_pause,
#                                                                                              z_flat,
#                                                                                              velocity, 
#                                                                                              acceleration, 
#                                                                                              worm_curvature,
#                                                                                             #  X_all,  
#                                                                                              plot=False, 
#                                                                                              sep_pos_neg_resample = True)


# onsets = plot_data(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, experiment, fps=1.6, orig = False)

# times = onsets
# inferred_phases, final_behaviors, final_behaviors_all, inferred_phases_all, inferred_rad_all, behaviors_all, inferred_rad, initial_behaviors, initial_cts_beh, X_all_LSTM = analyze_data(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, times, model_beh_directed, z_norm,input_stats,  z, lag=16, inclusion_thresh = 2, model_type="ff",normalize=False, remove_revs=False)


# fig,ax = plt.subplots(figsize=(4,4))
# donut_x = inferred_rad_all*np.cos(inferred_phases_all)
# donut_y = inferred_rad_all*np.sin(inferred_phases_all)

# ax.scatter(donut_x[::10], donut_y[::10], c=behaviors_all[::10], cmap=cmap, vmin=-1, vmax=7, alpha = 0.1)


# pickle_fname = f"/Users/friederikebuck/Downloads/worm notes/processed_checkpts/aim1_FB_beh_{neuron}_{condition}_{experiment}_phase_features_exp.pickle"
# with open(pickle_fname, "wb") as handle:
#     pickle.dump([resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn,behaviors_all,  inferred_phases_all, inferred_rad_all,], handle) 

# %%


# %%


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree

import numpy as np
from sklearn.model_selection import train_test_split

import warnings
# from predict_latency_prob_from_features_donut_6 import prep_FB_inputs, prep_FB_inputs_donut_only, 

# # # from predict_latency_prob_from_features_donut_6 import visualize_model_classification, combine_FB_features_across_stim, prep_FB_inputs_select_features_only, latency_to_reversal, fit_visualize_model_regression

# from predict_latency_prob_from_features_donut_6 import  combine_FB_features_across_stim
# from predict_latency_prob_from_features_donut_6 import prep_data_rev_prob, fit_eval_linearSVC, visualize_model_classification
# from predict_latency_prob_from_features_donut import visualize_model_classification,fit_eval_linearSVC combine_FB_features_across_stim


feature_list = [resampled_vel, resampled_acc, resampled_curve]
feature_labels = ["vel", "accel", "curv"]
rev_bin = resampled_rev
fps = 1.6
all_upcoming_revs,  X_features, Y_time_to_revs_flat, feature_names = prep_data_spont_rev_prob(feature_list, feature_labels, rev_bin, n_timesteps = 3, win_start = 1, win_end = 10, fps = fps, flatten = True)

print(all_upcoming_revs.shape) # n_tracks, n_fremas 
print(X_features.shape) # (n_tracks, n_frames-1, n_features)
print(resampled_vel.shape)
# data_prep_fnc = prep_data_rev_prob
# model_fit_fnc = fit_eval_linearSVC
# result_visualization_fnc = visualize_model_classification




# %%

feature_list = [resampled_vel, resampled_acc, resampled_curve]
feature_labels = ["vel", "accel", "curv"]
rev_bin = resampled_rev
fps = 1.6

all_timesteps = range(1, 15, 5) #range(1, 15, 3)
win_start = 1
win_end = 5
n_splits = 10
test_size = 0.3

score_label = "balanced accuracy"
model_fit_fnc = fit_eval_linearSVC
prep_data_fnc = prep_data_spont_rev_prob #vs prep_data_spont_rev_latency
title_label = "spont revs"
result_visualization_fnc = visualize_model_classification


best_models = predict_spont_revs_from_features(   feature_list, 
                                     feature_labels, rev_bin,
                                     fps, 
                                     win_start, win_end,
                                     model_fit_fnc, 
                                     score_label,
                                     result_visualization_fnc, 
                                     prep_data_fnc, 
                                     
                                     title_label = title_label, 
                                     all_timesteps = all_timesteps, 
                                    test_size = test_size, 
                                    n_splits = n_splits,  
                                    # n_train = n_train,
                                    # n_test = n_test, 

                                    single_feature = False, 
                                    n_pre_start = 0 ) 


feature_to_best_models = predict_spont_revs_from_features_single_feature(   feature_list, 
                                     feature_labels, rev_bin,
                                     fps, 
                                     win_start, win_end,
                                     model_fit_fnc, 
                                     score_label,
                                     result_visualization_fnc, 
                                     prep_data_fnc, 
                                     
                                     title_label = title_label, 
                                     all_timesteps = all_timesteps, 
                                    test_size = test_size, 
                                    n_splits = n_splits,  
                                    # n_train = n_train,
                                    # n_test = n_test, 

                                    single_feature = False, 
                                    n_pre_start = 0 ) 



##########
###########
###########

all_timesteps = range(1, 25, 5) #range(1, 15, 3)

feature_list = [exp_vel, exp_acc, exp_curve]
feature_labels = ["vel", "accel", "curv"]
rev_bin =  exp_rev


win_start = 1
win_end = 5
n_splits = 10
test_size = 0.3

fps = 6
score_label = "balanced accuracy"
model_fit_fnc = fit_eval_linearSVC
prep_data_fnc = prep_data_spont_rev_prob #vs prep_data_spont_rev_latency
title_label = "spont revs"
result_visualization_fnc = visualize_model_classification


best_models = predict_spont_revs_from_features(   feature_list, 
                                     feature_labels, rev_bin,
                                     fps, 
                                     win_start, win_end,
                                     model_fit_fnc, 
                                     score_label,
                                     result_visualization_fnc, 
                                     prep_data_fnc, 
                                     
                                     title_label = title_label, 
                                     all_timesteps = all_timesteps, 
                                    test_size = test_size, 
                                    n_splits = n_splits,  
                                    # n_train = n_train,
                                    # n_test = n_test, 

                                    single_feature = False, 
                                    n_pre_start = 0 ) 


feature_to_best_models = predict_spont_revs_from_features_single_feature(   feature_list, 
                                     feature_labels, rev_bin,
                                     fps, 
                                     win_start, win_end,
                                     model_fit_fnc, 
                                     score_label,
                                     result_visualization_fnc, 
                                     prep_data_fnc, 
                                     
                                     title_label = title_label, 
                                     all_timesteps = all_timesteps, 
                                    test_size = test_size, 
                                    n_splits = n_splits,  
                                    # n_train = n_train,
                                    # n_test = n_test, 

                                    single_feature = False, 
                                    n_pre_start = 0 ) 



            

# %%
all_timesteps = range(1, 15, 5) #range(1, 15, 3)

feature_list = [resampled_vel, resampled_acc, resampled_curve]
feature_labels = ["vel", "accel", "curv"]
rev_bin = resampled_rev

win_start = 1
win_end = 5
n_splits = 10
test_size = 0.3

fps = 1.6
score_label = "r2 score "
prep_data_fnc = prep_data_spont_rev_latency
model_fit_fnc = fit_eval_lasso
result_visualization_fnc = visualize_model_regression
title_label = "spont latency to rev\n resampled fps"


best_models = predict_spont_revs_from_features(   feature_list, 
                                     feature_labels, rev_bin,
                                     fps, 
                                     win_start, win_end,
                                     model_fit_fnc, 
                                     score_label,
                                     result_visualization_fnc, 
                                     prep_data_fnc, 
                                     
                                     title_label = title_label, 
                                     all_timesteps = all_timesteps, 

                                    n_splits = n_splits,  
                                    test_size = test_size, 

                                    single_feature = False, 
                                    n_pre_start = 0 ) 





# %%


all_timesteps = range(1, 25, 5) #range(1, 15, 3)

feature_list = [exp_vel, exp_acc, exp_curve]
feature_labels = ["vel", "accel", "curv"]
rev_bin =  exp_rev


win_start = 1
win_end = 5
n_splits = 10
test_size = 0.3

fps = 6
score_label = "r2 score "
prep_data_fnc = prep_data_spont_rev_latency
model_fit_fnc = fit_eval_lasso
result_visualization_fnc = visualize_model_regression
title_label = "spont latency to rev\n exp fps"


best_models = predict_spont_revs_from_features(   feature_list, 
                                     feature_labels, rev_bin,
                                     fps, 
                                     win_start, win_end,
                                     model_fit_fnc, 
                                     score_label,
                                     result_visualization_fnc, 
                                     prep_data_fnc, 
                                     
                                     title_label = title_label, 
                                     all_timesteps = all_timesteps, 

                                    n_splits = n_splits,  
                                    test_size = test_size, 

                                    single_feature = False, 
                                    n_pre_start = 0 ) 


# %%
for win_end in range(3,15, 2):
    all_timesteps = range(1, 15, 5) #range(1, 15, 3)

    # win_start = 1
    # win_end = 5
    n_splits = 10
    test_size = 0.3
    n_train, n_test = 1000,300
    fps = 1.6
    score_label = "balanced accuracy"
    model_fit_fnc = fit_eval_linearSVC
    prep_data_fnc = prep_data_spont_rev_prob #vs prep_data_spont_rev_latency
    title_label = "spont revs"
    result_visualization_fnc = visualize_model_classification


    best_models = predict_spont_revs_from_features(   feature_list, 
                                        feature_labels, rev_bin,
                                        fps, 
                                        win_start, win_end,
                                        model_fit_fnc, 
                                        score_label,
                                        result_visualization_fnc, 
                                        prep_data_fnc, 
                                        
                                        title_label = title_label, 
                                        all_timesteps = all_timesteps, 
                                        test_size = test_size, 
                                        n_splits = n_splits,  
                                        # n_train = n_train,
                                        # n_test = n_test, 

                                        single_feature = False, 
                                        n_pre_start = 0 ) 


# %%
all_timesteps = None

win_start = 1
win_end = 5
n_splits = 10
test_size = 0.3
n_train, n_test = 1000,300

model_fit_fnc = fit_eval_linearSVC
single_feature = False
score_label = "balanced accuracy"
result_visualization_fnc = visualize_model_classification
i = -1 #spont
title_label = "spont revs"
n_pre_start = 0 

with warnings.catch_warnings():
    for win_end in range(3,15, 2):
        
        fig, ax = plt.subplots()
        best_models = {} 
        all_scores_per_i = {}
        results_dict = {}


        score_means = []
        score_stds = []
        n_timesteps_list = []
        n_frames = rev_bin.shape[1]
        if all_timesteps is None: 
            all_timesteps = range(1, 15, 3)

        for n_timesteps in all_timesteps:
            print("i, n_timesteps, win_end",i,  n_timesteps, win_end)
            scores = []
            models = []

            y,  X, feature_names = prep_data_spont_rev_prob(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)
            ###if want to see how far back of a window cna be used) 
            y = y[n_pre_start:]
            X = X[:(X.shape[0]-n_pre_start)]
            # X_bal, y_bal = downsample_majority_np(X, y, majority_label=0, random_state=42)
            
            for split_idx in range(n_splits):
                # X_train, X_test, y_train, y_test = train_test_split(
                #     X, Y, test_size=test_size, random_state=split_idx)
                # X_train, X_test, y_train, y_test = train_test_split(
                #     X_bal, y_bal, test_size=test_size, random_state=split_idx)
                
                X_train, X_test, y_train, y_test = balanced_train_test_split(X, y, n_train, n_test, random_state=split_idx)
                best_model, y_pred, score = model_fit_fnc(X_train, X_test, y_train, y_test)
                scores.append(score)
                models.append((best_model,X_train, X_test, y_train, y_test))

            scores_mean = np.mean(scores)
            scores_std = np.std(scores)
            best_model_idx = np.argmax(scores)
            best_model_overall = models[best_model_idx]

            # Save results
            n_timesteps_list.append(n_timesteps)
            # all_r2_means.append(r2s)
            all_scores_per_i[(i, n_timesteps)] = scores
            score_means.append(scores_mean)
            score_stds.append(scores_std)
            results_dict[(i, n_timesteps)] = (scores_mean, scores_std)
            best_models[(i, n_timesteps)] = best_model_overall
            

        # cmap1 = cm.get_cmap('tab10', len(resampled_onsets))
        ax.errorbar(np.array(n_timesteps_list)/fps, score_means, yerr=score_stds, #color=cmap1(i),
                    fmt='-o', capsize=5, label = f"stim {i}")

        ax.set_title( f"{title_label}; stim{i}; win start {win_start}; win end {win_end}" )
        ax.set_xlabel("max time delay (sec)")
        ax.set_ylabel(f"{score_label}")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        model_label= f"{title_label}; stim{i}; win start {win_start}; win end {win_end}"#; lasso_a{np.round(alpha,2)}"
        best_model,X_train, X_test, y_train, y_test = best_models[(i, n_timesteps)]
        result_visualization_fnc(best_model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False)#, xlim = [-2,1])


        n_timesteps = all_timesteps[-2]
        model,X_train, X_test, y_train, y_test = best_models[(i, n_timesteps)] 

       
        if single_feature: 
            feature_lbl = feature_labels[0]
            feature_names = np.array([[f"{feature_lbl}_t-{np.round((n_timesteps - t - 1)/fps, 2)}"]
                                for t in range(n_timesteps)]).flatten()
            result_visualization_fnc(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False, coeffs = model.coef_[0])#, xlim = [-2,1])
            
        else: 
            _,  _, feature_names = prep_data_spont_rev_prob(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)
            
            model_label = ""
            best_model,X_train, X_test, y_train, y_test = best_models[(i, n_timesteps)]
            result_visualization_fnc(best_model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False)#, xlim = [-2,1])


            print("feature_names_all",feature_names)
        


# %%


# %%



