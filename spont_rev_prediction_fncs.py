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



import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from predict_latency_prob_from_features_donut import prep_FB_inputs_select_features_only_no_onset, fit_eval_linearSVC, visualize_model_classification
from predict_latency_prob_from_features_donut import fit_eval_lasso, visualize_model_regression

def mark_future_events(arr, start=1, end=10):
    arr = np.asarray(arr)
    out = np.zeros_like(arr, dtype=float)

    # Set NaNs where arr is 1
    out[arr == 1] = np.nan

    n = len(arr)
    for i in range(n):
        if arr[i] == 0:
            # Ensure window stays in bounds
            window = arr[i + start : min(i + end + 1, n)]
            if np.any(window == 1):
                out[i] = 1.0

    return out
def mark_future_events_with_distance(arr, start=1, end=10):
    arr = np.asarray(arr)
    out_dist = np.zeros_like(arr, dtype=float)

    # Set NaNs where arr is 1
    out_dist[arr == 1] = np.nan

    n = len(arr)
    for i in range(n):
        if not arr[i]:
            for offset in range(start, end + 1):
                j = i + offset
                if j < n and arr[j] == 1:
                    out_dist[i] = offset
                    break  # take the first (nearest) future 1
            # If no 1 found, stays 0

    return out_dist
def get_upcoming_revs(rev_bin, win_start = 1, win_end = 10):
    n_tracks, n_frames = rev_bin.shape
    all_upcoming_revs = []
    # features = []
    for track in range(n_tracks):
        
        rev_bin_track = rev_bin[track, :]
        upcoming_revs = mark_future_events(rev_bin_track, start=win_start, end=win_end)
        all_upcoming_revs.append(upcoming_revs[:, None])

    return np.concatenate(all_upcoming_revs, axis = 1).T
def get_time_to_rev(rev_bin, win_start = 1, win_end = 10):
    n_tracks, n_frames = rev_bin.shape
    all_time_to_revs = []
    # features = []
    for track in range(n_tracks):
        
        rev_bin_track = rev_bin[track, :]
       
        time_to_rev = mark_future_events_with_distance(rev_bin_track, start=win_start, end=win_end)
        
        all_time_to_revs.append(time_to_rev[:,None])
        
        # fig, ax = plt.subplots(3,1, figsize = (25,5))
        # rev_bin_track1 = rev_bin_track[~np.isnan(rev_bin_track)]
        # upcoming_revs = mark_future_events(rev_bin_track, start=win_start, end=win_end)
        # upcoming_revs1 = upcoming_revs[~np.isnan(rev_bin_track)]
        # time_to_rev1 = time_to_rev[~np.isnan(rev_bin_track)]
        # ax[0].imshow(rev_bin_track1[:, None].T, aspect='auto')
        # ax[1].imshow(upcoming_revs1[:, None].T, aspect='auto')
        # ax[1].imshow(time_to_rev1[:, None].T, aspect='auto')
        
        # if track == 6: 
        #     break
    return np.concatenate(all_time_to_revs, axis = 1).T
    
    
def prep_data_spont_rev_prob(features_list, feature_labels, rev_bin, n_timesteps = 3, win_start = 1, win_end = 10, fps = 1, flatten = True):
    n_tracks, n_frames = rev_bin.shape
    all_upcoming_revs = get_upcoming_revs(rev_bin, win_start = win_start, win_end = win_end)
    Y_time_to_revs = get_time_to_rev(rev_bin, win_start = win_start,  win_end = win_end)
    
    X_features,feature_names =  prep_FB_inputs_select_features_only_no_onset(
    
                features_list,
                feature_labels,     
                n_timesteps,
                fps = fps
                )
    Y_time_to_revs = Y_time_to_revs[:, n_timesteps:]
    Y_upcoming_revs = all_upcoming_revs[:, n_timesteps:]
    if flatten: 
        Y_upcoming_revs_flat = Y_upcoming_revs.reshape(-1) 
        Y_time_to_revs_flat = Y_time_to_revs.reshape(-1)
        X_features_flat = X_features.reshape(-1, X_features.shape[-1])
        non_nan_is = np.logical_and(np.logical_not(np.isnan(Y_upcoming_revs_flat)), np.all(np.logical_not(np.isnan(X_features_flat)), axis = 1)).astype('bool')
        return Y_upcoming_revs_flat[non_nan_is], X_features_flat[non_nan_is, :], Y_time_to_revs_flat[non_nan_is], feature_names
    else: 
        return Y_upcoming_revs, X_features, Y_time_to_revs_flat, feature_names 
    
    
def prep_data_spont_rev_latency(features_list, feature_labels, rev_bin, n_timesteps = 3, win_start = 1, win_end = 10, fps = 1, flatten = True):
    Y_upcoming_revs, X_features, Y_time_to_revs_flat, feature_names  = prep_data_spont_rev_prob(features_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = flatten)
    will_rev_is =(Y_upcoming_revs==1).astype('bool')
    print("CHECK TRUE!:" , np.all(Y_time_to_revs_flat[will_rev_is] >0))
    # return Y_upcoming_revs[will_rev_is], X_features[will_rev_is], Y_time_to_revs_flat[will_rev_is], feature_names
    return Y_time_to_revs_flat[will_rev_is], X_features[will_rev_is], Y_time_to_revs_flat[will_rev_is], feature_names


def downsample_majority_np(X, y, majority_label=0, random_state=42):
    rng = np.random.default_rng(random_state)

    # Boolean masks
    majority_mask = y == majority_label
    minority_mask = ~majority_mask

    # Split arrays
    X_majority = X[majority_mask]
    y_majority = y[majority_mask]
    X_minority = X[minority_mask]
    y_minority = y[minority_mask]

    # Downsample majority
    n_minority = len(y_minority)
    indices = rng.choice(len(X_majority), size=n_minority, replace=False)
    X_majority_down = X_majority[indices]
    y_majority_down = y_majority[indices]

    # Combine and shuffle
    X_bal = np.vstack([X_majority_down, X_minority])
    y_bal = np.concatenate([y_majority_down, y_minority])
    
    perm = rng.permutation(len(y_bal))
    return X_bal[perm], y_bal[perm]


def balanced_train_test_split(X, y, n_train, n_test, random_state=42):
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []

    for cat in classes:
        # Find indices for this class
        cls_indices = np.where(y == cat)[0]
        if len(cls_indices) < n_train + n_test:
            raise ValueError(f"Not enough samples of class {cat}: needed {n_train + n_test}, got {len(cls_indices)}")

        # Shuffle and split
        cls_indices = rng.permutation(cls_indices)
        train_idx = cls_indices[:n_train]
        test_idx = cls_indices[n_train:n_train + n_test]

        # Append data
        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])
        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])

    # Combine and shuffle
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)

    # Final shuffle
    train_perm = rng.permutation(len(y_train))
    test_perm = rng.permutation(len(y_test))

    return (
        X_train[train_perm], X_test[test_perm],
        y_train[train_perm], y_test[test_perm]
    )
    
# def downsample_all_classes_np_multiclass(X, y,classes_timeseries,  random_state=42):
#     rng = np.random.default_rng(random_state)
#     X = np.asarray(X)
#     y = np.asarray(y)

#     classes, counts = np.unique(classes_timeseries, return_counts=True)
#     min_count = np.min(counts)

#     X_bal_list = []
#     y_bal_list = []

#     for cls in classes:
#         cls_indices = np.where(classes_timeseries == cls)[0]
#         sampled_indices = rng.choice(cls_indices, size=min_count, replace=False)
#         X_bal_list.append(X[sampled_indices])
#         y_bal_list.append(y[sampled_indices])

#     # Combine and shuffle
#     X_bal = np.vstack(X_bal_list)
#     y_bal = np.concatenate(y_bal_list)

#     perm = rng.permutation(len(y_bal))
#     return X_bal[perm], y_bal[perm]



# def downsample_all_classes_and_split(arrays, label_index, test_size=0.2, random_state=42):
#     """
#     Parameters:
#     - arrays: list of np.arrays, all with same first dimension
#     - label_index: index of the array in `arrays` that contains the class labels (1D array)
#     - test_size: proportion of data to allocate to test set
#     - random_state: seed for reproducibility

#     Returns:
#     - train_arrays: list of arrays for training
#     - test_arrays: list of arrays for testing
#     """

#     rng = np.random.default_rng(random_state)
#     arrays = [np.asarray(arr) for arr in arrays]
#     y = arrays[label_index]
    
#     classes, counts = np.unique(y, return_counts=True)
#     min_count = np.min(counts)

#     # Collect balanced indices
#     balanced_indices = []
#     for cls in classes:
#         cls_indices = np.where(y == cls)[0]
#         sampled_indices = rng.choice(cls_indices, size=min_count, replace=False)
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

def downsample_all_classes_and_split_1(arrays, label_index, test_size=0.2, random_state=42, max_per_class=1000):
    """
    Parameters:
    - arrays: list of np.arrays, all with same first dimension
    - label_index: index of the array in `arrays` that contains the class labels (1D array)
    - test_size: proportion of data to allocate to test set
    - random_state: seed for reproducibility
    - max_per_class: maximum number of samples to retain per class

    Returns:
    - train_arrays: list of arrays for training
    - test_arrays: list of arrays for testing
    """
    from sklearn.model_selection import train_test_split
    import numpy as np

    rng = np.random.default_rng(random_state)
    arrays = [np.asarray(arr) for arr in arrays]
    y = arrays[label_index]
    
    classes, counts = np.unique(y, return_counts=True)
    min_count = min(np.min(counts), max_per_class)

    # Collect balanced indices
    balanced_indices = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        sampled_count = min(len(cls_indices), max_per_class)
        sampled_indices = rng.choice(cls_indices, size=sampled_count, replace=False)
        balanced_indices.extend(sampled_indices)

    # Shuffle balanced indices
    balanced_indices = rng.permutation(balanced_indices)

    # Extract balanced data for all arrays
    balanced_arrays = [arr[balanced_indices] for arr in arrays]
    y_bal = balanced_arrays[label_index]

    # Get train/test indices with stratification
    train_idx, test_idx = train_test_split(
        np.arange(len(y_bal)), test_size=test_size, random_state=random_state, stratify=y_bal
    )

    # Split all arrays using same indices
    train_arrays = [arr[train_idx] for arr in balanced_arrays]
    test_arrays = [arr[test_idx] for arr in balanced_arrays]

    return train_arrays, test_arrays
    
def balanced_train_test_split_1(X, y,Y_time_to_revs_flat,  n_train, n_test, random_state=42):
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = [], []

    for cat in classes:
        # Find indices for this class
        cls_indices = np.where(y == cat)[0]
        if len(cls_indices) < n_train + n_test:
            raise ValueError(f"Not enough samples of class {cat}: needed {n_train + n_test}, got {len(cls_indices)}")

        # Shuffle and split
        cls_indices = rng.permutation(cls_indices)
        train_idx = cls_indices[:n_train]
        test_idx = cls_indices[n_train:n_train + n_test]

        # Append data
        X_train_list.append(X[train_idx])
        y_train_list.append(y[train_idx])
        Y_time_to_revs_flat_train.append(Y_time_to_revs_flat[train_idx])
        
        
        X_test_list.append(X[test_idx])
        y_test_list.append(y[test_idx])
        Y_time_to_revs_flat_test.append(Y_time_to_revs_flat[test_idx])

    # Combine and shuffle
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    
    Y_time_to_revs_flat_train = np.concatenate(Y_time_to_revs_flat_train)
    Y_time_to_revs_flat_test = np.concatenate(Y_time_to_revs_flat_test)

    # Final shuffle
    train_perm = rng.permutation(len(y_train))
    test_perm = rng.permutation(len(y_test))

    return (
        X_train[train_perm], X_test[test_perm],
        y_train[train_perm], y_test[test_perm], 
        Y_time_to_revs_flat_train[train_perm], Y_time_to_revs_flat_test[test_perm]
    )


def predict_spont_revs_from_features(   feature_list, 
                                     feature_labels, rev_bin, fps, 
                                     win_start, win_end,
                                     model_fit_fnc, score_label,
                                     result_visualization_fnc,
                                     prep_data_fnc, 
                                     title_label = "spont revs", 
                                     all_timesteps = None, 

                                    n_splits = 10, 
                                    test_size = 0.3, 
                                    # n_train = 1000, n_test = 300, 

                                    single_feature = False, 
                                    n_pre_start = 0 , 
                                    max_per_class = 1000, 
            

                                    ):

    i = -1 #spont
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
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

            # y,  X, Y_time_to_revs_flat, feature_names = prep_data_spont_rev_prob(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)

            y,  X, Y_time_to_revs_flat, feature_names = prep_data_fnc(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)

            # y = y[np.logical_not(np.isnan(Y_time_to_revs_flat))]
            # X = X[np.logical_not(np.isnan(Y_time_to_revs_flat))]
            # Y_time_to_revs_flat = Y_time_to_revs_flat[np.logical_not(np.isnan(Y_time_to_revs_flat))]
            
            unique, counts = np.unique(Y_time_to_revs_flat, return_counts = True)
            for un, cnt in zip( unique, counts):
                    print("Y_time_to_revs_flat all unqiuecounts", un, cnt)
            ###if want to see how far back of a window cna be used) 
            y = y[n_pre_start:]
            Y_time_to_revs_flat = Y_time_to_revs_flat[n_pre_start:]
            X = X[:(X.shape[0]-n_pre_start)]
            # X_bal, y_bal = downsample_majority_np(X, y, majority_label=0, random_state=42)
            
            for split_idx in range(n_splits):
                # X_train, X_test, y_train, y_test = train_test_split(
                #     X, Y, test_size=test_size, random_state=split_idx)
                
                # X_train, X_test, y_train, y_test = train_test_split(
                #     X_bal, y_bal, test_size=test_size, random_state=split_idx)
                
                # X_train, X_test, y_train, y_test = balanced_train_test_split(X, y, n_train, n_test, random_state=split_idx)
                # train_arrays, test_arrays = downsample_all_classes_and_split([X, y, Y_time_to_revs_flat], 2, test_size=0.2, random_state=split_idx)
                print("Y_time_to_revs_flat all unqiuecounts","split_idx", i,  n_timesteps, win_end, split_idx)
                train_arrays, test_arrays = downsample_all_classes_and_split_1([X, y, Y_time_to_revs_flat], 1, test_size=test_size, random_state=split_idx, max_per_class=max_per_class)
                X_train, y_train, Y_time_to_revs_flat_train = train_arrays
                X_test, y_test, Y_time_to_revs_flat_test =  test_arrays
                
                # unique, counts = np.unique(Y_time_to_revs_flat_train, return_counts = True)
                # for un, cnt in zip( unique, counts):
                #     print("Y_time_to_revs_flat_train all unqiuecounts", un, cnt)
                
                # unique, counts = np.unique(Y_time_to_revs_flat_test, return_counts = True)
                # for un, cnt in zip( unique, counts):
                #     print("Y_time_to_revs_flat_test all unqiuecounts", un, cnt)
                # X_train, X_test, y_train, y_test, Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = balanced_train_test_split_1(X, y,Y_time_to_revs_flat,  n_train, n_test, random_state=split_idx)
                
                
                best_model, y_pred, score = model_fit_fnc(X_train, X_test, y_train, y_test )
                scores.append(score)
                models.append((best_model,X_train, X_test, y_train, y_test, Y_time_to_revs_flat_train, Y_time_to_revs_flat_test))

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
        ###visualiza
        n_timesteps = all_timesteps[-2]
        ax.errorbar(np.array(n_timesteps_list)/fps, score_means, yerr=score_stds, #color=cmap1(i),
                    fmt='-o', capsize=5, label = f"stim {i}")

        ax.set_title( f"{title_label}; stim{i}; win start {win_start}; win end {win_end}" )
        ax.set_xlabel("max time delay (sec)")
        ax.set_ylabel(f"{score_label}")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        model_label= f"{title_label}; stim{i}; win start {win_start}; win end {win_end}"#; lasso_a{np.round(alpha,2)}"
        best_model, X_train, X_test, y_train, y_test,  Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = best_models[(i, n_timesteps)]



        
        # best_model,X_train, X_test, y_train, y_test,  Y_time_to_revs_flat_train, Y_time_to_revs_flat_test = best_models[(i, n_timesteps)]
        
        # if single_feature: 
        #     feature_lbl = feature_labels[0]
        #     feature_names = np.array([[f"{feature_lbl}_t-{np.round((n_timesteps - t - 1)/fps, 2)}"]
        #                         for t in range(n_timesteps)]).flatten()
        #     result_visualization_fnc(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False, coeffs = model.coef_[0])#, xlim = [-2,1])
            
        # else: ]
        _,  _,_,  feature_names = prep_data_spont_rev_prob(feature_list, feature_labels, rev_bin, n_timesteps = n_timesteps, win_start = win_start, win_end = win_end, fps = fps, flatten = True)
        
        
        fig, ax = plt.subplots()
        y_pred = best_model.predict(X_test)
        if prep_data_fnc == prep_data_spont_rev_prob:
            print("prep_data_fnc == prep_data_spont_rev_prob")
            ax.hist(Y_time_to_revs_flat_test[(y_test==y_pred).astype('bool')],alpha = 0.3, density = True,  label = "hit")
            unique, counts = np.unique(Y_time_to_revs_flat_test[(y_test==y_pred).astype('bool')], return_counts = True)
            for un, cnt in zip( unique, counts):
                    print("hit unqiuecounts", un, cnt)
            
            ax.hist(Y_time_to_revs_flat_test[(y_test!=y_pred).astype('bool')],alpha = 0.3, density = True,  color = "purple", label = "miss")
            
            unique, counts = np.unique(Y_time_to_revs_flat_test[(y_test!=y_pred).astype('bool')], return_counts = True)
            for un, cnt in zip( unique, counts):
                    print("miss unqiuecounts", un, cnt)
            ax.legend()
            # ax.scatter(Y_time_to_revs_flat_test, err )
            ax.set_xlabel("time_to_rev")
            ax.set_ylabel("prob")
            ax.set_title(model_label)
        result_visualization_fnc(best_model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False)#, xlim = [-2,1])
        
        
        fig, ax = plt.subplots()
        ax.hist(Y_time_to_revs_flat_test)
        ax.set_title("Y_time_to_revs_flat_test")

        print("feature_names_all",feature_names)
            
        return best_models
    
            
def predict_spont_revs_from_features_single_feature(   
                                                    feature_list, 
                                     feature_labels, rev_bin, fps, 
                                     win_start, win_end,
                                     model_fit_fnc, score_label,
                                     result_visualization_fnc,
                                     prep_data_fnc, 
                                     title_label = "spont revs", 
                                     all_timesteps = None, 

                                    n_splits = 10, 
                                    test_size = 0.3, 
                                    # n_train = 1000, n_test = 300, 

                                    single_feature = False, 
                                    n_pre_start = 0 , 
            

                                    ):
    feature_to_best_models = {}
    for feature, feature_lbl in zip(feature_list, feature_labels):
        print(feature_lbl)
        best_models = predict_spont_revs_from_features(   [feature], 
                                        [feature_labels],
                                        rev_bin, fps, 
                                        win_start, win_end,
                                        model_fit_fnc, score_label,
                                        result_visualization_fnc,
                                        prep_data_fnc, 
                                        title_label = f"{feature_lbl}; {title_label}", 
                                        all_timesteps = all_timesteps, 

                                        n_splits = n_splits, 
                                        test_size = test_size, 
                                        # n_train = 1000, n_test = 300, 

                                        single_feature = single_feature, 
                                        n_pre_start = n_pre_start
                

                                        )
        feature_to_best_models[feature_lbl] = best_models
    return feature_to_best_models



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

def plot_mean_line_graph_with_individ_traces(data, n_pre_frames, fps = 6,  fig = None, ax = None, seconds = None,vline = None, title = "", 
                                             xlabel ="seconds post-stim start", 
                                             ylabel ="speed (um/s)", color = "blue", ylim =None,  ylabel_font_size = 8, 
                                             max_n_tracks = 30,line_width=1.0
                                             
                                             ):
    if fig is None:# types.NoneType 
        fig, ax = plt.subplots()
    if data.shape[0] ==0:
        return fig, ax
    
    n_tracks, n_frames = data.shape
    if n_tracks>max_n_tracks:
        track_is = np.arange(n_tracks)
        np.random.shuffle(track_is)
        sampled_track_is = track_is[:max_n_tracks]
        sampled_data = data[sampled_track_is,:]
    else:
        sampled_data = data
    average_row = np.nanmean(sampled_data, axis=0)
   
    # Plot each row with grayed lines
    if seconds is None:
        n_frames = sampled_data.shape[1]
        # seconds = np.arange(n_frames*params.n_stim_types)/params.fps
        seconds = np.arange(n_frames)/fps - n_pre_frames/fps
        vline = 0
    for row in sampled_data:
        ax.plot(seconds, row, color='gray', alpha=0.5, linewidth=line_width)

    # Plot the average from BehaviorAnalyzer.PlottingFunctions.get_event_speed_line_plots row with a more prominent line
    ax.plot(seconds, average_row, color=color, label='Average', linewidth=2)

    # Customize labels and legend
    if xlabel is not None: 
        ax.set_xlabel(xlabel)
    if ylabel is not None: 
        ax.set_ylabel(ylabel, fontsize = ylabel_font_size)
    if title is not None: 
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if vline is not None:
        ax.axvline(x=vline, color='black', linestyle='--')

    return fig, ax


# def align_feature_to_rev_start( exp_date_to_beh, exp_date_to_feature, n_pre_frames = 60, max_plotting_length_frames = 500, 
#                                rev_duration_min_frames = None, rev_duration_max_frames = None ,  rev_state = 1
#                                ):
   
#     buffered_event_features = []
#     for exp_date, featureseq in exp_date_to_feature.items():
#         # rsldsseq = date_to_discrete_rslds_states[exp_date]
#         behseq = exp_date_to_beh[exp_date]
#         featureseq = exp_date_to_feature[exp_date]
#         state_starts, state_ends = get_state_start_ends(rev_state , behseq,  starts= None, ends = None)
#         for rev_start_i,rev_end_i in zip( state_starts, state_ends):
#             rev_duration_frames = rev_end_i-rev_start_i
#             if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
#                 continue
#             if rev_duration_max_frames is not None and rev_duration_frames>rev_duration_max_frames:
#                 continue
#             frame_start = max(rev_start_i-n_pre_frames, 0)
#             frame_end = min(rev_start_i+max_plotting_length_frames, rev_end_i)
        
#             n_rev_frames = frame_end-rev_start_i
#             n_fwd_frames = rev_start_i-frame_start
#             buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
#             buffered_event_feature[:] = np.nan
#             buffered_event_feature[n_pre_frames+-1*n_fwd_frames:n_pre_frames+n_rev_frames] = featureseq[frame_start:frame_end]
#             buffered_event_features.append(buffered_event_feature[:, None])

#     if len(buffered_event_features)==0: 
#         return np.array([])

#     # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
#     buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
#     return buffered_event_features


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


def align_feature_to_rev_end_from_start_end_is(feature_val, event_start_end_is,  
                            n_pre_frames = 90, max_plotting_length_frames = 150, 
                            rev_duration_min_frames = -1, rev_duration_max_frames = np.inf
    ):

    n_frames = feature_val.shape[1]
    buffered_event_features = []
    durations = event_start_end_is[:,2]-event_start_end_is[:,1]
    dur_bool = np.logical_and(durations>rev_duration_min_frames, durations<=rev_duration_max_frames).astype('bool')
    event_start_end_is_duration = event_start_end_is[dur_bool]
    for track, rev_start_i, rev_end_i in event_start_end_is_duration:
    
    # for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
       
    #     # rsldsseq = date_to_discrete_rslds_states[exp_date]
    #     # behseq = exp_date_to_beh[exp_date]
    #     # featureseq = exp_date_to_feature[exp_date]
    #     state_starts, state_ends = get_state_start_ends(rev_state , z_w,  starts= None, ends = None)
        # for rev_start_i,rev_end_i in zip( state_starts, state_ends):
        #     rev_duration_frames = rev_end_i-rev_start_i
        #     if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
        #         continue
        #     if rev_duration_max_frames is not None and rev_duration_min_frames>rev_duration_max_frames:
        #         continue
            frame_start = max(0, rev_end_i-n_pre_frames) # which frame to reference in freature mat as start of window to plot  
            frame_end = min(rev_end_i+max_plotting_length_frames, n_frames)# which frame to reference in freature mat as end of window to plot
            frame_window = frame_end-frame_start # full elngth to plot 
            max_window_size = max_plotting_length_frames+n_pre_frames
            if n_frames< rev_end_i+max_plotting_length_frames:
                # n_pre_frames = rev_end_i+max_plotting_length_frames-n_frames
                # window_start = max_window_size-frame_window #+rev_end_i+max_plotting_length_frames-n_frames
                diff = rev_end_i+max_plotting_length_frames-n_frames
                window_start = max_window_size-frame_window
                window_start = 0
            else: 
                window_start = max_window_size-frame_window#n_pre_frames
                diff = 0
                

            buffered_event_feature = np.zeros(max_window_size)
            buffered_event_feature[:] = np.nan
            buffered_event_feature[window_start:n_pre_frames+max_plotting_length_frames-diff] = feature_val[track, frame_start:frame_end]
            buffered_event_features.append(buffered_event_feature[:, None])
                
            
            # n_rev_frames = frame_end-rev_start_i
            # n_fwd_frames = rev_start_i-frame_start
            # buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
            # buffered_event_feature[:] = np.nan
            # buffered_event_feature[n_pre_frames+-1*n_fwd_frames:n_pre_frames+n_rev_frames] = featureseq[frame_start:frame_end]
            # buffered_event_features.append(buffered_event_feature[:, None])
  
    if len(buffered_event_features)==0: 
        return np.array([])

    buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
    return buffered_event_features

def align_rev_to_stim_start_from_start_end_is(feature_val, event_start_end_is,  stim_start_frames, 
                            n_pre_frames = 60, max_plotting_length_frames = 500, 
                               rev_duration_min_frames = -1, rev_duration_max_frames = np.inf ,  rev_state = 1
                               ):
   
    n_frames = feature_val.shape[1]
    buffered_event_features = []
    durations = event_start_end_is[:,2]-event_start_end_is[:,1]
    dur_bool = np.logical_and(durations>rev_duration_min_frames, durations<=rev_duration_max_frames).astype('bool')
    event_start_end_is_duration = event_start_end_is[dur_bool]
    for track, rev_start_i, rev_end_i in event_start_end_is_duration:

            # print("n_pre_frames", n_pre_frames)
            # print("stim_start_frames", stim_start_frames)
            frame_start = max(stim_start_frames-n_pre_frames, 0)
            frame_end = min(stim_start_frames+max_plotting_length_frames, n_frames)
            # print("frame_start, frame_end", frame_start, frame_end)
            n_poststim_frames = frame_end-stim_start_frames# n_pre_frames
            n_prestim_frames = stim_start_frames-frame_start
            # print("n_prestim_frames", n_prestim_frames, "n_poststim_frames", n_poststim_frames)
            buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
            # print("n_pre_frames+max_plotting_length_frames", n_pre_frames+max_plotting_length_frames)
            # print("n_pre_frames+n_poststim_frames", n_pre_frames+n_poststim_frames )
            # print("n_pre_frames-n_prestim_frames", n_pre_frames-n_prestim_frames)
            buffered_event_feature[:] = np.nan
            buffered_event_feature[(n_pre_frames-n_prestim_frames):(n_pre_frames+n_poststim_frames)] = feature_val[track, frame_start:frame_end]
            buffered_event_features.append(buffered_event_feature[:, None])

    if len(buffered_event_features)==0: 
        return np.array([])

    # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
    buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
    return buffered_event_features
def align_feature_to_rev_start( z, feature_val, n_pre_frames = 60, max_plotting_length_frames = 500, 
                               rev_duration_min_frames = -1, rev_duration_max_frames = np.inf ,  rev_state = 1
                               ):

        
    buffered_event_features = []
    for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
        # rsldsseq = date_to_discrete_rslds_states[exp_date]
        
        # featureseq = exp_date_to_feature[exp_date]
        state_starts, state_ends = get_state_start_ends(
                                                        rev_state,
                                                        z_w,  
                                                        starts= None, 
                                                        ends = None
                                                        )
        for rev_start_i,rev_end_i in zip( state_starts, state_ends):
            rev_duration_frames = rev_end_i-rev_start_i
            if rev_duration_frames>=rev_duration_min_frames and rev_duration_frames<rev_duration_max_frames:

                frame_start = max(rev_start_i-n_pre_frames, 0)
                frame_end = min(rev_start_i+max_plotting_length_frames, rev_end_i)
            
                n_rev_frames = frame_end-rev_start_i
                n_fwd_frames = rev_start_i-frame_start
                buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
                buffered_event_feature[:] = np.nan
                buffered_event_feature[(n_pre_frames+-1*n_fwd_frames):(n_pre_frames+n_rev_frames)] = featureseq[frame_start:frame_end]
                buffered_event_features.append(buffered_event_feature[:, None])

    if len(buffered_event_features)==0: 
        return np.array([])

    # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
    buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
    return buffered_event_features


# def align_feature_to_rev_end(exp_date_to_beh, exp_date_to_feature,
#                             n_pre_frames = 90, max_plotting_length_frames = 150, 
#                             rev_duration_min_frames = None, rev_duration_max_frames = None,  rev_state = 1
# ):

    
#     buffered_event_features = []    
#     for exp_date, featureseq in exp_date_to_feature.items():
#         n_frames = featureseq.shape[0]
#         # rsldsseq = date_to_discrete_rslds_states[exp_date]
#         behseq = exp_date_to_beh[exp_date]
#         featureseq = exp_date_to_feature[exp_date]
#         state_starts, state_ends = get_state_start_ends(rev_state , behseq,  starts= None, ends = None)
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

def align_feature_to_rev_end( z, feature_val,
                            n_pre_frames = 90, max_plotting_length_frames = 150, 
                            rev_duration_min_frames = None, rev_duration_max_frames = None,  rev_state = 1
    ):

    
    buffered_event_features = []    
    # for exp_date, featureseq in exp_date_to_feature.items():
    for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
        n_frames = featureseq.shape[0]
        # rsldsseq = date_to_discrete_rslds_states[exp_date]
        # behseq = exp_date_to_beh[exp_date]
        # featureseq = exp_date_to_feature[exp_date]
        state_starts, state_ends = get_state_start_ends(rev_state , z_w,  starts= None, ends = None)
        for rev_start_i,rev_end_i in zip( state_starts, state_ends):
            rev_duration_frames = rev_end_i-rev_start_i
            if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
                continue
            if rev_duration_max_frames is not None and rev_duration_min_frames>rev_duration_max_frames:
                continue
            frame_start = max(0, rev_end_i-n_pre_frames) # which frame to reference in freature mat as start of window to plot  
            frame_end = min(rev_end_i+max_plotting_length_frames, n_frames)# which frame to reference in freature mat as end of window to plot
            frame_window = frame_end-frame_start # full elngth to plot 
            max_window_size = max_plotting_length_frames+n_pre_frames
            if n_frames< rev_end_i+max_plotting_length_frames:
                # n_pre_frames = rev_end_i+max_plotting_length_frames-n_frames
                # window_start = max_window_size-frame_window #+rev_end_i+max_plotting_length_frames-n_frames
                diff = rev_end_i+max_plotting_length_frames-n_frames
                window_start = max_window_size-frame_window
                window_start = 0
            else: 
                window_start = max_window_size-frame_window#n_pre_frames
                diff = 0
                

            buffered_event_feature = np.zeros(max_window_size)
            buffered_event_feature[:] = np.nan
            buffered_event_feature[window_start:n_pre_frames+max_plotting_length_frames-diff] = featureseq[frame_start:frame_end]
            buffered_event_features.append(buffered_event_feature[:, None])
                
            
            # n_rev_frames = frame_end-rev_start_i
            # n_fwd_frames = rev_start_i-frame_start
            # buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
            # buffered_event_feature[:] = np.nan
            # buffered_event_feature[n_pre_frames+-1*n_fwd_frames:n_pre_frames+n_rev_frames] = featureseq[frame_start:frame_end]
            # buffered_event_features.append(buffered_event_feature[:, None])
  
    if len(buffered_event_features)==0: 
        return np.array([])

    buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
    return buffered_event_features

def align_rev_to_stim_start(z, feature_val,stim_start_frames, stim_end_frames,
                            n_pre_frames = 60, max_plotting_length_frames = 500, 
                               rev_duration_min_frames = None, rev_duration_max_frames = None ,  rev_state = 1
                               ):
   
    buffered_event_features = []
    # for exp_date, featureseq in exp_date_to_feature.items():
    #     # rsldsseq = date_to_discrete_rslds_states[exp_date]
    #     behseq = exp_date_to_beh[exp_date]
    #     featureseq = exp_date_to_feature[exp_date]
    for i, (featureseq, z_w) in enumerate(zip(feature_val,z)):
        state_starts, state_ends = get_state_start_ends(rev_state , z_w,  starts= None, ends = None)
        for rev_start_i,rev_end_i in zip( state_starts, state_ends):
            rev_duration_frames = rev_end_i-rev_start_i
            if rev_duration_min_frames is not None and rev_duration_frames<rev_duration_min_frames:
                continue
            if rev_duration_max_frames is not None and rev_duration_frames>rev_duration_max_frames:
                continue
            frame_start = max(stim_start_frames-n_pre_frames, 0)
            frame_end = min(stim_start_frames+max_plotting_length_frames, rev_end_i)
        
            n_poststim_frames = frame_end-n_pre_frames
            n_prestim_frames = stim_start_frames-frame_start
            buffered_event_feature = np.zeros(n_pre_frames+max_plotting_length_frames)
            
            buffered_event_feature[:] = np.nan
            buffered_event_feature[(n_pre_frames-n_prestim_frames):n_pre_frames+n_poststim_frames] = featureseq[frame_start:frame_end]
            buffered_event_features.append(buffered_event_feature[:, None])

        if len(buffered_event_features)==0: 
            return np.array([])

    # buffered_event_features = np.concatenate(buffered_event_features, axis = 0)
    buffered_event_features = np.concatenate(buffered_event_features, axis = 1)
    return buffered_event_features