# import numpy as np
import matplotlib.pyplot as plt
import torch
import copy

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import copy
import pickle
import matplotlib.cm as cm

import warnings
    
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import copy
import pickle

import pandas as pd 
import numpy as np

from plot_donut_vs_phase import create_X_sequences
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.calibration import CalibratedClassifierCV

def fit_eval_linearSVC(X_train, X_test, y_train, y_test):
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    model = LinearSVC(penalty='l1', dual=False, max_iter=10000)
    # Grid search
    grid = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy')
    grid.fit(X_train, y_train)
    c = grid.best_params_['C']
    best_model = grid.best_estimator_
    
    y_pred = best_model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)  


    return best_model, y_pred, score


def fit_eval_calibrated_linearSVC_with_pro(X_train, X_test, y_train, y_test):
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    model = LinearSVC(penalty='l1', dual=False, max_iter=10000)
    # Grid search
    grid = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy')
    grid.fit(X_train, y_train)
    c = grid.best_params_['C']
    best_model = grid.best_estimator_
    
    # y_pred = best_model.predict(X_test)
    # score = balanced_accuracy_score(y_test, y_pred)  

    calibrated_svc = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
    calibrated_svc.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)  
    # probabilities = calibrated_svc.predict_proba(X)

    return calibrated_svc, y_pred, score



def fit_eval_lasso(X_train, X_test, y_train, y_test):
    alphas = np.logspace(-4, 2, 20)
    lasso = Lasso(max_iter=10000)
    grid_search = GridSearchCV(lasso, {'alpha': alphas}, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    score = r2_score(y_test, y_pred)
    return best_model, y_pred, score


    
def plot_coeffs(model, feature_names, model_label,  n_time_steps, feature_names_ordered = None, xlim = None, coeffs = None):
    if coeffs is None: 
            
        try: 
            
            coeffs = model.coef_
        except: 
            # print(model.coef_.shape)
            coeffs = model.coef_[0]
            # coeffs = pd.Series(model.coef_[0], index=feature_names)
            
        if feature_names_ordered : 
            coeffs = coeffs.reshape(n_time_steps,-1 ).reshape((len(feature_names), -1), order='F').flatten()
            feature_names = feature_names.reshape( n_time_steps, -1).reshape((len(feature_names), -1), order='F').flatten()
    # if coeffs is None: 
    try: 
        coeffs = pd.Series(model.coef_, index=feature_names)
    except:
        coeffs = pd.Series(model.coef_[0], index=feature_names)
        # sort_indices = np.argsort(np.argsort(feature_names_ordered)[np.argsort(feature_names)])
        # coeffs = coeffs[sort_indices]
        # feature_names = feature_names[sort_indices]

    # Filter non-zero coefficients and sort
    # nonzero_coeffs = coeffs[coeffs != 0].sort_values()
    nonzero_coeffs = coeffs

    # Plot
    fig_height = max(2, 0.3 * len(nonzero_coeffs))
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Plot on ax
    nonzero_coeffs.plot(kind='barh', ax=ax)

    # Labels and formatting
    ax.set_xlabel("Coefficient Value")
    if xlim is not None: 
        
        ax.set_xlim(xlim)
    ax.set_title(f"{model_label} coefficients")
    ax.grid(True, axis='x')

    fig.tight_layout()
    return fig, ax 
    # plt.show()
    
    
def visualize_model_classification(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, get_coeffs = True,  feature_names_ordered = None,  xlim = None, coeffs = None):
    
    # model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{model_label}Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model_label}Precision:", precision_score(y_test, y_pred, average='weighted'))
    print(f"{model_label}Recall:", recall_score(y_test, y_pred, average='weighted'))
    print(f"{model_label}F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    # disp = ConfusionMatrixDisplay.from_estimator(model, X_train,  y_train, cmap='Blues')
    # disp.ax_.set_title(f"{model_label}train")
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test,  y_test, cmap='Blues')
    disp.ax_.set_title(f"{model_label}test")
    if get_coeffs:
        fig, ax = plot_coeffs(model, feature_names, model_label, n_time_steps, feature_names_ordered = feature_names_ordered, xlim = xlim, coeffs = coeffs)
        ax.set_title(f"{model_label}")
    return model, y_pred


def visualize_model_regression(model, model_label, feature_names, X_train, X_test, y_train, y_test,n_time_steps, feature_names_ordered = None, xlim = None, coeffs = None, get_coeffs = True):
    model_real = copy.deepcopy(model)
    # model_real.fit(X_train, y_train)
    # # print(f"{model_label} Linear Regression R²:", model_real.score(X_test, y_test))
    # # plot_results(model_real, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, feature_names_ordered = feature_names_ordered, compare_to_suffle = compare_to_suffle  )
    # # print("Coefficients:", model.coef_)
    # # model_label= f"{i};lin"
    # model_shuffle = copy.deepcopy(model)
    # model_shuffle.fit(X_train, np.random.permutation(y_train))
    # # plot_results(model_shuffle, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, feature_names_ordered = feature_names_ordered, compare_to_suffle = compare_to_suffle  )
    
    
    y_pred_real = model_real.predict(X_test)
    fig, axs = plt.subplots(1,2)
    ax = axs[0]
    ax.scatter(y_test, y_pred_real, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Test Actual Latency")
    ax.set_ylabel("Test Predicted Latency")
    ax.set_title( f"{model_label}; R²:{np.round(model_real.score(X_test, y_test),2)}")
    
    # y_pred_shuffled = model_shuffle.predict(X_test)
    # ax = axs[1]
    # ax.scatter(y_test, y_pred_shuffled, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Test Actual Latency")
    ax.set_ylabel("Test Predicted Latency")
    # test_r2 = model_shuffle.score(X_test, y_test)
    
    # ax.set_title( f"{model_label}; shuffled; R²:{np.round(test_r2,2)}")
    plt.tight_layout()
    y_pred_real = model_real.predict(X_train)
    # fig, ax = plt.subplots()
    ax = axs[1]
    ax.scatter(y_train, y_pred_real, alpha=0.7)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax.set_xlabel("Train Actual Latency")
    ax.set_ylabel("Train Predicted Latency")
    test_r2 = model_real.score(X_train, y_train)
    ax.set_title( f"{model_label}; R²:{np.round(test_r2,2)}")

    ax.set_xlabel("Train Actual Latency")
    ax.set_ylabel("Train Predicted Latency")
    # ax.set_title( f"{model_label}; shuffled; R²:{np.round(model_shuffle.score(X_train, y_train),2)}")
    plt.tight_layout()
    if get_coeffs: 
        fig, ax = plot_coeffs(model, feature_names, model_label, n_time_steps, feature_names_ordered = feature_names_ordered, xlim = xlim, coeffs = coeffs)
    



def prep_FB_inputs_select_features_only_no_onset(
                    features_list, feature_labels, 
                     n_timesteps,
                    fps = None
                   ): 
    if fps is None: 
        fps = 1

    X_all_LSTM = []
    for new_worm_idx in range(len(features_list[0])):
        X_new_worm = np.stack([feature[new_worm_idx] for feature in features_list
                                ], axis=1)  
        feature_names = np.array([[f"{lbl}_t-{np.round((n_timesteps - t - 1)/fps, 2)}" for lbl in feature_labels] for t in range(n_timesteps)]).flatten()
        X_new_tensor = torch.tensor(X_new_worm, dtype=torch.float32)
        
        if n_timesteps> 0:
            X_new_seq1 = create_X_sequences(X_new_tensor, n_timesteps).numpy() #torch.Size([475, 5, 4]), = n_frames, time delay, 4 is the feature  #
        else: 
            X_new_seq1 = X_new_tensor.numpy()
        
        n_frames, delay, n_features = X_new_seq1.shape
        X_new_seq1 = X_new_seq1.reshape((n_frames,  delay*n_features)) # check how this is shaped it will be f1_t-T, f2_t-T, f3_t-T, f4_t-T, ; f1_t-T+1, f2_T+1, f3_T+1 f4_T+1

        
        X_all_LSTM.append(X_new_seq1)
    X_all_LSTM = np.array(X_all_LSTM)

    return X_all_LSTM, feature_names


def prep_FB_inputs_select_features_only_1(
                    features_list, feature_labels, 
                     n_timesteps, onset,
                    fps = None
                   ): 
    if fps is None: 
        fps = 1

    X_all_LSTM = []
    # for new_worm_idx in range(len(features_list[0])):
    # X_new_worm = np.stack([feature[new_worm_idx] for feature in features_list
    #                         ], axis=1)  
    feature_names = np.array([[f"{lbl}_t-{np.round((n_timesteps - t - 1)/fps, 2)}" 
                                                        for t in range(n_timesteps)  ]
                                                    for lbl in feature_labels ]).flatten()
        
    X_all  = []
    feature_names = []
    for feature_mat, lbl in zip(features_list, feature_labels):
        X_all.append(feature_mat[(onset-n_timesteps):onset])#ntrack, n frame
        feature_names.extend([f"{lbl}_t-{np.round((n_timesteps - t - 1)/fps, 2)}" 
                                                        for t in range(n_timesteps) ])
        


    return X_all_LSTM, feature_names


import numpy as np
import matplotlib.pyplot as plt
import torch
import copy

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import copy
import pickle
import matplotlib.cm as cm

import warnings
    
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, TweedieRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
import copy
import pickle

import pandas as pd 

from plot_donut_vs_phase import create_X_sequences
from predict_latency_prob_from_features_donut import prep_FB_inputs_select_features_only_no_onset 


def feature_all_to_resampled(inferred_phases_all, n_tracks):
   
    inferred_phases_all = inferred_phases_all.reshape(n_tracks, -1)
   
    n_timesteps = inferred_phases_all.shape[1]
    print("n_timesteps", n_timesteps)
    # inferred_phases_all_shifted = np.zeros((n_tracks,n_timesteps+5+8))+np.nan #ask Bennet about the 8 extra time steps missing
    # inferred_phases_all_shifted[:, 5:-8] = inferred_phases_all
    
    inferred_phases_all_shifted = np.zeros((n_tracks,n_timesteps+5+8+8))+np.nan #ask Bennet about the 8 extra time steps missing
    inferred_phases_all_shifted[:, 5:-16] = inferred_phases_all
    return inferred_phases_all_shifted

def flatten_and_remove_nans(resampled_features):
    return resampled_features.flatten()

def prep_FB_inputs(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, z,   T = 5
                #    times, beh_map, 
                #    lag=16, inclusion_thresh =3, 
                #    remove_revs=False, 
                  
                   ): 

    behavior_input = np.array(resampled_rev+2*resampled_turn, dtype=np.float64)
    behavior_input -= np.array(z).mean()
    behavior_input /= np.array(z).std()

    X_all_LSTM = []
    for new_worm_idx in range(len(resampled_vel)):
       
        X_new_worm = np.stack([resampled_vel[new_worm_idx], resampled_acc[new_worm_idx], resampled_curve[new_worm_idx], behavior_input[new_worm_idx]], axis=1)  
        # X_new_worm = np.stack([resampled_vel[new_worm_idx], resampled_acc[new_worm_idx], behavior_input[new_worm_idx]], axis=1)  
        X_new_tensor = torch.tensor(X_new_worm, dtype=torch.float32)
        
        if T > 0:
            X_new_seq1 = create_X_sequences(X_new_tensor, T).numpy() #torch.Size([475, 5, 4]), = n_frames, time delay, 4 is the feature  #
        else: 
            X_new_seq1 = X_new_tensor.numpy()
        
        n_frames, delay, n_features = X_new_seq1.shape
        X_new_seq1 = X_new_seq1.reshape((n_frames,  delay*n_features)) # check how this is shaped it will be f1_t-T, f2_t-T, f3_t-T, f4_t-T, ; f1_t-T+1, f2_T+1, f3_T+1 f4_T+1
        
            
        X_all_LSTM.append(X_new_seq1)
    # X_all_LSTM = np.concatenate(X_all_LSTM, axis = 1 ) # i guess should check if reshape how it reshapes.. ie if feautres are all together or not 
    X_all_LSTM = np.array(X_all_LSTM)#shape (n_tracks, n_frames, n_features)
    # feature_names = np.array([[f"vel_t-{T - t - 1}", f"accel_t-{T - t - 1}",
    #                                f"curv_t-{T - t - 1}", f"beh_t-{T - t - 1}"]
    #                               for t in range(T)]).flatten()

    return X_all_LSTM


    
    
# def plot_coeffs(model, feature_names, model_label,  n_time_steps, feature_names_ordered = None, xlim = None, coeffs = None):
#     if coeffs is None: 
            
#         try: 
            
#             coeffs = model.coef_
#         except: 
#             # print(model.coef_.shape)
#             coeffs = model.coef_[0]
#             # coeffs = pd.Series(model.coef_[0], index=feature_names)
            
#         if feature_names_ordered : 
#             coeffs = coeffs.reshape(n_time_steps,-1 ).reshape((len(feature_names), -1), order='F').flatten()
#             feature_names = feature_names.reshape( n_time_steps, -1).reshape((len(feature_names), -1), order='F').flatten()
#     # if coeffs is None: 
#     try: 
#         coeffs = pd.Series(model.coef_, index=feature_names)
#     except:
#         coeffs = pd.Series(model.coef_[0], index=feature_names)
#         # sort_indices = np.argsort(np.argsort(feature_names_ordered)[np.argsort(feature_names)])
#         # coeffs = coeffs[sort_indices]
#         # feature_names = feature_names[sort_indices]

#     # Filter non-zero coefficients and sort
#     # nonzero_coeffs = coeffs[coeffs != 0].sort_values()
#     nonzero_coeffs = coeffs

#     # Plot
#     fig_height = max(2, 0.3 * len(nonzero_coeffs))
#     fig, ax = plt.subplots(figsize=(8, fig_height))

#     # Plot on ax
#     nonzero_coeffs.plot(kind='barh', ax=ax)

#     # Labels and formatting
#     ax.set_xlabel("Coefficient Value")
#     if xlim is not None: 
        
#         ax.set_xlim(xlim)
#     ax.set_title(f"{model_label} coefficients")
#     ax.grid(True, axis='x')

#     fig.tight_layout()
#     return fig, ax 
#     # plt.show()
    
def fit_visualize_model_classifiction(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, get_coeffs = True,  feature_names_ordered = None,  xlim = None, coeffs = None):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # model_label = f"{i}; logistic"
    print(f"{model_label}Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model_label}Precision:", precision_score(y_test, y_pred, average='weighted'))
    print(f"{model_label}Recall:", recall_score(y_test, y_pred, average='weighted'))
    print(f"{model_label}F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    disp = ConfusionMatrixDisplay.from_estimator(model, X_train,  y_train, cmap='Blues')
    disp.ax_.set_title(f"{model_label}train")
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test,  y_test, cmap='Blues')
    disp.ax_.set_title(f"{model_label}test")
    if get_coeffs:
        fig, ax = plot_coeffs(model, feature_names, model_label, n_time_steps, feature_names_ordered = feature_names_ordered, xlim = xlim, coeffs = coeffs)
        ax.set_title(f"{model_label}")
    return model, y_pred

def fit_visualize_model_regression(model, model_label, feature_names, X_train, X_test, y_train, y_test,n_time_steps, feature_names_ordered = None, coeffs = None, get_coeffs = True):
    # model_real = copy.deepcopy(model)
    # model_real.fit(X_train, y_train)
    # print(f"{model_label} Linear Regression R²:", model.score(X_test, y_test))
    # # print("Coefficients:", model.coef_)
    # # model_label= f"{i};lin"
    # plot_results(model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, feature_names_ordered = feature_names_ordered)
    fit_visualize_model_regression_vs_shuffle(model, model_label, feature_names, X_train, X_test, y_train, y_test,n_time_steps, feature_names_ordered = feature_names_ordered, coeffs = coeffs, get_coeffs = get_coeffs)


def latency_to_reversal(reshaped_rev, start_time, max_latency=None):
    n_worms, T = reshaped_rev.shape
    latencies = np.full(n_worms, np.nan)

    for i in range(n_worms):
        end_time = T if max_latency is None else min(T, start_time + max_latency)
        future = reshaped_rev[i, start_time:end_time]
        reversal_indices = np.where(future)[0]
        if reversal_indices.size > 0:
            latencies[i] = reversal_indices[0]
    
    return latencies

    
def fit_visualize_model_regression_vs_shuffle(model, model_label, feature_names, X_train, X_test, y_train, y_test,n_time_steps, feature_names_ordered = None, coeffs = None, get_coeffs = True):
    model_real = copy.deepcopy(model)
    model_real.fit(X_train, y_train)
    # print(f"{model_label} Linear Regression R²:", model_real.score(X_test, y_test))
    # plot_results(model_real, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, feature_names_ordered = feature_names_ordered, compare_to_suffle = compare_to_suffle  )
    # print("Coefficients:", model.coef_)
    # model_label= f"{i};lin"
    model_shuffle = copy.deepcopy(model)
    model_shuffle.fit(X_train, np.random.permutation(y_train))
    # plot_results(model_shuffle, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, feature_names_ordered = feature_names_ordered, compare_to_suffle = compare_to_suffle  )
    
    
    y_pred_real = model_real.predict(X_test)
    fig, axs = plt.subplots(1,2)
    ax = axs[0]
    ax.scatter(y_test, y_pred_real, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Test Actual Latency")
    ax.set_ylabel("Test Predicted Latency")
    ax.set_title( f"{model_label}; R²:{np.round(model_real.score(X_test, y_test),2)}")
    
    y_pred_shuffled = model_shuffle.predict(X_test)
    ax = axs[1]
    ax.scatter(y_test, y_pred_shuffled, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Test Actual Latency")
    ax.set_ylabel("Test Predicted Latency")
    ax.set_title( f"{model_label}; shuffled; R²:{np.round(model_shuffle.score(X_test, y_test),2)}")
    # plt.tight_layout()
    y_pred_real = model_real.predict(X_train)
    fig, axs = plt.subplots(1,2)
    ax = axs[0]
    ax.scatter(y_train, y_pred_real, alpha=0.7)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax.set_xlabel("Train Actual Latency")
    ax.set_ylabel("Train Predicted Latency")
    ax.set_title( f"{model_label}; R²:{np.round(model_real.score(X_train, y_train),2)}")
    
    y_pred_shuffled = model_shuffle.predict(X_train)
    ax = axs[1]
    ax.scatter(y_train, y_pred_shuffled, alpha=0.7)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax.set_xlabel("Train Actual Latency")
    ax.set_ylabel("Train Predicted Latency")
    ax.set_title( f"{model_label}; shuffled; R²:{np.round(model_shuffle.score(X_train, y_train),2)}")
    # plt.tight_layout()
    
    if get_coeffs: 
        plot_coeffs(model_real, feature_names,model_label,n_time_steps,   feature_names_ordered = feature_names_ordered, coeffs = coeffs)


############
##############

def prep_FB_inputs_donut_only( inferred_phases_all, inferred_rad_all, resampled_onset, n_tracks):
    phase_resampled = feature_all_to_resampled(inferred_phases_all, n_tracks)
    radii_resampled = feature_all_to_resampled(inferred_rad_all, n_tracks) 
    
    X_donut_stim = np.concatenate([phase_resampled[:, resampled_onset, None] ,  radii_resampled[:, resampled_onset, None] ], axis = 1)
    # X_donut_stim = X_donut[:, resampled_onset,:]
    
    feature_names = ["phase", "radius"]
    return X_donut_stim, feature_names

    
def prep_FB_inputs_feature_and_donut(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, inferred_phases_all, inferred_rad_all,resampled_onset, n_tracks,   n_timesteps, z,lstm_lag = 0, include_beh = False):
    # print("n_timesteps", n_timesteps)
    X_stim_features, feature_names = prep_FB_inputs_features_only(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, resampled_onset,   n_timesteps,z, lstm_lag = lstm_lag, include_beh = include_beh)#shape (n_tracks, n_frames, n_features)

    X_donut_stim, donut_names = prep_FB_inputs_donut_only( inferred_phases_all, inferred_rad_all, resampled_onset, n_tracks)

    X_stim_all =  np.concatenate([X_stim_features ,  X_donut_stim], axis = 1)
    feature_names_all = np.array(feature_names.tolist()+donut_names)
    return X_stim_all, feature_names_all


    
# def visualize_model_classification(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, get_coeff = True,  feature_names_ordered = None,  xlim = None, coeffs = None):
    
#     # model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     print(f"{model_label}Accuracy:", accuracy_score(y_test, y_pred))
#     print(f"{model_label}Precision:", precision_score(y_test, y_pred, average='weighted'))
#     print(f"{model_label}Recall:", recall_score(y_test, y_pred, average='weighted'))
#     print(f"{model_label}F1 Score:", f1_score(y_test, y_pred, average='weighted'))
#     # disp = ConfusionMatrixDisplay.from_estimator(model, X_train,  y_train, cmap='Blues')
#     # disp.ax_.set_title(f"{model_label}train")
#     disp = ConfusionMatrixDisplay.from_estimator(model, X_test,  y_test, cmap='Blues')
#     disp.ax_.set_title(f"{model_label}test")
#     if get_coeff:
#         fig, ax = plot_coeffs(model, feature_names, model_label, n_time_steps, feature_names_ordered = feature_names_ordered, xlim = xlim, coeffs = coeffs)
#         ax.set_title(f"{model_label}")
#     return model, y_pred

def resample_fps(feature_arr, target_fps,  original_fps):
    # Resample using linear interpolation
    indices = np.arange(len(feature_arr))
    new_indices = np.linspace(0, len(feature_arr) - 1, int(len(feature_arr) * (target_fps / original_fps)))
    resampled_values = np.interp(new_indices, indices, feature_arr)
    return resampled_values

def resample_2d(inferred_phases_all_shifted, target_fps,  original_fps):
    n_tracks = inferred_phases_all_shifted.shape[0]
    resampled_tracks = []
    for track_i in range(n_tracks):
        resampled = resample_fps(inferred_phases_all_shifted[track_i, :], target_fps,  original_fps)
        resampled_tracks.append(resampled[None,:])

    inferred_phases_all_shifted_high_fps = np.concatenate(resampled_tracks, axis= 0 )
    return inferred_phases_all_shifted_high_fps



def plot_results_prediction(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_time_steps, feature_names_ordered = None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Test Actual Latency (s) ")
    ax.set_ylabel("Test Predicted Latency (s) ")
    ax.set_title( f"{model_label}; R²:{model.score(X_test, y_test)}")
    ax.set_ylim([0,6])
    ax.set_xlim([0,6])
    
    
    
    y_pred = model.predict(X_train)
    fig, ax = plt.subplots()
    ax.scatter(y_train, y_pred, alpha=0.7)
    ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2)
    ax.set_xlabel("Train Actual Latency (s) ")
    ax.set_ylabel("Train Predicted Latency (s) ")
    ax.set_title( f"{model_label}; R²:{model.score(X_train, y_train)}")
    ax.set_ylim([0,6])
    ax.set_xlim([0,6])
    
    plot_coeffs(model, feature_names,model_label,n_time_steps,   feature_names_ordered = feature_names_ordered)
  
# def prep_FB_inputs_select_features_only_no_onset(
#                     features_list, feature_labels, 
#                      n_timesteps,
#                     fps = None
#                    ): 
#     if fps is None: 
#         fps = 1

#     X_all_LSTM = []
#     for new_worm_idx in range(len(features_list[0])):
#         X_new_worm = np.stack([feature[new_worm_idx] for feature in features_list
#                                 ], axis=1)  
#         feature_names = np.array([[f"{lbl}_t-{(n_timesteps - t - 1)/fps}" for lbl in feature_labels] for t in range(n_timesteps)]).flatten()
#         X_new_tensor = torch.tensor(X_new_worm, dtype=torch.float32)
        
#         if n_timesteps> 0:
#             X_new_seq1 = create_X_sequences(X_new_tensor, n_timesteps).numpy() #torch.Size([475, 5, 4]), = n_frames, time delay, 4 is the feature  #
#         else: 
#             X_new_seq1 = X_new_tensor.numpy()
        
#         n_frames, delay, n_features = X_new_seq1.shape
#         X_new_seq1 = X_new_seq1.reshape((n_frames,  delay*n_features)) # check how this is shaped it will be f1_t-T, f2_t-T, f3_t-T, f4_t-T, ; f1_t-T+1, f2_T+1, f3_T+1 f4_T+1

        
#         X_all_LSTM.append(X_new_seq1)
#     X_all_LSTM = np.array(X_all_LSTM)

#     # if stim_onset is None: 
#     #     stim_onset = resampled_onset #- lstm_lag
#     # X_stim_features = X_all_LSTM[:, stim_onset, :]
#     return X_all_LSTM, feature_names

    # return X_stim_features, feature_names
  
def prep_FB_inputs_select_features_only(
                    features_list, feature_labels, 
                    resampled_onset,  n_timesteps,
                    z, 
                    lstm_lag = 0,
                    stim_onset = None, fps = None
                   ): 
    # if fps is None: 
    #     fps = 1

    # X_all_LSTM = []
    # for new_worm_idx in range(len(features_list[0])):
    #     X_new_worm = np.stack([feature[new_worm_idx] for feature in features_list
    #                             ], axis=1)  
    #     feature_names = np.array([[f"{lbl}_t-{(n_timesteps - t - 1)/fps}" for lbl in feature_labels] for t in range(n_timesteps)]).flatten()
    #     X_new_tensor = torch.tensor(X_new_worm, dtype=torch.float32)
        
    #     if n_timesteps> 0:
    #         X_new_seq1 = create_X_sequences(X_new_tensor, n_timesteps).numpy() #torch.Size([475, 5, 4]), = n_frames, time delay, 4 is the feature  #
    #     else: 
    #         X_new_seq1 = X_new_tensor.numpy()
        
    #     n_frames, delay, n_features = X_new_seq1.shape
    #     X_new_seq1 = X_new_seq1.reshape((n_frames,  delay*n_features)) # check how this is shaped it will be f1_t-T, f2_t-T, f3_t-T, f4_t-T, ; f1_t-T+1, f2_T+1, f3_T+1 f4_T+1

        
    #     X_all_LSTM.append(X_new_seq1)
    # X_all_LSTM = np.array(X_all_LSTM)
    X_all_LSTM, feature_names = prep_FB_inputs_select_features_only_no_onset(
                    features_list, feature_labels, 
                     n_timesteps,
                    fps = fps
                   )

    if stim_onset is None: 
        stim_onset = resampled_onset #- lstm_lag
    X_stim_features = X_all_LSTM[:, stim_onset, :]

    return X_stim_features, feature_names

def prep_FB_inputs_features_only(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, resampled_onset,  n_timesteps, z,
                                 lstm_lag = 0,
                                 stim_onset = None, 
                                 include_beh = False, 
                                 fps = None
                   ): 

    behavior_input = np.array(resampled_rev+2*resampled_turn, dtype=np.float64)
    behavior_input -= np.array(z).mean()
    behavior_input /= np.array(z).std()
    if fps is None: 
        fps = 1

    X_all_LSTM = []
    for new_worm_idx in range(len(resampled_vel)):
        if include_beh:
            X_new_worm = np.stack([resampled_vel[new_worm_idx], resampled_acc[new_worm_idx], resampled_curve[new_worm_idx], behavior_input[new_worm_idx]], axis=1)  

            feature_names = np.array([[f"vel_t-{np.round((n_timesteps - t - 1)/fps, 2)}", f"accel_t-{np.round((n_timesteps - t - 1)/fps, 2)}",
                                f"curv_t-{np.round((n_timesteps - t - 1)/fps, 2)}", f"beh_t-{np.round((n_timesteps - t - 1)/fps, 2)}"]
                                for t in range(n_timesteps)]).flatten()
        else: 
            X_new_worm = np.stack([resampled_vel[new_worm_idx], resampled_acc[new_worm_idx], resampled_curve[new_worm_idx], 
                                   
                                #    behavior_input[new_worm_idx]
                                   
                                   ], axis=1)  
            feature_names = np.array([[f"vel_t-{np.round((n_timesteps - t - 1)/fps, 2)}", f"accel_t-{np.round((n_timesteps - t - 1)/fps, 2)}",
                                f"curv_t-{np.round((n_timesteps - t - 1)/fps, 2)}"]
                                for t in range(n_timesteps)]).flatten()
        # X_new_worm = np.stack([resampled_vel[new_worm_idx], resampled_acc[new_worm_idx], behavior_input[new_worm_idx]], axis=1)  
        X_new_tensor = torch.tensor(X_new_worm, dtype=torch.float32)
        
        if n_timesteps> 0:
            X_new_seq1 = create_X_sequences(X_new_tensor, n_timesteps).numpy() #torch.Size([475, 5, 4]), = n_frames, time delay, 4 is the feature  #
        else: 
            X_new_seq1 = X_new_tensor.numpy()
        
        n_frames, delay, n_features = X_new_seq1.shape
        X_new_seq1 = X_new_seq1.reshape((n_frames,  delay*n_features)) # check how this is shaped it will be f1_t-T, f2_t-T, f3_t-T, f4_t-T, ; f1_t-T+1, f2_T+1, f3_T+1 f4_T+1

        
        X_all_LSTM.append(X_new_seq1)
    # X_all_LSTM = np.concatenate(X_all_LSTM, axis = 1 ) # i guess should check if reshape how it reshapes.. ie if feautres are all together or not 
    X_all_LSTM = np.array(X_all_LSTM)#shape (n_tracks, n_frames, n_features)

    if stim_onset is None: 
        stim_onset = resampled_onset #- lstm_lag
    X_stim_features = X_all_LSTM[:, stim_onset, :]
    # X_stim_features = X_all_LSTM[:, resampled_onset - lstm_lag, :]

    return X_stim_features, feature_names


def combine_FB_features_across_stim(
    stim_i_to_features,stim_i_to_Ys,  feature_names, 
  
                   ): 
    feature_names_all = np.array(feature_names.tolist()+[f"stim_{stim_i}" for stim_i in stim_i_to_features.keys()])
    
    all_stim_X_stim_features  = []
    all_stim_Y = []
    n_stim = len(stim_i_to_features.keys())
    for i, (stim_i, X_stim_features) in enumerate(stim_i_to_features.items()):
        n_frames, n_features = X_stim_features.shape
        stim_bin_features = np.zeros((n_frames,n_stim ))
        stim_bin_features[:, i] = 1
        all_stim_X_stim_features.append(np.concatenate([X_stim_features, stim_bin_features], axis =1))
        all_stim_Y.append(stim_i_to_Ys[stim_i])
    all_stim_X_stim_features = np.concatenate(all_stim_X_stim_features, axis =0)
    all_stim_Y = np.concatenate(all_stim_Y, axis = 0 )
        
    
    return all_stim_X_stim_features, all_stim_Y, feature_names_all



def prep_data_latency(X_stim_all, Y_latency):
    valid_indices = np.argwhere((~np.isnan(Y_latency)) & (Y_latency != 0)).flatten() 
    # print("in pre data latency")
    # print("valid_indices", valid_indices)
    # print("valid_indices", valid_indices.shape)
    # print("X_stim_all.shape", X_stim_all.shape)
    
    X_stim_all_latency = X_stim_all[valid_indices]
    # print("X_stim_all_latency.shape", X_stim_all_latency.shape)
    Y_latency_valid = Y_latency[valid_indices]
    return  X_stim_all_latency, Y_latency_valid


def prep_data_rev_prob(X_stim_all, Y_latency):
    no_rev_at_onset = Y_latency!=0
    Y_prob = np.logical_not(np.isnan(Y_latency[no_rev_at_onset]))# classfy y/n did rev happen
    X_prob = copy.deepcopy(X_stim_all[no_rev_at_onset])
    return X_prob, Y_prob

# def fit_eval_lasso(X_train, X_test, y_train, y_test):
#     alphas = np.logspace(-4, 2, 20)
#     lasso = Lasso(max_iter=10000)
#     grid_search = GridSearchCV(lasso, {'alpha': alphas}, cv=5, scoring='r2')
#     grid_search.fit(X_train, y_train)

#     best_model = grid_search.best_estimator_
#     y_pred = best_model.predict(X_test)
#     score = r2_score(y_test, y_pred)
#     return best_model, y_pred, score

# def fit_eval_linearSVC(X_train, X_test, y_train, y_test):
    
#     param_grid = {
#         'C': [0.001, 0.01, 0.1, 1, 10, 100]
#     }

#     model = LinearSVC(penalty='l1', dual=False, max_iter=10000)
#     # Grid search
#     grid = GridSearchCV(model, param_grid, cv=5, scoring='balanced_accuracy')
#     grid.fit(X_train, y_train)
#     c = grid.best_params_['C']
#     best_model = grid.best_estimator_
    
#     y_pred = best_model.predict(X_test)
#     score = balanced_accuracy_score(y_test, y_pred)  


#     return best_model, y_pred, score

def train_model_feature_list(data_prep_fnc,
                               model_fit_fnc, 
                               result_visualization_fnc, 
                               feature_list, 
                               feature_labels,
                               z, 
                               data_onsets,
                               exp_onsets, 
                               exp_rev, 
                               durations,
                               fps,
                           
                               title_label,
                               test_size = 0.3, 
                               n_splits = 10, 
                               lstm_lag = 0, 
                               score_label = "",
                               single_feature = False, 
                               rev_bin = None, 
                               all_timesteps = None, 
                               check_warnings = True
                            ):
    if check_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fig, ax = plt.subplots()
            best_models = {} 
            all_scores_per_i = {}
            results_dict = {}
            for i, (data_onset, exp_onset) in enumerate(zip(data_onsets, exp_onsets)):
                
                onset = data_onset-1
                score_means = []
                score_stds = []
                n_timesteps_list = []
                Y_latency = latency_to_reversal(exp_rev, exp_onset, max_latency = 6*durations[i]+1)/6 # nan where not 
                if rev_bin is not None: 
                    fig3, ax3 = plt.subplots(figsize = (8,20))
                    vline = onset - lstm_lag
                    sort = np.argsort(Y_latency)
                    ax3.imshow(rev_bin[sort, vline-25:vline+25])
                    
                    ax3.set_title(f"onset: {onset}")
                    ax3.axvline(25, color = "red")
                    
                
                if all_timesteps is None: 
                    all_timesteps = range(1, 15, 3)

                for n_timesteps in all_timesteps:
                    print("i, n_timesteps",i,  n_timesteps)
                    scores = []
                    models = []

                

                    X_stim_all, feature_names_all = prep_FB_inputs_select_features_only(
                        feature_list, feature_labels, onset,  n_timesteps, z, lstm_lag = lstm_lag, fps = fps
                                    )
                    


                    
                    X,Y = data_prep_fnc(X_stim_all, Y_latency)



                    for split_idx in range(n_splits):
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, Y, test_size=test_size, random_state=split_idx)
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

                ax.set_title(title_label)
                ax.set_xlabel("max time delay (sec)")
                ax.set_ylabel(f"{score_label}")
                ax.legend()
                plt.grid(True)
                plt.tight_layout()


            n_timesteps = all_timesteps[-2]
            for i in range(len(exp_onsets)):
                model,X_train, X_test, y_train, y_test = best_models[(i, n_timesteps)] 

                model_label= f"{title_label}; stim{i}"#; lasso_a{np.round(alpha,2)}"
                if single_feature: 
                    feature_lbl = feature_labels[0]
                    feature_names = np.array([[f"{feature_lbl}_t-{np.round((n_timesteps - t - 1)/fps, 2)}"]
                                        for t in range(n_timesteps)]).flatten()
                    result_visualization_fnc(model, model_label, feature_names, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False, coeffs = model.coef_[0])#, xlim = [-2,1])
                    
                else: 
                    _,  feature_names_all = prep_FB_inputs_select_features_only(
                        # resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn,
                        feature_list, feature_labels, onset,  n_timesteps, z, lstm_lag = lstm_lag,  fps = fps
                                    )
                    print("feature_names_all",feature_names_all)
                    result_visualization_fnc(model, model_label, feature_names_all, X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False)#, xlim = [-2,1])


    
def train_model_single_feature(data_prep_fnc,
                               model_fit_fnc, 
                               result_visualization_fnc, 
                               feature_list, 
                               feature_labels,
                               z, 
                               data_onsets,
                               exp_onsets, 
                               exp_rev, 
                               durations,
                               fps,
                           
                               title_label,
                               test_size = 0.3, 
                               n_splits = 10, 
                               lstm_lag = 0, 
                               rev_bin = None, 
                               score_label = "",
                               all_timesteps= None, 
                               check_warnings = True
                               ):
    
    
    for feature, feature_lbl in zip(feature_list, feature_labels
                                                    ):
        
        single_feature_list = [feature]
        single_feature_labels = [feature_lbl]
        train_model_feature_list(data_prep_fnc,
                               model_fit_fnc, 
                               result_visualization_fnc, 
                               single_feature_list, 
                               single_feature_labels,
                               z, 
                               data_onsets,
                               exp_onsets, 
                               exp_rev, 
                               durations,
                               fps,
                           
                               title_label,
                               test_size =test_size, 
                               n_splits = n_splits, 
                               score_label = score_label,
                               lstm_lag = lstm_lag, 
                               single_feature = True, 
                               rev_bin = rev_bin, 
                               all_timesteps  = all_timesteps, 
                               check_warnings = check_warnings
                            )
        



def get_plot_multi_stim_model(
                        data_prep_fnc,
                        model_fit_fnc, 
                        result_visualization_fnc, 
                        feature_list, 
                        feature_labels,
                        z, 
                        data_onsets,
                        exp_onsets, 
                        exp_rev, 
                        durations,
                        fps,
                        score_fnc, 
                        # title_label,
                        test_size = 0.3, 
                        n_splits = 10, 
                        lstm_lag = 0, 
                        
                        title_label = "", 
                        all_timesteps = None, 
                        check_warnings = True, 
                        visualize_individ_stim = False, 
                        score_label = ""
                        
        ):
        
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                if all_timesteps is None: 
                        all_timesteps = np.arange(1, 15, 3)

                        
                results_dict_score_overall = {}
                best_models_score_overall ={}
                results_dict_score_stim ={}
                best_models_score_stim = {}

                
                t_to_all_data = {}
                t_to_all_feature_names ={}
                t_to_all_models = {}
                t_to_all_stim_scores ={}
                t_to_best_model_using_min_idx_score ={}
                
                # n_timesteps_list = []

                t_to_best_model_idx_score = {}
                        

                fig, ax = plt.subplots()
                ax.set_title(f"{title_label}")
                best_models = {} 
                all_scores_per_i = {}
                results_dict = {}
                for n_timesteps in all_timesteps:
                        
                        all_data = []
                        n_stim = len(data_onsets)
                        # stim_i_to_score_means = {i:[] for i in range(n_stim)}
                        # stim_i_to_score_stds = {i:[] for i in range(n_stim)}
                        
                        stim_i_to_Xs = {}
                        stim_i_to_Ys = {}
                
                        for i, (data_onset, exp_onset) in enumerate(zip(data_onsets, exp_onsets)):
                                onset = data_onset-1
                                score_means = []
                                score_stds = []
                                # n_timesteps_list = []
                                Y_latency = latency_to_reversal(exp_rev, exp_onset, max_latency = 6*durations[i]+1)/6 # nan where not 

                                X_stim_all, feature_names_all = prep_FB_inputs_select_features_only(
                                feature_list, feature_labels, onset,  n_timesteps, z, lstm_lag = lstm_lag, fps = fps
                                                )
                                # print("feature_labels", feature_labels)
                                print(len(feature_list))
                                print(len(feature_list))
                                X,Y = data_prep_fnc(X_stim_all, Y_latency)
                                
                                stim_i_to_Xs[i] = X
                                stim_i_to_Ys[i] = Y
                                print("i, n_timesteps",i,  n_timesteps)
                                # print("i, X.shape", i, X.shape)
                                scores = []
                                models = []

                        all_stim_X_stim_features, all_stim_Y, feature_names_all = combine_FB_features_across_stim(stim_i_to_Xs, stim_i_to_Ys,  feature_names_all)
                        t_to_all_feature_names[n_timesteps] = feature_names_all
                        overall_scores = []
                        all_stim_scores =[]# np.zeros((n_splits,n_stim))
                        best_model_idx = []
                        models = [] # n_splits
                        for split_idx in range(n_splits):
                                X_train, X_test, y_train, y_test = train_test_split(all_stim_X_stim_features, all_stim_Y,  test_size=test_size, random_state=split_idx)
                                all_data.append((X_train, X_test, y_train, y_test))
                                
                                #get best model over all stim 
                                best_model, y_pred, score = model_fit_fnc(X_train, X_test, y_train, y_test)
                                scores.append(score)
                                models.append(best_model)#,X_train, X_test, y_train, y_test))

                                scores_mean = np.mean(scores)
                                scores_std = np.std(scores)
                                best_model_idx = np.argmax(scores)
                                best_model_overall = models[best_model_idx]
                                
                                
                                #get stim_i for each row in test data
                                stim_bin_test = X_test[:, -1*n_stim:]
                                stim_i_test = np.argmax(stim_bin_test, axis = 1)
                                stim_scores = [] #model specfic for each stim 
                                for j in range(n_stim):
                                        
                                        X_test_stim = X_test[stim_i_test == j, :]
                                        y_test_stim = y_test[stim_i_test == j]
                                        
                                        y_pred = best_model.predict(X_test_stim)
                                        stim_score = score_fnc(y_test_stim, y_pred)  
                                        stim_scores.append(stim_score)
                                all_stim_scores.append(stim_scores)
                                
                                y_pred = best_model.predict(X_test)
                                overall_score = score_fnc(y_test, y_pred) ###
                                
                                overall_scores.append(overall_score)
                                models.append(best_model)
                                

                                # Save results
                                # n_timesteps_list.append(n_timesteps)
                                # all_r2_means.append(r2s)
                                all_scores_per_i[(i, n_timesteps)] = scores
                                score_means.append(scores_mean)
                                score_stds.append(scores_std)
                                results_dict[(i, n_timesteps)] = (scores_mean, scores_std)
                                best_models[(i, n_timesteps)] = best_model_overall
                        
                        all_stim_scores = np.array(all_stim_scores)#n_splits,n_stim
                        t_to_all_models[ n_timesteps] = models
                        
                        
                        ###overall scores
                        accuracy_mean_overall = np.mean(overall_scores)
                        
                        accuracy_std_overall= np.std(overall_scores)
                        # print("accuracy_mean_overall", accuracy_mean_overall)
                        # print("accuracy_mean_overall", accuracy_mean_overall)
                        
                        best_model_idx_score_overall = np.argmax(overall_scores)
                        # np.argmin(overall_scores)
                        best_model_score_overall = models[best_model_idx_score_overall]
                        t_to_best_model_idx_score[n_timesteps] = best_model_idx_score_overall
                        
                        results_dict_score_overall[n_timesteps] = (accuracy_mean_overall, accuracy_std_overall)
                        best_models_score_overall[ n_timesteps] = best_model_score_overall
                        
                        #####stim scores
                        accuracy_mean_stim_specific = np.mean(all_stim_scores, axis =0)
                        accuracy_std_stim_specific = np.std(all_stim_scores, axis =0)

                        
                        best_model_idx_score_stim_min = np.argmax(np.min(all_stim_scores, axis = 1)) # has lowest min on all stim 
                        best_model_score_stim_min = models[best_model_idx_score_stim_min]
                        t_to_best_model_using_min_idx_score[n_timesteps] = best_model_idx_score_stim_min
                        

                        results_dict_score_stim[n_timesteps] = (accuracy_mean_stim_specific, accuracy_std_stim_specific)
                        best_models_score_stim[ n_timesteps] = best_model_score_stim_min
                        
                                        
                        t_to_all_stim_scores[n_timesteps] = all_stim_scores
                        t_to_all_data[n_timesteps] = all_data
                        

                        # n_timesteps_list.append(n_timesteps)
                ####plot overall mean and std overall and for each stim  
                fig, ax = plt.subplots()
                accuracy_mean_overall_over_t = []
                accuracy_std_overall_over_t = []
                stim_acc_mean_over_t = []
                stim_acc_std_over_t = []
                for n_timesteps in all_timesteps:

                        accuracy_mean_overall, accuracy_std_overall = results_dict_score_overall[n_timesteps]
                        accuracy_mean_overall_over_t.append(accuracy_mean_overall)
                        accuracy_std_overall_over_t.append(accuracy_std_overall)
                        accuracy_mean_stim_specific, accuracy_std_stim_specific = results_dict_score_stim[n_timesteps]
                        stim_acc_mean_over_t.append(accuracy_mean_stim_specific)
                        stim_acc_std_over_t.append(accuracy_std_stim_specific)
                # print(n_timesteps_list)/
                stim_acc_mean_over_t = np.array(stim_acc_mean_over_t)
                stim_acc_std_over_t = np.array(stim_acc_std_over_t)
                for j in range(n_stim):
                        ax.errorbar(np.array(all_timesteps)/fps, stim_acc_mean_over_t[:,j], yerr=stim_acc_std_over_t[:,j], fmt='-o', capsize=5, label = f"stim {j}")#color=cmap1(i),
                        
                ax.errorbar(np.array(all_timesteps)/fps, accuracy_mean_overall_over_t, yerr=accuracy_std_overall_over_t, #color=cmap1(i),
                                fmt='-o', capsize=5, label = f"overall")
                ax.set_title(f'{title_label} mean {score_label} performance over models for given stim ')
                ax.set_ylabel(f" {score_label} ")
                ax.set_xlabel("max time delay (s)")
                ax.legend()
                
                
                ####plot score overall score on each stim (each line a model)
                for n_timesteps in all_timesteps:
                        all_stim_scores = t_to_all_stim_scores[n_timesteps]
                        
                        fig, ax = plt.subplots()
                        for k in range(n_splits): 
                                ax.plot(np.arange(n_stim), all_stim_scores[k])
                        ax.set_title(f" {title_label} {score_label}  across all models; timestep{n_timesteps}")
                        ax.set_xlabel("stim number")
                        ax.set_ylabel(f"{score_label}")
                        idx = t_to_best_model_idx_score[n_timesteps]
                        ax.plot(np.arange(n_stim), all_stim_scores[idx], linewidth = 3, label = "best")
                        
                        idx = t_to_best_model_using_min_idx_score[n_timesteps]
                        ax.plot(np.arange(n_stim), all_stim_scores[idx], linewidth = 3, label = "best using min")
                        X_train, X_test, y_train, y_test = t_to_all_data[n_timesteps][idx]
                        # best_model,X_train, X_test, y_train, y_test = best_models_score_stim[n_timesteps] 
                        result_visualization_fnc(best_models_score_stim[n_timesteps] , f"{title_label}", t_to_all_feature_names[n_timesteps], X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False)#, xlim = [-2,1])
                        stim_bin_test = X_test[:, -1*n_stim:]
                        stim_i_test = np.argmax(stim_bin_test, axis = 1)
                        
                        stim_bin_train = X_train[:, -1*n_stim:]
                        stim_i_train = np.argmax(stim_bin_train, axis = 1)
                        print(stim_i_train.shape)
                        # print()
                        # print("stim_i_test", stim_i_test)
                        if visualize_individ_stim:
                                for j in range(n_stim):
                                                
                                        X_test_stim = X_test[stim_i_test == j, :]
                                        y_test_stim = y_test[stim_i_test == j]
                                        X_train_stim = X_train[stim_i_train == j, :]
                                        y_train_stim = y_train[stim_i_train == j]
                                        
                                        result_visualization_fnc(best_models_score_stim[n_timesteps] , f"{title_label}; stim: {j}", t_to_all_feature_names[n_timesteps], X_train_stim, X_test_stim, y_train_stim, y_test_stim, n_timesteps,  feature_names_ordered = False, get_coeffs = False)#, xlim = [-2,1])
                                        
                        
        return best_models_score_stim, t_to_best_model_using_min_idx_score, t_to_all_data, t_to_all_feature_names

def get_plot_multi_stim_model_single_feature(
                        data_prep_fnc,
                        model_fit_fnc, 
                        result_visualization_fnc, 
                        feature_list, 
                        feature_labels,
                        z, 
                        data_onsets,
                        exp_onsets, 
                        exp_rev, 
                        durations,
                        fps,

                        # title_label,
                        score_fnc,
                        test_size = 0.3, 
                        n_splits = 10, 
                        lstm_lag = 0, 
                        
                        all_timesteps = None, 
                        check_warnings = True, 
                        score_label = "", 
                        visualize_individ_stim = False, 
                        get_coeffs = True,
        ):
        
    feature_to_best_models = {}

    for feature, feature_lbl in zip(feature_list, feature_labels
                                                    ):
        
        single_feature_list = [feature]
        single_feature_labels = [feature_lbl]
        best_model_score_stim, t_to_best_model_using_min_idx_score, t_to_all_data, t_to_all_feature_names = get_plot_multi_stim_model(
                        data_prep_fnc,
                        model_fit_fnc, 
                        result_visualization_fnc, 
                        single_feature_list, 
                        single_feature_labels,
                        z, 
                        data_onsets,
                        exp_onsets, 
                        exp_rev, 
                        durations,
                        fps,
                        score_fnc, 

                        title_label = f"{feature_lbl}", 
                        test_size =test_size, 
                        n_splits = n_splits, 
                        lstm_lag = lstm_lag, 
                        
                        all_timesteps = all_timesteps, 
                        check_warnings = check_warnings, 
                        score_label = score_label, 
                        visualize_individ_stim = visualize_individ_stim, get_coeffs = get_coeffs
        )
        feature_to_best_models[feature_lbl] = [best_model_score_stim, t_to_best_model_using_min_idx_score, t_to_all_data, t_to_all_feature_names]
        
    return feature_to_best_models
def get_plot_multi_stim_model(
                        data_prep_fnc,
                        model_fit_fnc, 
                        result_visualization_fnc, 
                        feature_list, 
                        feature_labels,
                        z, 
                        data_onsets,
                        exp_onsets, 
                        exp_rev, 
                        durations,
                        fps,
                        score_fnc, 
                        # title_label,
                        test_size = 0.3, 
                        n_splits = 10, 
                        lstm_lag = 0, 
                        
                        title_label = "", 
                        all_timesteps = None, 
                        check_warnings = True, 
                        visualize_individ_stim = False, 
                        score_label = "", get_coeffs = True
                        
        ):
        
        with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                if all_timesteps is None: 
                        all_timesteps = np.arange(1, 15, 3)

                        
                results_dict_score_overall = {}
                best_models_score_overall ={}
                results_dict_score_stim ={}
                best_models_score_stim = {}

                
                t_to_all_data = {}
                t_to_all_feature_names ={}
                t_to_all_models = {}
                t_to_all_stim_scores ={}
                t_to_best_model_using_min_idx_score ={}
                
                # n_timesteps_list = []

                t_to_best_model_idx_score = {}
                        

                fig, ax = plt.subplots()
                ax.set_title(f"{title_label}")
                best_models = {} 
                all_scores_per_i = {}
                results_dict = {}
                for n_timesteps in all_timesteps:
                        
                        all_data = []
                        n_stim = len(data_onsets)
                        # stim_i_to_score_means = {i:[] for i in range(n_stim)}
                        # stim_i_to_score_stds = {i:[] for i in range(n_stim)}
                        
                        stim_i_to_Xs = {}
                        stim_i_to_Ys = {}
                
                        for i, (data_onset, exp_onset) in enumerate(zip(data_onsets, exp_onsets)):
                                onset = data_onset-1
                                score_means = []
                                score_stds = []
                                # n_timesteps_list = []
                                Y_latency = latency_to_reversal(exp_rev, exp_onset, max_latency = 6*durations[i]+1)/6 # nan where not 

                                X_stim_all, feature_names_all = prep_FB_inputs_select_features_only(
                                feature_list, feature_labels, onset,  n_timesteps, z, lstm_lag = lstm_lag, fps = fps
                                                )
                                # print("feature_labels", feature_labels)
                                # print(len(feature_list))
                                # print(len(feature_list))
                                X,Y = data_prep_fnc(X_stim_all, Y_latency)
                                
                                stim_i_to_Xs[i] = X
                                stim_i_to_Ys[i] = Y
                                print("i, n_timesteps",i,  n_timesteps)
                                # print("i, X.shape", i, X.shape)
                                scores = []
                                models = []

                        all_stim_X_stim_features, all_stim_Y, feature_names_all = combine_FB_features_across_stim(stim_i_to_Xs, stim_i_to_Ys,  feature_names_all)
                        t_to_all_feature_names[n_timesteps] = feature_names_all
                        overall_scores = []
                        all_stim_scores =[]# np.zeros((n_splits,n_stim))
                        best_model_idx = []
                        models = [] # n_splits
                        for split_idx in range(n_splits):
                                X_train, X_test, y_train, y_test = train_test_split(all_stim_X_stim_features, all_stim_Y,  test_size=test_size, random_state=split_idx)
                                all_data.append((X_train, X_test, y_train, y_test))
                                
                                #get best model over all stim 
                                best_model, y_pred, score = model_fit_fnc(X_train, X_test, y_train, y_test)
                                scores.append(score)
                                models.append(best_model)#,X_train, X_test, y_train, y_test))

                                scores_mean = np.mean(scores)
                                scores_std = np.std(scores)
                                best_model_idx = np.argmax(scores)
                                best_model_overall = models[best_model_idx]
                                
                                
                                #get stim_i for each row in test data
                                stim_bin_test = X_test[:, -1*n_stim:]
                                stim_i_test = np.argmax(stim_bin_test, axis = 1)
                                stim_scores = [] #model specfic for each stim 
                                for j in range(n_stim):
                                        
                                        X_test_stim = X_test[stim_i_test == j, :]
                                        y_test_stim = y_test[stim_i_test == j]
                                        
                                        y_pred = best_model.predict(X_test_stim)
                                        stim_score = score_fnc(y_test_stim, y_pred)  
                                        stim_scores.append(stim_score)
                                all_stim_scores.append(stim_scores)
                                
                                y_pred = best_model.predict(X_test)
                                overall_score = score_fnc(y_test, y_pred) 
                                
                                overall_scores.append(overall_score)
                                models.append(best_model)
                                

                                # Save results
                                # n_timesteps_list.append(n_timesteps)
                                # all_r2_means.append(r2s)
                                all_scores_per_i[(i, n_timesteps)] = scores
                                score_means.append(scores_mean)
                                score_stds.append(scores_std)
                                results_dict[(i, n_timesteps)] = (scores_mean, scores_std)
                                best_models[(i, n_timesteps)] = best_model_overall
                        
                        all_stim_scores = np.array(all_stim_scores)#n_splits,n_stim
                        t_to_all_models[ n_timesteps] = models
                        
                        
                        ###overall scores
                        accuracy_mean_overall = np.mean(overall_scores)
                        accuracy_std_overall= np.std(overall_scores)
                        
                        best_model_idx_score_overall = np.argmax(overall_scores)
                        # np.argmin(overall_scores)
                        best_model_score_overall = models[best_model_idx_score_overall]
                        t_to_best_model_idx_score[n_timesteps] = best_model_idx_score_overall
                        
                        results_dict_score_overall[n_timesteps] = (accuracy_mean_overall, accuracy_std_overall)
                        best_models_score_overall[ n_timesteps] = best_model_score_overall
                        
                        #####stim scores
                        accuracy_mean_stim_specific = np.mean(all_stim_scores, axis =0)
                        accuracy_std_stim_specific = np.std(all_stim_scores, axis =0)

                        
                        best_model_idx_score_stim_min = np.argmax(np.min(all_stim_scores, axis = 1)) # has lowest min on all stim 
                        best_model_score_stim_min = models[best_model_idx_score_stim_min]
                        t_to_best_model_using_min_idx_score[n_timesteps] = best_model_idx_score_stim_min
                        

                        results_dict_score_stim[n_timesteps] = (accuracy_mean_stim_specific, accuracy_std_stim_specific)
                        best_models_score_stim[ n_timesteps] = best_model_score_stim_min
                        
                                        
                        t_to_all_stim_scores[n_timesteps] = all_stim_scores
                        t_to_all_data[n_timesteps] = all_data
                        

                        # n_timesteps_list.append(n_timesteps)
                ####plot overall mean and std overall and for each stim  
                fig, ax = plt.subplots()
                accuracy_mean_overall_over_t = []
                accuracy_std_overall_over_t = []
                stim_acc_mean_over_t = []
                stim_acc_std_over_t = []
                for n_timesteps in all_timesteps:

                        accuracy_mean_overall, accuracy_std_overall = results_dict_score_overall[n_timesteps]
                        accuracy_mean_overall_over_t.append(accuracy_mean_overall)
                        accuracy_std_overall_over_t.append(accuracy_std_overall)
                        accuracy_mean_stim_specific, accuracy_std_stim_specific = results_dict_score_stim[n_timesteps]
                        stim_acc_mean_over_t.append(accuracy_mean_stim_specific)
                        stim_acc_std_over_t.append(accuracy_std_stim_specific)
                # print(n_timesteps_list)/
                stim_acc_mean_over_t = np.array(stim_acc_mean_over_t)
                stim_acc_std_over_t = np.array(stim_acc_std_over_t)
                for j in range(n_stim):
                        ax.errorbar(np.array(all_timesteps)/fps, stim_acc_mean_over_t[:,j], yerr=stim_acc_std_over_t[:,j], fmt='-o', capsize=5, label = f"stim {j}")#color=cmap1(i),
                        
                ax.errorbar(np.array(all_timesteps)/fps, accuracy_mean_overall_over_t, yerr=accuracy_std_overall_over_t, #color=cmap1(i),
                                fmt='-o', capsize=5, label = f"overall")
                ax.set_title(f'mean {score_label} performance over models for given stim ')
                ax.set_ylabel(f" {score_label} ")
                ax.set_xlabel("max time delay (s)")
                ax.legend()
                
                
                ####plot score overall score on each stim (each line a model)
                for n_timesteps in all_timesteps:
                        all_stim_scores = t_to_all_stim_scores[n_timesteps]
                        
                        fig, ax = plt.subplots()
                        for k in range(n_splits): 
                                ax.plot(np.arange(n_stim), all_stim_scores[k])
                        ax.set_title(f" {score_label}  across all models; timestep{n_timesteps}")
                        ax.set_xlabel("stim number")
                        ax.set_ylabel(f"{score_label}")
                        idx = t_to_best_model_idx_score[n_timesteps]
                        ax.plot(np.arange(n_stim), all_stim_scores[idx], linewidth = 3, label = "best")
                        
                        idx = t_to_best_model_using_min_idx_score[n_timesteps]
                        ax.plot(np.arange(n_stim), all_stim_scores[idx], linewidth = 3, label = "best using min")
                        X_train, X_test, y_train, y_test = t_to_all_data[n_timesteps][idx]
                        # best_model,X_train, X_test, y_train, y_test = best_models_score_stim[n_timesteps] 
                        result_visualization_fnc(best_models_score_stim[n_timesteps] , f"{title_label}", t_to_all_feature_names[n_timesteps], X_train, X_test, y_train, y_test, n_timesteps,  feature_names_ordered = False, get_coeffs = get_coeffs)#, xlim = [-2,1])
                        stim_bin_test = X_test[:, -1*n_stim:]
                        stim_i_test = np.argmax(stim_bin_test, axis = 1)
                        
                        stim_bin_train = X_train[:, -1*n_stim:]
                        stim_i_train = np.argmax(stim_bin_train, axis = 1)
                        print(stim_i_train.shape)
                        # print()
                        # print("stim_i_test", stim_i_test)
                        if visualize_individ_stim:
                                for j in range(n_stim):
                                                
                                        X_test_stim = X_test[stim_i_test == j, :]
                                        y_test_stim = y_test[stim_i_test == j]
                                        X_train_stim = X_train[stim_i_train == j, :]
                                        y_train_stim = y_train[stim_i_train == j]
                                        
                                        result_visualization_fnc(best_models_score_stim[n_timesteps] , f"{title_label}; stim: {j}", t_to_all_feature_names[n_timesteps], X_train_stim, X_test_stim, y_train_stim, y_test_stim, n_timesteps,  feature_names_ordered = False, get_coeffs = False)#, xlim = [-2,1])
                                        
                        
        return best_models_score_stim, t_to_best_model_using_min_idx_score, t_to_all_data, t_to_all_feature_names