import copy
import socket
import os
import pickle 
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')

# from sklearn import svm

import pandas as pd
from scipy import interpolate

import scipy.stats as stats
from scipy.stats import mode
import scipy
from load_data_fncs import smooth_trace

from plot_donut_vs_phase import create_X_sequences




def match_cdf(source_data, target_data):
    source_data = np.asarray(source_data)
    target_data = np.asarray(target_data)

    sorted_source = np.sort(source_data)
    sorted_target = np.sort(target_data)

    source_cdf = np.linspace(0, 1, len(sorted_source))
    target_cdf = np.linspace(0, 1, len(sorted_target))
    source_to_cdf = interpolate.interp1d(sorted_source, source_cdf, bounds_error=False, fill_value=(0, 1))
    cdf_to_target = interpolate.interp1d(target_cdf, sorted_target, bounds_error=False, fill_value=(sorted_target[0], sorted_target[-1]))
    source_data_cdf = source_to_cdf(source_data)
    matched_data = cdf_to_target(source_data_cdf)

    return matched_data

def load_data(experiment, condition, neuron, path = None):
    print("in load data")
    if experiment != "hiscl":
        data3 =  load_data_helper(experiment, condition, neuron, path= path)
    else: #stack together the data from diff strains
        if neuron == "AIB":
            return load_data_helper(experiment, condition, neuron, hiscl_neuron="RIM", path= path)
        elif neuron == "AVA":
            data1 = load_data_helper(experiment, condition, neuron, hiscl_neuron="RIM", path= path)
            data2 = load_data_helper(experiment, condition, neuron, hiscl_neuron="AIB", path= path)
        elif neuron == "RIM":
            data1 = load_data_helper(experiment, condition, neuron, hiscl_neuron="AVA", path= path)
            data2 = load_data_helper(experiment, condition, neuron, hiscl_neuron="AIB", path= path)

        data3 = []
        for i in range(len(data1)):
            data3.append(np.vstack((data1[i], data2[i])))

        if neuron == "AVA":
            for i in range(len(data3)):
                data3[i]= np.delete(data3[i], np.arange(1042,1137), axis=0) #???
    exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag = data3
    exp_curve = copy.deepcopy(exp_curve)
    exp_curve[exp_curve == 0] = np.nan
    # exp_shape = exp_vel.shape
    # data3 = [exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag]
    # exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag = data3
    # if replace_outliers:
    #     print("replacing outliers")
    #     exp_acc = replace_outliers_neg_post_seperate(exp_acc.flatten(), threshold = 5).reshape(exp_shape)
    #     exp_curve = replace_outliers_neg_post_seperate(exp_curve.flatten()).reshape(exp_shape)
    #     exp_vel = replace_outliers_neg_post_seperate(exp_vel.flatten(), threshold = 2).reshape(exp_shape)
    #     if body_angle_mag is not None:
    #         body_angle_mag = replace_outliers_neg_post_seperate(body_angle_mag.flatten(), threshold = 2).reshape(exp_shape)
    return exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag

def replace_outliers_exp(data3, exp_shape):
    
    print("in replace_outliers_exp")
    exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag = data3
    exp_curve = copy.deepcopy(exp_curve)
    exp_curve[exp_curve == 0] = np.nan
    # print("replacing outliers")
    exp_acc = replace_outliers_neg_post_seperate(exp_acc.flatten(), threshold = 5).reshape(exp_shape)
    exp_curve = replace_outliers_neg_post_seperate(exp_curve.flatten()).reshape(exp_shape)
    exp_vel = replace_outliers_neg_post_seperate(exp_vel.flatten(), threshold = 2).reshape(exp_shape)
    if body_angle_mag is not None:
        body_angle_mag = replace_outliers_neg_post_seperate(body_angle_mag.flatten(), threshold = 2).reshape(exp_shape)
    return exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag

def load_data_helper(experiment, condition, neuron, hiscl_neuron="", path = None, prefix = ""):
    
    if neuron == "RIM":
        neuron_name = "RIMpReaChR"
        prefix=""
    elif neuron == "AIB":
        neuron_name = "AIBpChrimson"
        prefix= "_AIBC-3"
    elif neuron == "AVA":
        neuron_name = "AVApChrimson"
        prefix=""

    if experiment != "hiscl":
        prefix += f"_Aim1_{neuron}_{experiment}_{neuron_name}_"
    else:
        if neuron == "RIM":
            neuron_n = "RIMR"
            if hiscl_neuron == "AIB":
                prefix += f"_{neuron_n}x{hiscl_neuron}_Hiscl_60s_exp_{neuron_name}_{hiscl_neuron}_788_HisCl_"
            else:
                prefix = f"_{neuron_n}x{hiscl_neuron}_Hiscl_60s_exp_{neuron_n}_{hiscl_neuron}HisCl_"
        elif neuron == "AVA":
            if hiscl_neuron == "RIM":
                prefix = "031224_RIM_Hiscl_60s_exp__ZM10206_RIMHisCl_806_"
            else:
                prefix = "022324_AIB_Hiscl_60s_exp__ZM10206_AIBHisCl_788_"
        elif neuron == "AIB":
            if hiscl_neuron == "RIM":
                prefix = "_AIBC-3xRIM_Hiscl_60s_exp_AIBC-3_RIMHisCl_"
    

    if path is None: 
        path = f"exp_data/{experiment}/{neuron}/{condition}/"


    if experiment == "hiscl":
        path = f"exp_data/{experiment}/{neuron}/{hiscl_neuron}/{condition}/"

    if condition == "control":

        prefix += "atr0his0_"

        exp_vel = pd.read_csv(path+f"{prefix}speed_reshaped.csv", header=None)
        exp_acc = pd.read_csv(path+f"{prefix}acceleration_reshaped.csv", header=None)
        exp_curve = pd.read_csv(path+f"{prefix}body_angle_mag_reshaped.csv", header=None)
        exp_behaviors = pd.read_csv(path+f"{prefix}rev_reshaped.csv", header=None)
        pauses = pd.read_csv(path+f"{prefix}pause_reshaped.csv", header=None)
        turns = pd.read_csv(path+f"{prefix}turn_reshaped.csv", header=None)
        times = pd.read_csv(path+f"{prefix}frame_time_reshaped.csv", header=None)
        try: 
            # midlines = load_from_csv(path+f"{prefix}dv_oriented_midlines_reshaped.csv")
            # n_frames = turns.shape[0]
            # n_tracks = turns.shape[1]
            # n_pts = 100
            # midlines = load_from_csv_midline(path+f"{prefix}dv_oriented_midlines_reshaped.csv",n_frames, n_tracks, n_pts )
            # body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = 80/140)
            # print(path+f"{prefix}__0.5714285714285714_body_angle_mag_from_midlines.csv")
            print(glob.glob(path+f"{prefix}_*body_angle_mag_from_midlines.csv"))
            body_angle_mag = pd.read_csv(path+f"{prefix}_*body_angle_mag_from_midlines.csv", header=None)
        except: 
            body_angle_mag = None
            # "_Aim1_RIM_duty_cycle_RIMpReaChR_atr0his0__0.5714285714285714_body_angle_mag_from_midlines.csv"
        
        
        thresh_time = 667000
        # exp_vel.values[times.values > thresh_time*6] = np.nan
        # exp_acc.values[times.values > thresh_time*6] = np.nan
        # exp_curve.values[times.values > thresh_time*6] = np.nan
        # exp_behaviors.values[times.values > thresh_time*6] = np.nan
        # pauses.values[times.values>thresh_time*6] = np.nan
        # turns.values[times.values>thresh_time*6] = np.nan

        cutoff= 250
        # exp_vel = exp_vel.dropna(axis=1, how="any").values[:,:cutoff].T
        # exp_acc = exp_acc.dropna(axis=1, how="any").values[:,:cutoff].T
        # exp_curve = exp_curve.dropna(axis=1, how="any").values[:,:cutoff].T
        # exp_behaviors = exp_behaviors.dropna(axis=1, how="any").values[:,:cutoff].T
        # pauses = pauses.dropna(axis=1, how="any").values[:,:cutoff].T
        # turns = turns.dropna(axis=1, how="any").values[:,:cutoff].T
        
        
        # Step 1: Stack all DataFrames into a list
        dfs = [exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns]

        # Step 2: Get a boolean mask of valid (non-NaN) columns for each DataFrame
        valid_masks = [~df.isna().any(axis=0) for df in dfs]

        # Step 3: Combine masks to find common valid columns (logical AND across all DataFrames)
        common_valid_mask = valid_masks[0].copy()
        for mask in valid_masks[1:]:
            common_valid_mask &= mask

        # Step 4: Use the mask to filter columns, then apply cutoff and transpose
        exp_vel = exp_vel.loc[:, common_valid_mask].values[:, :cutoff].T
        exp_acc = exp_acc.loc[:, common_valid_mask].values[:, :cutoff].T
        exp_curve = exp_curve.loc[:, common_valid_mask].values[:, :cutoff].T
        exp_behaviors = exp_behaviors.loc[:, common_valid_mask].values[:, :cutoff].T
        pauses = pauses.loc[:, common_valid_mask].values[:, :cutoff].T
        turns = turns.loc[:, common_valid_mask].values[:, :cutoff].T
        
        if body_angle_mag is not None: 
            # body_angle_mag  = body_angle_mag.dropna(axis=1, how="any").values[:,:cutoff].T
            body_angle_mag  = body_angle_mag.loc[:, common_valid_mask].values[:, :cutoff].T
            print("using body angle mag for curvature")
        else: 
            print("NOT using body angle mag for curvature")

    else:
        if neuron == "AIB":
            cutoff = 10500
        else:
            cutoff = 195000

        if experiment == "time_titration":
            cutoff = 91*15

        assert np.floor(cutoff * 1.6/6) == cutoff * 1.6/6, "make sure cutoff is divisible by 15 bc itll throw an error based on timestep resampling"

        prefix += f"atr1his0_"

        exp_vel = pd.read_csv(path+f"{prefix}speed_reshaped.csv", header=None)[:cutoff]
        exp_acc = pd.read_csv(path+f"{prefix}acceleration_reshaped.csv", header=None)[:cutoff]
        exp_curve = pd.read_csv(path+f"{prefix}body_angle_mag_reshaped.csv", header=None)[:cutoff]
        exp_behaviors = pd.read_csv(path+f"{prefix}rev_reshaped.csv", header=None)[:cutoff]
        pauses = pd.read_csv(path+f"{prefix}pause_reshaped.csv", header=None)[:cutoff]
        turns = pd.read_csv(path+f"{prefix}turn_reshaped.csv", header=None)[:cutoff]
        n_frames = turns.shape[0]
        n_tracks = turns.shape[1]
        n_pts = 100
        try: 
            # midlines = load_from_csv_midline(path+f"{prefix}dv_oriented_midlines_reshaped.csv",n_frames, n_tracks, n_pts )[:cutoff]
            # body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = 80/140)
            # body_angle_mag = pd.read_csv(path+f"{prefix}__0.5714285714285714_body_angle_mag_from_midlines.csv", header=None)[:cutoff]
            print(glob.glob(path+f"{prefix}_*body_angle_mag_from_midlines.csv"))
            body_angle_mag = pd.read_csv(path+f"{prefix}_*body_angle_mag_from_midlines.csv", header=None)[:cutoff]
            print("using body angle mag for curvature")
        except: 
            body_angle_mag = None
            print("NOT using body angle mag for curvature")

        

        # remove nans
        vel_nans = np.argwhere(exp_vel.isna().values.astype(int).sum(axis=0) >0)[:,0]
        vel_nans = [an for an in vel_nans]
        acc_nans = np.argwhere(exp_acc.isna().values.astype(int).sum(axis=0) >0)[:,0]
        acc_nans = [an for an in acc_nans]
        curve_nans = np.argwhere(exp_curve.isna().values.astype(int).sum(axis=0) >0)[:,0]
        curve_nans = [an for an in curve_nans]
        rev_nans = np.argwhere(exp_behaviors.isna().values.astype(int).sum(axis=0) >0)[:,0]
        rev_nans = [an for an in rev_nans]
        pause_nans = np.argwhere(pauses.isna().values.astype(int).sum(axis=0) >0)[:,0]
        pause_nans = [an for an in pause_nans]
        turn_nans = np.argwhere(turns.isna().values.astype(int).sum(axis=0) >0)[:,0]
        turn_nans = [an for an in turn_nans]


        all_nans = set()
        all_nans = all_nans.union(vel_nans)
        all_nans = all_nans.union(acc_nans)
        all_nans = all_nans.union(curve_nans)
        all_nans = all_nans.union(rev_nans)
        all_nans = all_nans.union(pause_nans)
        all_nans = all_nans.union(turn_nans)

        all_nans = np.array(list(all_nans))

        exp_vel = exp_vel.drop(columns=all_nans).values.T
        exp_acc = exp_acc.drop(columns=all_nans).values.T
        exp_curve = exp_curve.drop(columns=all_nans).values.T
        exp_behaviors = exp_behaviors.drop(columns=all_nans).values.T
        pauses = pauses.drop(columns=all_nans).values.T
        turns = turns.drop(columns=all_nans).values.T
        if body_angle_mag is not None: 
            body_angle_mag = body_angle_mag.drop(columns=all_nans).values.T
        # midlines = np.delete(midlines, all_nans, axis=1)
        

    return exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag
        


def plot_data(exp_vel, exp_acc, exp_curve, exp_behaviors, turns, experiment, fps=6, orig=True):
    print("in plot data")
    fig,ax = plt.subplots(2,5, figsize=(18,6))
    if orig:
        im = ax[0,0].imshow(exp_vel, aspect="auto", interpolation="None", cmap="coolwarm", vmin=-500, vmax=500)
        #fig.colorbar(im)
        im = ax[0,1].imshow(exp_acc, aspect="auto", interpolation="None", cmap="coolwarm", vmin=-200, vmax=200)
    else:
        im = ax[0,0].imshow(exp_vel, aspect="auto", interpolation="None", cmap="coolwarm", vmin=-5, vmax=5)
        #fig.colorbar(im)
        im = ax[0,1].imshow(exp_acc, aspect="auto", interpolation="None", cmap="coolwarm", vmin=-5, vmax=5)
    #fig.colorbar(im)
    im = ax[0,2].imshow(exp_curve, aspect="auto", interpolation="None")
    #fig.colorbar(im)
    ax[0,3].imshow(exp_behaviors, aspect="auto", interpolation="None")
    #fig.colorbar(im)
    ax[0,4].imshow(turns, aspect="auto", interpolation="None")



    ax[1,0].plot(exp_vel.mean(axis=0))
    ax[1,0].set_xlim(0,exp_behaviors.shape[1])
    if orig:
        ax[1,0].set_ylim(-500,500)
        ax[1,1].set_ylim(-200,200)
        ax[1,2].set_ylim(0,30)
    else:
        ax[1,0].set_ylim(-5,5)
        ax[1,1].set_ylim(-5,5)
        ax[1,2].set_ylim(-5,5)



    ax[1,1].plot(exp_acc.mean(axis=0))
    ax[1,1].set_xlim(0,exp_behaviors.shape[1])

    ax[1,2].plot(exp_curve.mean(axis=0))
    ax[1,2].set_xlim(0,exp_behaviors.shape[1])

    ax[1,3].plot(exp_behaviors.mean(axis=0))
    ax[1,3].set_xlim(0,exp_behaviors.shape[1])
    ax[1,3].set_ylim(0,1)

    ax[1,4].plot(turns.mean(axis=0))
    ax[1,4].set_xlim(0,exp_behaviors.shape[1])
    ax[1,4].set_ylim(0,1)

    off= 105*fps

    if experiment == "duty_cycle":
        onsets = np.array([120*fps-off, 155*fps-off, 190*fps-off, 225*fps - off, 260*fps-off, 295*fps-off])
        offsets = onsets + 6*fps
    elif experiment == "time_titration":
        onsets = np.array([120*fps-off, 151*fps-off, 184*fps-off, 219*fps-off])
        offsets = onsets + np.array([1*fps, 3*fps, 5*fps, 10*fps])
    elif experiment == "hiscl":
        onsets = np.array([120*fps-off, 185*fps-off])
        offsets = onsets + np.array([5*fps, 5*fps])


    for i in range(2):
        for j in range(5):
            ax[0,j].set_xlim(0, exp_behaviors.shape[1])

            for stim_number in range(len(onsets)):
                ax[i,j].axvline(onsets[stim_number], color='k', lw=0.1*(1+stim_number%5))
                ax[i,j].axvline(offsets[stim_number], color='k', lw=0.1*(1+stim_number%5))


    ax[0,0].set_title("velocity")
    ax[0,1].set_title("acceleration")
    ax[0,2].set_title("curvature")
    ax[0,3].set_title("reversals")
    ax[0,4].set_title("turns")


    fig.tight_layout()

    return onsets.astype(int)

def replace_outliers_neg_post_seperate(values, threshold=9):
        
    # Compute Z-scores
    neg_is = np.argwhere(values<0).flatten()
    z_scores_neg = np.abs(stats.zscore(values[neg_is], nan_policy='omit'))
    outliers_neg = z_scores_neg > threshold
     
    pos_is = np.argwhere(values>=0).flatten()
    z_scores_pos = np.abs(stats.zscore(values[pos_is], nan_policy='omit'))
    outliers_pos = z_scores_pos > threshold
    
    # Identify outliers
   
    outliers =  np.isnan(values)
    outliers[neg_is] = np.logical_or(outliers_neg, outliers[neg_is])
    outliers[pos_is]= np.logical_or(outliers_pos, outliers[pos_is])
    
    # Interpolate over outliers
    indices = np.arange(len(values))
    clean_values = values.copy()
    clean_values[outliers] = np.interp(indices[outliers], indices[~outliers], values[~outliers])
    
    return clean_values
def resample_values(values, original_fps=6, target_fps=1.6):
    # print("values.shape", values.shape)
    clean_values = values
    # print("lean_values.shape", clean_values.shape)
    indices = np.arange(len(values))
        # Resample using linear interpolation
        
    new_indices = np.linspace(0, len(clean_values) - 1, int(len(clean_values) * (target_fps / original_fps)))
    
    # print("new_indices.shape", new_indices.shape)
    # print("indices.shape", indices.shape)
    resampled_values = np.interp(new_indices, indices, clean_values)   

    return resampled_values

def replace_outliers_neg_post_seperate_and_resample(values, threshold=9,  original_fps=6, target_fps=1.6):
    # print("values.shape", values.shape)
    clean_values = replace_outliers_neg_post_seperate(values, threshold=threshold)     
    resampled_values = resample_values(clean_values, original_fps=original_fps, target_fps=target_fps)
    # print("lean_values.shape", clean_values.shape)
    # indices = np.arange(len(values))
    #     # Resample using linear interpolation
        
    # new_indices = np.linspace(0, len(clean_values) - 1, int(len(clean_values) * (target_fps / original_fps)))
    
    # # print("new_indices.shape", new_indices.shape)
    # # print("indices.shape", indices.shape)
    # resampled_values = np.interp(new_indices, indices, clean_values)   

    return resampled_values

def replace_outliers_and_resample(values, threshold=9, original_fps=6, target_fps=1.6):
    values = np.array(values)
    
    # Compute Z-scores
    z_scores = np.abs(stats.zscore(values, nan_policy='omit'))
    
    # Identify outliers
    outliers = z_scores > threshold
    outliers = np.logical_or(z_scores > threshold, np.isnan(values))
    
    # Interpolate over outliers
    indices = np.arange(len(values))
    clean_values = values.copy()
    clean_values[outliers] = np.interp(indices[outliers], indices[~outliers], values[~outliers])
    
    # Resample using linear interpolation
    new_indices = np.linspace(0, len(clean_values) - 1, int(len(clean_values) * (target_fps / original_fps)))
    resampled_values = np.interp(new_indices, indices, clean_values)
    
    return resampled_values

def resample_binary(values, original_fps=6, target_fps=1.6):
    values = np.array(values)
    indices = np.arange(len(values))
    new_indices = np.linspace(0, len(values) - 1, int(len(values) * (target_fps / original_fps)))
    
    # Linear interpolation
    interpolated_values = np.interp(new_indices, indices, values)
    
    # Thresholding: Convert values to binary (0 or 1)
    resampled_values = (interpolated_values >= 0.5).astype(int)
    
    return resampled_values

def causal_moving_average(x, window_size):
    result = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        start = max(0, i - window_size + 1)
        result[i] = np.mean(x[start:i+1])
    return result

def reformat_data(exp_vel, exp_acc, exp_curve, exp_behaviors, turns, pauses, flavell_beh, flavell_velocity, flavell_acceleration, flavell_worm_curvature,
                  match_vel_only = False, 
                  palette =None, plot=False,replace_outliers =False,  sep_pos_neg_resample = False):
    if palette is None:
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            
            "purple","red", "yellow", "black", "pink"]  
    print("in reformar data")
        
    if plot:
        fig,axs = plt.subplots(4, 3, figsize=(18,8))
        axs[0,0].set_ylabel("Steve's data")
        axs[1,0].set_ylabel("FB data (raw)")
        axs[2,0].set_ylabel("FB (interpolated)")
        axs[3,0].set_ylabel("FB (scaled n centered)")

        bins = [np.linspace(-6, 6, 1000), np.linspace(-5,5, 1000), np.linspace(-5, 5, 1000)]

        # variables = ["velocity", "acceleration", "total curvature"]
        # for i in range(3):
        # #     quantity= X_all[:,0,i]
        # #     axs[0,i].hist(quantity, bins=bins[i])
        # #     axs[0,i].set_title(variables[i])
        #     axs[0,0].hist(flavell_velocity, bins=bins[i])
        #     axs[0,0].set_title("velocity")
            
        #     axs[0,1].hist(flavell_acceleration, bins=bins[i])
        #     axs[0,1].set_title("accel")
            
        #     axs[0,2].hist(flavell_worm_curvature, bins=bins[i])
        #     axs[0,2].set_title("curvature")


    all_vel = exp_vel.flatten()/1000
    all_acc = exp_acc.flatten()/1000
    all_curve = exp_curve.flatten()
    all_rev = exp_behaviors.flatten()
    all_pause = pauses.flatten()
    all_turn = turns.flatten()

    if plot:
        axs[1,0].hist(all_vel[all_pause == 1], bins=1000, density=True, color="purple", label="pause")
        axs[1,0].hist(all_vel[all_rev == 1], bins=1000, density=True, color=palette[1], label="rev")
        axs[1,0].hist(all_vel[(all_rev == 0)*(all_pause==0)], bins=1000, density=True, color=palette[0], label="fwd + turn")
        axs[1,0].hist(all_vel, bins=100, density=True, color='k', label="all")
        axs[1,0].legend()


        axs[1,1].hist(all_acc, bins=1000, density=True)
        axs[1,2].hist(all_curve, bins=1000, density=True)
    print("plotted")

    print("exp_acc1", all_acc.shape)
    if replace_outliers: 
        if sep_pos_neg_resample: 
            print("resamplsep_pos_neg_resampleing")
            exp_acc1 = replace_outliers_neg_post_seperate_and_resample(all_acc,  threshold = 5)
            print("exp_acc1.shape)", exp_acc1.shape)
            resampled_acc = exp_acc1
            
            exp_curve1 = replace_outliers_neg_post_seperate_and_resample(all_curve)
            resampled_curve = exp_curve1
            
            exp_vel1 = replace_outliers_neg_post_seperate_and_resample(all_vel, threshold = 2)
            resampled_vel = exp_vel1
            
            print()
        else: 
            resampled_vel = replace_outliers_and_resample(all_vel)
            resampled_acc = replace_outliers_and_resample(all_acc)
            resampled_curve = replace_outliers_and_resample(all_curve)
    
    
    print("resampling")
    resampled_rev = resample_binary(all_rev)
    resampled_pause = resample_binary(all_pause)
    print("resampled_pause", resampled_pause.shape)
    resampled_turn = resample_binary(all_turn)

    resampled_vel = resampled_vel.reshape((exp_vel.shape[0],-1))
    resampled_acc = resampled_acc.reshape((exp_vel.shape[0],-1))
    resampled_curve = resampled_curve.reshape((exp_vel.shape[0],-1))

    resampled_vel = smooth_trace(resampled_vel, 0.5).flatten() 
    resampled_acc = smooth_trace(resampled_acc, 1.8).flatten()
    resampled_curve = smooth_trace(resampled_curve, 1).flatten()

    
    if plot:
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
        #    "purple","red", "yellow", "black", "pink"
            ]    # pause
        axs[2,0].hist(resampled_vel[resampled_pause == 1], bins=100, density=True, color="purple", label="pause")
        axs[2,0].hist(resampled_vel[resampled_rev == 1], bins=100, density=True, color=palette[1], label="rev")
        axs[2,0].hist(resampled_vel[(resampled_rev == 0)*(resampled_pause==0)], bins=100, density=True, color=palette[0], label="fwd + turn")
        axs[2,0].hist(resampled_vel, bins=100, density=True, color='k', label="all")
        axs[2,0].legend()

        axs[2,1].hist(resampled_acc, bins=1000)
        axs[2,1].set_xlim(axs[0,1].get_xlim())

        axs[2,2].hist(resampled_curve, bins=1000)
        axs[2,2].set_xlim(axs[0,2].get_xlim())

    print("matching cdfs")

    criteria = (resampled_rev == 0) * (resampled_turn ==0)
    resampled_vel[criteria] = match_cdf(resampled_vel[criteria] , flavell_velocity.flatten()[flavell_beh== 0])

    criteria = (resampled_rev == 1) 
    resampled_vel[criteria] = match_cdf(resampled_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 1])

    criteria = (resampled_turn ==1)
    resampled_vel[criteria] = match_cdf(resampled_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 2])

    
    resampled_acc = match_cdf(resampled_acc, flavell_acceleration.flatten())

    resampled_curve = match_cdf(resampled_curve, flavell_worm_curvature.flatten())


    if plot:
        axs[3,0].hist(resampled_vel[resampled_rev == 1], bins=bins[0], density=True, color=palette[1], label="FB rev only")
        axs[3,0].hist(resampled_vel[(resampled_rev == 0)*(resampled_pause==0)], bins=bins[0], density=True, color=palette[0], label="FB fwd + turn only")
        # axs[3,0].hist(resampled_vel[(resampled_pause==1)], bins=100, density=True, color="purple", label="FB pause only")

        axs[3,0].hist(resampled_vel, bins=bins[0], density=True, label="FB all")


        axs[3,1].hist(resampled_acc, bins=bins[1], density=True, label="FB")
        axs[3,1].set_xlim(axs[0,1].get_xlim())


        axs[3,2].hist(resampled_curve, bins=bins[2], density=True, label="FB")
        axs[3,2].set_xlim(axs[0,2].get_xlim())

        # for i in range(3):
        #     quantity= X_all[:,0,i]
        #     axs[3,i].hist(quantity, bins=bins[i], alpha=0.3, density=True, label="steve")
        #     axs[3,i].legend()

        fig.tight_layout()



    resampled_vel = resampled_vel.reshape((exp_vel.shape[0],-1))
    resampled_acc = resampled_acc.reshape((exp_vel.shape[0],-1))
    resampled_curve = resampled_curve.reshape((exp_vel.shape[0],-1))
    resampled_rev = resampled_rev.reshape((exp_vel.shape[0],-1))
    resampled_turn = resampled_turn.reshape((exp_vel.shape[0],-1))

    # resampled_vel = smooth_trace(resampled_vel, 0.5)
    # resampled_acc = smooth_trace(resampled_acc, 1.8)
    # resampled_curve = smooth_trace(resampled_curve, 1)

    

    return resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn



def reformat_data_1(exp_vel, exp_acc, exp_curve, exp_behaviors, turns, pauses, flavell_beh, flavell_velocity, flavell_acceleration, flavell_worm_curvature, palette =None, plot=True,replace_outliers = True,  sep_pos_neg_resample = True, match_vel_only = True):
    if palette is None:
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            
            "purple","red", "yellow", "black", "pink"]  
    print("in reformar data")
        
    if plot:
        fig,axs = plt.subplots(4, 3, figsize=(18,8))
        axs[0,0].set_ylabel("Steve's data")
        axs[1,0].set_ylabel("FB data (raw)")
        axs[2,0].set_ylabel("FB (interpolated)")
        axs[3,0].set_ylabel("FB (scaled n centered)")

        bins = [np.linspace(-6, 6, 1000), np.linspace(-5,5, 1000), np.linspace(-5, 5, 1000)]

        # variables = ["velocity", "acceleration", "total curvature"]
        print("flavell_velocity.shahpe", flavell_velocity.shape)
        # # for i in range(3):
        # #     quantity= X_all[:,0,i]
        # #     axs[0,i].hist(quantity, bins=bins[i])
        # #     axs[0,i].set_title(variables[i])
        #     # axs[0,0].hist(flavell_velocity, bins=bins[i])
        # step = 10
        # axs[0,0].hist(flavell_velocity[::step], bins=200)
        # axs[0,0].set_title("velocity")
        
        # # axs[0,1].hist(flavell_acceleration, bins=bins[i])
        # axs[0,1].hist(flavell_acceleration[::step], bins=200)
        # axs[0,1].set_title("accel")
        
        # # axs[0,2].hist(flavell_worm_curvature, bins=bins[i])
        # axs[0,2].hist(flavell_worm_curvature[::step], bins=200)
        # axs[0,2].set_title("curvature")


    all_vel = exp_vel.flatten()/1000
    all_acc = exp_acc.flatten()/1000
    all_curve = exp_curve.flatten()
    all_rev = exp_behaviors.flatten()
    all_pause = pauses.flatten()
    all_turn = turns.flatten()

    if plot:
        axs[1,0].hist(all_vel[all_pause == 1], bins=1000, density=True, color="purple", label="pause")
        axs[1,0].hist(all_vel[all_rev == 1], bins=1000, density=True, color=palette[1], label="rev")
        axs[1,0].hist(all_vel[(all_rev == 0)*(all_pause==0)], bins=1000, density=True, color=palette[0], label="fwd + turn")
        axs[1,0].hist(all_vel, bins=100, density=True, color='k', label="all")
        axs[1,0].legend()


        axs[1,1].hist(all_acc, bins=1000, density=True)
        axs[1,2].hist(all_curve, bins=1000, density=True)
    print("plotted")

    print("exp_acc1", all_acc.shape)
    if replace_outliers:
        if sep_pos_neg_resample: 
            print("resamplsep_pos_neg_resampleing")
            exp_acc1 = replace_outliers_neg_post_seperate_and_resample(all_acc,  threshold = 5)
            print("exp_acc1.shape)", exp_acc1.shape)
            resampled_acc = exp_acc1
            
            exp_curve1 = replace_outliers_neg_post_seperate_and_resample(all_curve)
            resampled_curve = exp_curve1
            
            exp_vel1 = replace_outliers_neg_post_seperate_and_resample(all_vel, threshold = 2)
            resampled_vel = exp_vel1
            
            print()
        else: 
            resampled_vel = replace_outliers_and_resample(all_vel)
            resampled_acc = replace_outliers_and_resample(all_acc)
            resampled_curve = replace_outliers_and_resample(all_curve)
    resampled_features = [copy.deepcopy(resampled_vel).reshape((exp_vel.shape[0],-1)), copy.deepcopy(resampled_acc).reshape((exp_vel.shape[0],-1)), copy.deepcopy(resampled_curve).reshape((exp_vel.shape[0],-1))]
    resampled_rev = resample_binary(all_rev)#.reshape((exp_vel.shape[0],-1))
    resampled_pause = resample_binary(all_pause)#.reshape((exp_vel.shape[0],-1))
    print("resampled_pause", resampled_pause.shape)
    resampled_turn = resample_binary(all_turn)#.reshape((exp_vel.shape[0],-1))
    
    if plot:
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
        #    "purple","red", "yellow", "black", "pink"
            ]    # pause
        axs[2,0].hist(resampled_vel[resampled_pause == 1], bins=100, density=True, color="purple", label="pause")
        axs[2,0].hist(resampled_vel[resampled_rev == 1], bins=100, density=True, color=palette[1], label="rev")
        axs[2,0].hist(resampled_vel[(resampled_rev == 0)*(resampled_pause==0)], bins=100, density=True, color=palette[0], label="fwd + turn")
        axs[2,0].hist(resampled_vel, bins=100, density=True, color='k', label="all")
        axs[2,0].legend()

        axs[2,1].hist(resampled_acc, bins=1000)
        axs[2,1].set_xlim(axs[0,1].get_xlim())

        axs[2,2].hist(resampled_curve, bins=1000)
        axs[2,2].set_xlim(axs[0,2].get_xlim())
    print("resampling")


    # smooth_vel = smooth_trace(resampled_vel, 0.5).flatten() 
    # smooth_acc = smooth_trace(resampled_acc, 1.8).flatten()
    # smooth_curve = smooth_trace(resampled_curve, 1).flatten()
    smooth_vel = causal_moving_average(resampled_vel, 3)
    smooth_acc = causal_moving_average(resampled_acc, 3)
    smooth_curve = causal_moving_average(resampled_curve, 3)
    
    smooth_features = [copy.deepcopy(smooth_vel).reshape((exp_vel.shape[0],-1)), copy.deepcopy(smooth_acc).reshape((exp_vel.shape[0],-1)), copy.deepcopy(smooth_curve).reshape((exp_vel.shape[0],-1))]

    


    print("matching cdfs")
    matched_vel = copy.deepcopy(smooth_vel)
    matched_acc = copy.deepcopy(smooth_acc)
    matched_curve = copy.deepcopy(smooth_curve)
    if match_vel_only:

        criteria = (resampled_rev == 0) * (resampled_turn ==0)
        matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh== 0])

        criteria = (resampled_rev == 1) 
        matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 1])

        criteria = (resampled_turn ==1)
        matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 2])

        
        matched_acc = match_cdf(smooth_acc, flavell_acceleration.flatten())

        matched_curve = match_cdf(smooth_curve, flavell_worm_curvature.flatten())
    else: 
        criteria = (resampled_rev == 0) * (resampled_turn ==0)
        matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_acceleration.flatten()[flavell_beh== 0])
        matched_acc[criteria] = match_cdf(smooth_acc[criteria] , flavell_acceleration.flatten()[flavell_beh== 0])
        matched_curve[criteria] = match_cdf(smooth_curve[criteria] , flavell_worm_curvature.flatten()[flavell_beh== 0])
        
        
        criteria = (resampled_rev == 1) 
        matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 1])
        matched_acc[criteria] = match_cdf(smooth_acc[criteria] , flavell_acceleration.flatten()[flavell_beh== 1])
        matched_curve[criteria] = match_cdf(smooth_curve[criteria] , flavell_worm_curvature.flatten()[flavell_beh== 1])

        criteria = (resampled_turn ==1)
        matched_vel[criteria] = match_cdf(smooth_vel[criteria] , flavell_velocity.flatten()[flavell_beh == 2])
        matched_acc[criteria] = match_cdf(smooth_acc[criteria] , flavell_acceleration.flatten()[flavell_beh== 2])
        matched_curve[criteria] = match_cdf(smooth_curve[criteria] , flavell_worm_curvature.flatten()[flavell_beh== 2])
    matched_features = [copy.deepcopy(matched_vel).reshape((exp_vel.shape[0],-1)), copy.deepcopy(matched_acc).reshape((exp_vel.shape[0],-1)), copy.deepcopy(matched_curve).reshape((exp_vel.shape[0],-1))]


    if plot:
        axs[3,0].hist(matched_vel[resampled_rev == 1], bins=bins[0], density=True, color=palette[1], label="FB rev only")
        axs[3,0].hist(matched_vel[(resampled_rev == 0)*(resampled_pause==0)], bins=bins[0], density=True, color=palette[0], label="FB fwd + turn only")
        # axs[3,0].hist(resampled_vel[(resampled_pause==1)], bins=100, density=True, color="purple", label="FB pause only")

        axs[3,0].hist(matched_vel, bins=bins[0], density=True, label="FB all")


        axs[3,1].hist(matched_acc, bins=bins[1], density=True, label="FB")
        axs[3,1].set_xlim(axs[0,1].get_xlim())


        axs[3,2].hist(matched_curve, bins=bins[2], density=True, label="FB")
        axs[3,2].set_xlim(axs[0,2].get_xlim())

        # for i in range(3):
        #     quantity= X_all[:,0,i]
        #     axs[3,i].hist(quantity, bins=bins[i], alpha=0.3, density=True, label="steve")
        #     axs[3,i].legend()

        fig.tight_layout()



    matched_vel = matched_vel.reshape((exp_vel.shape[0],-1))
    matched_acc = matched_acc.reshape((exp_vel.shape[0],-1))
    matched_curve = matched_curve.reshape((exp_vel.shape[0],-1))
    resampled_rev = resampled_rev.reshape((exp_vel.shape[0],-1))
    resampled_turn = resampled_turn.reshape((exp_vel.shape[0],-1))

    # resampled_vel = smooth_trace(resampled_vel, 0.5)
    # resampled_acc = smooth_trace(resampled_acc, 1.8)
    # resampled_curve = smooth_trace(resampled_curve, 1)

    

    return  resampled_rev, resampled_turn,resampled_features,  smooth_features, matched_features

# def create_X_sequences(X, T):
#     X_seq = []
#     for i in range(len(X) - T):
#         X_seq.append(X[i:i+T])  
#     return torch.stack(X_seq)

# def circular_moving_success_stats(theta, values, window_size):

#     theta = np.radians(theta)
#     smoothed_rates = np.zeros_like(values, dtype=float)
#     smoothed_sem = np.zeros_like(values, dtype=float)

#     for i, t in enumerate(theta):
#         # Compute circular distances
#         circular_diff = np.abs(np.angle(np.exp(1j * (theta - t))))  # Circular distance
#         mask = circular_diff <= window_size / 2  # Select points within window

#         if np.any(mask):
#             local_values = values[mask]
#             smoothed_rates[i] = np.nanmean(local_values)  # Compute success rate (fraction of 1s)
#             smoothed_sem[i] = scipy.stats.sem(local_values)
#         else:
#             smoothed_rates[i] = values[i]  # Default to original if no neighbors
#             smoothed_sem[i] = 0  # No variance in a single point

#     return smoothed_rates, smoothed_sem

# from plot_donut_vs_phase import bin_phase_data


# def plot_rev_pro2(phase, beh, z=[], label=None, ax=None, color=None, window_size=50, beh1=True, 
#                   plot_type="box", num_bins=6, vert=True):

#     phase = np.array(phase)
#     if beh1:
#         beh = np.array(beh) == 1  # Convert to boolean

#     idx = np.argsort(phase)
#     sorted_phase = phase[idx]
#     sorted_beh = beh[idx]

#     # Compute smoothed behavioral response
#     sorted_beh_mean, sorted_beh_std = circular_moving_success_stats(sorted_phase, sorted_beh, window_size=window_size)
#     #sorted_beh_std = np.sqrt(sorted_beh_mean*(1-sorted_beh_mean)/len(idx))

#     #sorted_beh_mean = sorted_beh

#     if ax is None:
#         fig, ax = plt.subplots()

#     if plot_type == "line":
#         # Line plot with shaded standard deviation
#         l1 = ax.plot(sorted_phase, sorted_beh_mean, label=label, color=color)[0]
#         ax.fill_between(sorted_phase, sorted_beh_mean - sorted_beh_std , 
#                         sorted_beh_mean + sorted_beh_std , alpha=0.3, color=l1.get_color())

#     elif plot_type == "box":
#         # Bin phase data
#         bin_edges = np.linspace(-180, 180, num_bins + 1)

#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
#         bin_indices = np.digitize(sorted_phase, bin_edges) - 1  # Assign each phase to a bin
#         print("bin_edges",bin_edges)
#         print("bin_centers",bin_centers)
        
#         # Collect data in bins from the smoothed response
#         binned_data = [sorted_beh_mean[bin_indices == i] for i in range(num_bins)]  
        
#         # Remove empty bins
#         valid_bins = [i for i, data in enumerate(binned_data) if len(data) > 0]
#         bin_centers = bin_centers[valid_bins]
#         binned_data = [binned_data[i] for i in valid_bins]
#         print([(sum(~np.isnan(bd))) for bd in binned_data])

#         boot_data = []
#         for bd in binned_data:
#             bin_boot = []
#             for i in range(1000):
#                 n_samples = 4
#                 idx2 = np.arange(len(bd))
#                 np.random.shuffle(idx2)
#                 bin_boot.append(bd[idx2[:n_samples]].mean())
#             boot_data.append(np.array(bin_boot))

#         filtered_data = []
#         filtered_positions = []

#         ### Uncomment if you want to Filter out bins with fewer than X samples
#         # for data, pos in zip(binned_data, bin_centers):
#         #     if len(data) >= 8:
#         #         filtered_data.append(data)
#         #         filtered_positions.append(pos)

#         filtered_data = binned_data
#         filtered_positions=bin_centers

#         # Plot boxplot using the smoothed behavioral response
#         ax.boxplot(filtered_data, positions=filtered_positions, widths=(360 / num_bins) * 0.7, patch_artist=True,
#                    boxprops=dict(facecolor=color, alpha=0.7, edgecolor="k"), showfliers=False,
#                    medianprops=dict(color="black"), manage_ticks=False, vert=vert)
        

#     # Optional: Plot discrete states
#     if np.size(z) > 0:
#         sorted_z = z[idx]
#         _, states, _ = bin_phase_data(sorted_phase, sorted_z)
#         ax.imshow(np.array(states)[np.newaxis, :], extent=[-180, 180, -0.02, 0], cmap=cmap, vmin=0, vmax=6, aspect="auto")

#     if vert: 
#         ax.set_xlabel("Phase")
#         ax.set_ylabel("Rev. prob")
#         ax.set_ylim(-0.02, 1.1)
#         ax.set_xlim(-180, 180)
#     else:
#         ax.set_xlim(-0.02, 1.1)
#         ax.set_ylim(-180, 180)

#     if plot_type == "box":
#         return ax, np.array([bin_i.mean() for bin_i in boot_data])
#     else:
#         return ax



# def phase_response_curve2(slds, init_cons, input, pca, offset=-1, thresh=3, remove_rev = False):
#     T = input.shape[0]
#     init_behs = np.concatenate(np.array(z[0:5])[:,10:]) # assuming init_cons == np.concatenate(np.array(q_x[0:5])[:,10:])

#     phase_shifts = []
#     orig_thetas = []
#     final_thetas = []
#     final_behs = []
#     all_behs = []
#     for i in range(init_cons.shape[0]):
#         prefix = [np.array([0]), np.array([init_cons[i]]), np.array([np.zeros(slds.N)])]
#         theta_orig = compute_phase(init_cons[i], pca)
#         _, x_pert, y_pert = slds.sample(T=T, prefix=prefix, input=input)
#         x_est = y_pert@(np.linalg.pinv(slds.emissions.Cs[0]).T) # get back neural activity ? oh no I think its get latents from neural actiity 
#         if init_behs[i] == 1 and remove_rev:
#             final_behs.append(np.nan)
#         else:
#             final_behs.append((beh_decoder.predict(x_est)[0:] == 1).sum()>=thresh)
#         theta_pert = compute_phase(x_pert[-1], pca)
#         delta_theta = np.unwrap(np.radians(theta_pert - theta_orig))  # Convert to radians and unwrap
#         delta_theta = np.degrees(delta_theta)  # Convert back to degrees
#         delta_theta = (theta_pert - theta_orig + 180) % 360 - 180 # Keep in 0 to 2π
#         phase_shifts.append(delta_theta[0])
#         orig_thetas.append(theta_orig[0])
#         final_thetas.append(theta_pert[0])
#         all_behs.append(beh_decoder.predict(x_est)[0:])


#     return orig_thetas, phase_shifts, final_thetas, final_behs, all_behs



def analyze_data(resampled_vel, resampled_acc, resampled_curve, resampled_rev, resampled_turn, times, model, z_norm,input_stats,  z,   lag=16, inclusion_thresh =3, model_type="LSTM",normalize=False,  remove_revs=False): 
    '''
    times: onset times
    '''
    T = 5 #num time lags
    beh_map = [z_norm.min(), z_norm.min()/2 + z_norm.max()/2, z_norm.max()]
    if remove_revs:
        rev_id = beh_map[1]
    else:
        rev_id = 100000

    behavior_input = np.array(resampled_rev+2*resampled_turn, dtype=np.float64)
    behavior_input -= np.array(z).mean()
    behavior_input /= np.array(z).std()

    inferred_phases = np.zeros((len(times), len(resampled_vel))) +np.nan
    inferred_rad = np.zeros((len(times), len(resampled_vel))) +np.nan
    final_behaviors = np.zeros((len(times), len(resampled_vel))) +np.nan
    initial_behaviors = np.zeros((len(times), len(resampled_vel))) +np.nan
    initial_cts_beh = np.zeros((len(times), len(resampled_vel))) +np.nan

    final_behaviors_all = []
    inferred_phases_all = []
    inferred_rad_all = []
    behaviors_all = []
    X_all_LSTM = []
    for new_worm_idx in range(len(resampled_vel)):
        X_new_worm = np.stack([resampled_vel[new_worm_idx], resampled_acc[new_worm_idx], resampled_curve[new_worm_idx], behavior_input[new_worm_idx]], axis=1)  
        # X_new_worm = np.stack([resampled_vel[new_worm_idx], resampled_acc[new_worm_idx], behavior_input[new_worm_idx]], axis=1)  
        X_new_tensor = torch.tensor(X_new_worm, dtype=torch.float32)

        if T > 0:
            X_new_seq1 = create_X_sequences(X_new_tensor, T)
        else: 
            X_new_seq1 = X_new_tensor
        

        # Normalize the inputs
        X_train_mean, X_train_std = X_new_seq1.mean(axis=0), X_new_seq1.std(axis=0)
        if normalize:
            X_new_seq1 = (X_new_seq1 - X_train_mean) / (X_train_std + 1e-8)  

        if model_type == "LSTM":
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                y_new_pred = model(X_new_seq1).squeeze()
            
        else:
            model.eval()
            with torch.no_grad():
                y_new_pred = model(X_new_seq1).squeeze()


        
        X_all_LSTM.append(X_new_seq1)
        
        # Denormalize the predicted outputs
        y_new_pred = y_new_pred * input_stats["y_std"] + input_stats['y_mean']  #note: y_train_std,  y_train_mean come from earlier

        inferred_phases_all.append(np.arctan2(y_new_pred[:-lag, 1], y_new_pred[:-lag, 0]))
        inferred_rad_all.append(y_new_pred[:-lag,2])
        behaviors_all.append(behavior_input[new_worm_idx][T:-lag])
        final_behaviors_all.append(np.array([resampled_rev[new_worm_idx][t:t+lag].sum()>=inclusion_thresh for t in range(T,resampled_rev[new_worm_idx].shape[0]-lag)]))
        
        for i, end_time in enumerate(times):
            if np.abs(behavior_input[new_worm_idx][end_time-1] - rev_id) > 0.01:
                inferred_phases[i, new_worm_idx] = np.arctan2(y_new_pred[end_time-T-1, 1], y_new_pred[end_time-T-1, 0])
                inferred_rad[i,new_worm_idx] = y_new_pred[end_time-T-1, 2]
                final_behaviors[i, new_worm_idx] = resampled_rev[new_worm_idx][end_time:end_time+lag].sum()>=inclusion_thresh
                initial_behaviors[i, new_worm_idx] = behavior_input[new_worm_idx][end_time-1]
                initial_cts_beh[i, new_worm_idx] = resampled_vel[new_worm_idx][end_time-1]
                
                
            # else:
            #     inferred_phases[i, new_worm_idx] = np.arctan2(y_new_pred[end_time-T-1, 1], y_new_pred[end_time-T-1, 0])
            #     final_behaviors[i, new_worm_idx] = np.nan
    # inferred_phases_shifted = np.zeros_like(inferred_phases)
    # if shift_angle_degrees is not None: 
    #     for i in range(inferred_phases.shape[0]):
    #         inferred_phases_shifted[i] = shift_angles(inferred_phases[i], np.radians(shift_angle_degrees))
    #     inferred_phases = inferred_phases_shifted   
    inferred_phases = np.array(inferred_phases)
    
    final_behaviors = np.array(final_behaviors)
    final_behaviors_all = np.concatenate(final_behaviors_all)
    inferred_phases_all= np.concatenate(inferred_phases_all)
    
    # if shift_angle_degrees is not None: 
    #     inferred_phases_all_shifted =  shift_angles(inferred_phases_all, np.radians(shift_angle_degrees))
    #     inferred_phases_all = inferred_phases_all_shifted
    
    inferred_rad_all= np.concatenate(inferred_rad_all)
    behaviors_all = np.concatenate(behaviors_all)

    return inferred_phases, final_behaviors, final_behaviors_all, inferred_phases_all, inferred_rad_all, behaviors_all, inferred_rad, initial_behaviors, initial_cts_beh, X_all_LSTM
