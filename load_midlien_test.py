import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import ssm
import pandas as pd
import copy
from tqdm import tqdm
import pickle 
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
from ssm.util import random_rotation, find_permutation
from ipywidgets import interact, IntSlider
import sys
import glob
sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')
from collections import defaultdict
import random 
from scipy.ndimage import gaussian_filter1d
from get_data.read_neuro_beh_data import get_exp_features, load_json
from get_data.process_neural_data import get_derivative_of_neural_activity, get_neural_activity_from_labels, get_neural_activity_and_labels
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification
from beh_classification.beh_classification_utils import get_start_end_is_of_ones_in_binary_array
import socket
import sklearn
from sklearn.cross_decomposition import CCA
from sklearn.utils import shuffle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn import svm
from matplotlib.gridspec import GridSpec
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

import torch
import torch.nn as nn
import torch.optim as optim
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
# from beh_classification.process_midlines.midline_utils import save_as_csv, load_from_csv, convert_coordinates_to_angles
def compute_signed_curvature(points):
    """
    Compute signed curvature from a sequence of 2D points.

    Parameters:
    - points: np.array of shape (n_pts, 2)

    Returns:
    - curvature: np.array of shape (n_pts-2,)
    """
    # Step 1: Compute tangent vectors
    tangents = points[1:] - points[:-1]  # shape (n_pts-1, 2)
    # tangents = midline[1:,:]-midline[0:-1,:]

    # Step 2: Compute angles of tangent vectors
    # angles = np.arctan2(tangents[:, 1], tangents[:, 0])  # shape (n_pts-1,)
    
    angles = np.arccos(tangents[:,1]/np.sqrt(np.sum(tangents**2,axis=1)))*np.sign(tangents[:,0])
    # angles = np.unwrap(angles)

    # Step 3: Compute angle differences (dθ)
    dtheta = np.diff(angles)  # shape (n_pts-2,)

    # Unwrap angles to prevent discontinuities at pi -> -pi jumps
    dtheta = np.unwrap(dtheta)

    # Step 4: Compute segment lengths (ds)
    ds = np.linalg.norm(tangents[1:], axis=1)  # shape (n_pts-2,)

    # Step 5: Curvature = dθ / ds
    curvature = dtheta / ds

    return curvature #, angles
def convert_coordinates_to_angles(midline):

    diffs = midline[1:,:]-midline[0:-1,:]
    angles = np.arccos(diffs[:,1]/np.sqrt(np.sum(diffs**2,axis=1)))*np.sign(diffs[:,0])
    angles = np.unwrap(angles)
    mean_angle = np.mean(angles)
    mean_subtracted_angle = angles-np.mean(angles)
    return mean_subtracted_angle, mean_angle

# def get_body_angle_mag_from_midline(midlines):
#     body_angle_mag_all = []
#     body_angles = []
#     for midline in midlines: 
        
#         mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
        
#         body_angle_mags = np.linalg.norm(mean_subtracted_angles)#, axis = 1)
#         body_angle_mag_all.append(body_angle_mags)
#         body_angles.append(mean_subtracted_angles)
    
#     return np.array(body_angle_mag_all), np.array(body_angles)

def get_body_angle_mag_from_midline(midlines,ratio=1):
    body_angle_mag_all = []
    body_angles = []
    for midline in midlines: 
        
        mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
        n_total_pts = mean_subtracted_angles.shape[0]
        n_pts = int(ratio*n_total_pts)
        body_angle_mags = np.linalg.norm(mean_subtracted_angles[:n_pts])#, axis = 1)
        body_angle_mag_all.append(body_angle_mags)
        body_angles.append(mean_subtracted_angles)
    
    return np.array(body_angle_mag_all), np.array(body_angles)

def get_body_angle_mag_from_midline_all_tracks(midlines, ratio=1):
    
    body_angles_all = []
    n_frames, n_tracks, n_total_pts, _  = midlines.shape
    body_angle_mag_all = np.zeros( (n_frames, n_tracks))
    for frame in range(n_frames):
        body_angles = []
        for track in range(n_tracks):
            midline = midlines[frame,track, :, : ]
            # if np.logical_not(np.any(np.isnan(midline))):
            #     print(track)
            mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
            n_pts = int(ratio*n_total_pts)
            body_angle_mags = np.linalg.norm(mean_subtracted_angles[:n_pts])#, axis = 1)
            body_angle_mag_all[frame, track] = body_angle_mags
            body_angles.append(mean_subtracted_angles)
        body_angles_all.append(body_angles)
    return np.array(body_angle_mag_all), np.array(body_angles_all)

def get_signed_curvature_from_midline_all_tracks(midlines, ratio=1):
    
    # body_angles_all = []
    n_frames, n_tracks, n_total_pts, _  = midlines.shape
    signed_curvature_all = np.zeros( (n_frames, n_tracks))
    for frame in range(n_frames):
        # body_angles = []
        for track in range(n_tracks):
            midline = midlines[frame,track, :, : ]
            # if np.logical_not(np.any(np.isnan(midline))):
            #     print(track)
            
            n_pts = int(ratio*n_total_pts)
            signed_curvature = np.sum(compute_signed_curvature(midline[:n_pts, :]))

            signed_curvature_all[frame, track] = signed_curvature
    return np.array(signed_curvature_all)#, np.array(body_angles_all)



def get_body_angle_signed_sum_from_midline_all_tracks(midlines, ratio=1):
    
    body_angles_all = []
    n_frames, n_tracks, n_total_pts, _  = midlines.shape
    body_angle_mag_all = np.zeros( (n_frames, n_tracks))
    for frame in range(n_frames):
        body_angles = []
        for track in range(n_tracks):
            midline = midlines[frame,track, :, : ]
            mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
            n_pts = int(ratio*n_total_pts)
            body_angle_mags = np.sum(mean_subtracted_angles[:n_pts])#, axis = 1)
            body_angle_mag_all[frame, track] = body_angle_mags
            body_angles.append(mean_subtracted_angles)
        body_angles_all.append(body_angles)
    return np.array(body_angle_mag_all), np.array(body_angles_all)

# def get_body_angle_l1_norm_from_midline_all_tracks(midlines, ratio=1):
    
#     body_angles_all = []
#     n_frames, n_tracks, n_total_pts, _  = midlines.shape
#     body_angle_mag_all = np.zeros( (n_frames, n_tracks))
#     for frame in range(n_frames):
#         body_angles = []
#         for track in range(n_tracks):
#             midline = midlines[frame,track, :, : ]
        
#             mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
#             n_pts = int(ratio*n_total_pts)
#             body_angle_mags = np.linalg.norm(mean_subtracted_angles[:n_pts], ord = 1)#, axis = 1)
#             body_angle_mag_all[frame, track] = body_angle_mags
#             body_angles.append(mean_subtracted_angles)
#         body_angles_all.append(body_angles)
#     return np.array(body_angle_mag_all), np.array(body_angles_all)


def load_from_csv(filename, n_frames = None, n_pts = None, n_dim = 2):
    # Load the CSV file
    reshaped_array = np.loadtxt(filename, delimiter=",")
    if n_frames is None: 
        n_frames = reshaped_array.shape[0]
    if n_pts is None: 
        n_pts = int(reshaped_array.shape[1]/2)
    # Reshape back to the original shape (n_frames, n_pts, 2)
    original_shape_array = reshaped_array.reshape(n_frames, n_pts, n_dim)
    
    return original_shape_array

def smooth_trace(raw_trace, sigma = 2):
    denoised_data = gaussian_filter1d(raw_trace, sigma=sigma)
    return denoised_data

def shift_array_by_one(array):
    _, n_dim = array.shape
    tmp = np.ones((T, n_dim))*np.nan
    tmp[1:] = array
    array = tmp
    return array

def get_midline_and_curvature_exp_date(exp_date, time_bin=1, target_length = 80, ratio=1):
    midline_dir  = f"/Users/friederikebuck/Desktop/MBL/project/outputs/{exp_date}/csvs1"
    midline_txt = os.path.join(midline_dir,f"cropped_midlines_wormtracer_target_length_{target_length}_timebin_{time_bin}.csv" )
    midlines = load_from_csv(midline_txt)
    
    fig, ax = plt.subplots()
    for i in range(50):
        ax.plot(*midlines[i, :, :].T)
    body_angle_mag, body_angles = get_body_angle_mag_from_midline(midlines, ratio = ratio)
    
    
    
    
    return midlines,body_angles,  body_angle_mag



def load_data(experiment, condition, neuron, hiscl_neuron="", path = None):
    if experiment != "hiscl":
        return load_data_helper(experiment, condition, neuron, path= path)
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
        return data3



def load_data(experiment, condition, neuron, hiscl_neuron="", path = None):
    if experiment != "hiscl":
        return load_data_helper(experiment, condition, neuron, path= path)
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
        return data3
    
def load_from_csv_midline(filename, n_frames = None, n_tracks = None, n_pts = None, n_dim = 2):
    # Load the CSV file
    reshaped_array = np.loadtxt(filename, delimiter=",")
    if n_frames is None: 
        n_frames = reshaped_array.shape[0]
    if n_pts is None: 
        n_pts = int(reshaped_array.shape[1]/2)
    # Reshape back to the original shape (n_frames, n_pts, 2)
    original_shape_array = reshaped_array.reshape(n_frames,n_tracks,  n_pts, n_dim)

    
    return original_shape_array



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
        midlines = load_from_csv(path+f"{prefix}dv_oriented_midlines_reshaped.csv")
        n_frames = turns.shape[0]
        n_tracks = turns.shape[1]
        n_pts = 100
        midlines = load_from_csv_midline(path+f"{prefix}dv_oriented_midlines_reshaped.csv",n_frames, n_tracks, n_pts )
        ratio = 80/140
        body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = ratio)
        np.savetxt(path+f"{prefix}_{ratio}_body_angle_mag_from_midlines.csv", body_angle_mag, delimiter=",")
        
        
                # ratio = 80/140
        len_thresh = 80
        total_len = 140
        ratio = len_thresh/total_len
        body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = ratio)#[:cutoff]
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_body_angle_mag_from_midlines.csv", body_angle_mag, delimiter=",")
        
        signed_curvature = get_signed_curvature_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_curvature_from_midlines.csv", signed_curvature, delimiter=",")
        
        signed_body_angle_sum = get_body_angle_signed_sum_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_body_angle_sum_from_midlines.csv", signed_body_angle_sum, delimiter=",")
        
        len_thresh = 110
        total_len = 140
        ratio = len_thresh/total_len
        body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = ratio)#[:cutoff]
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_body_angle_mag_from_midlines.csv", body_angle_mag, delimiter=",")
        
        signed_curvature = get_signed_curvature_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_curvature_from_midlines.csv", signed_curvature, delimiter=",")
        
        signed_body_angle_sum = get_body_angle_signed_sum_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_body_angle_sum_from_midlines.csv", signed_body_angle_sum, delimiter=",")
        
        
        len_thresh = 140
        total_len = 140
        ratio = len_thresh/total_len
        body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = ratio)#[:cutoff]
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_body_angle_mag_from_midlines.csv", body_angle_mag, delimiter=",")
        
        signed_curvature = get_signed_curvature_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_curvature_from_midlines.csv", signed_curvature, delimiter=",")
        
        signed_body_angle_sum = get_body_angle_signed_sum_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_body_angle_sum_from_midlines.csv", signed_body_angle_sum, delimiter=",")
        
        #'/Volumes/Seagate Por/csv_1/_Aim1_RIM_duty_cycle/_Aim1_RIM_duty_cycle_RIMpReaChR_atr1his0__0.5714285714285714_body_angle_mag_from_midlines.csv'
        #'/Volumes/Seagate Por/csv_1/_Aim1_AVA_duty_cycle/_Aim1_AVA_duty_cycle_AVApChrimson_atr1his0__0.5714285714285714_body_angle_mag_from_midlines.csv'
        ##svae body angel mag as csv 
        
        thresh_time = 667000
        exp_vel.values[times.values > thresh_time*6] = np.nan
        exp_acc.values[times.values > thresh_time*6] = np.nan
        exp_curve.values[times.values > thresh_time*6] = np.nan
        exp_behaviors.values[times.values > thresh_time*6] = np.nan
        pauses.values[times.values>thresh_time*6] = np.nan
        turns.values[times.values>thresh_time*6] = np.nan

        cutoff= 250
        exp_vel = exp_vel.dropna(axis=1, how="any").values[:,:cutoff].T
        exp_acc = exp_acc.dropna(axis=1, how="any").values[:,:cutoff].T
        exp_curve = exp_curve.dropna(axis=1, how="any").values[:,:cutoff].T
        exp_behaviors = exp_behaviors.dropna(axis=1, how="any").values[:,:cutoff].T
        pauses = pauses.dropna(axis=1, how="any").values[:,:cutoff].T
        turns = turns.dropna(axis=1, how="any").values[:,:cutoff].T
        body_angle_mag = np.delete(body_angle_mag, all_nans, axis=1).T
        # midlines = np.delete(midlines, all_nans, axis=1)
        # midlines = np.swapaxes(midlines, 0, 1)

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
        midlines = load_from_csv_midline(path+f"{prefix}dv_oriented_midlines_reshaped.csv",n_frames, n_tracks, n_pts )
        # ratio = 80/140
        len_thresh = 80
        total_len = 140
        ratio = len_thresh/total_len
        body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = ratio)#[:cutoff]
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_body_angle_mag_from_midlines.csv", body_angle_mag, delimiter=",")
        
        # signed_curvature = get_signed_curvature_from_midline_all_tracks(midlines,  ratio = ratio)
        # np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_curvature_from_midlines.csv", signed_curvature, delimiter=",")
        
        signed_body_angle_sum = get_body_angle_signed_sum_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_body_angle_sum_from_midlines.csv", signed_body_angle_sum, delimiter=",")
        
        len_thresh = 110
        total_len = 140
        ratio = len_thresh/total_len
        body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = ratio)#[:cutoff]
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_body_angle_mag_from_midlines.csv", body_angle_mag, delimiter=",")
        
        # signed_curvature = get_signed_curvature_from_midline_all_tracks(midlines,  ratio = ratio)
        # np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_curvature_from_midlines.csv", signed_curvature, delimiter=",")
        
        signed_body_angle_sum = get_body_angle_signed_sum_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_body_angle_sum_from_midlines.csv", signed_body_angle_sum, delimiter=",")
        
        
        len_thresh = 140
        total_len = 140
        ratio = len_thresh/total_len
        body_angle_mag, _ = get_body_angle_mag_from_midline_all_tracks(midlines, ratio = ratio)#[:cutoff]
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_body_angle_mag_from_midlines.csv", body_angle_mag, delimiter=",")
        
        # signed_curvature = get_signed_curvature_from_midline_all_tracks(midlines,  ratio = ratio)
        # np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_curvature_from_midlines.csv", signed_curvature, delimiter=",")
        
        signed_body_angle_sum = get_body_angle_signed_sum_from_midline_all_tracks(midlines,  ratio = ratio)
        np.savetxt(path+f"{prefix}_{len_thresh}_{total_len}_signed_body_angle_sum_from_midlines.csv", signed_body_angle_sum, delimiter=",")
        
        #'/Volumes/Seagate Por/csv_1/_Aim1_RIM_duty_cycle/_Aim1_RIM_duty_cycle_RIMpReaChR_atr1his0__0.5714285714285714_body_angle_mag_from_midlines.csv'
        #'/Volumes/Seagate Por/csv_1/_Aim1_AVA_duty_cycle/_Aim1_AVA_duty_cycle_AVApChrimson_atr1his0__0.5714285714285714_body_angle_mag_from_midlines.csv'
        ##svae body angel mag as csv 
        

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
        body_angle_mag = np.delete(body_angle_mag, all_nans, axis=1).T
        midlines = np.delete(midlines, all_nans, axis=1)
        midlines = np.swapaxes(midlines, 0, 1)

    return exp_vel, exp_acc, exp_curve, exp_behaviors, pauses, turns, body_angle_mag
        




def load_all_data():
    # this just returns lists of traces, neural labels, and behavior for each worm
    # Also includes dtrace
    # be warned that all elements of the list have different shapes
    host = socket.gethostname()[0:6]
    if host == 'Bennet' or host == 'bennet' or 'northwestern' in socket.gethostname():
        json_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/Neuropal_no_heat/"
        h5_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/processed_h5/"
        beh_data_dir_parent = "/Users/bennetsakelaris/Desktop/behavior/outputs_sparse/"
    else:
        json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
        h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
        beh_data_dir_parent = "/Users/friederikebuck/Desktop/MBL/project/outputs/"


    directories = glob.glob(json_dir+"*.json") #get filename/paths of all the json files
    exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates
    for i in range(len(exp_dates)):
        print("Worm {}, date: {}".format(i, exp_dates[i]))

    full_traces = []
    full_neural_labels = []
    full_beh_classification = []
    full_beh_data = []

    for exp_date in exp_dates:
        dt, T, beh_data, neural_data, neuroID_to_key = get_exp_features(exp_date, 
                                                                    json_dir = json_dir, 
                                                                    h5_dir = h5_dir)
        behavior_features = get_behavior_features(beh_data)
        beh_data.update(behavior_features)
        midlines, body_angles,  body_angle_mag = get_midline_and_curvature_exp_date(exp_date, time_bin=1, target_length = 80)
        beh_data["body_angle_mag"] = body_angle_mag
        beh_data["midlines"] = midlines
        beh_data["body_angles"] = body_angles
# T = 1599
# full_traces, full_neural_labels, full_beh_classification, full_beh_data = load_all_data()
print()

# experiment = "duty_cycle"
# # condition = "exp"
# condition = "control"
# neuron = "RIM"
# # path = "/Users/friederikebuck/beh_analysis_results/csv/_Aim1_RIM_duty_cycle/"
# path = "/Volumes/Seagate Por/csv_1/_Aim1_RIM_duty_cycle/"

# exp_vel, exp_acc, exp_curve1, exp_behaviors, pauses, turns, body_angle_mag = load_data(experiment, condition, neuron, path= path)


# experiment = "duty_cycle"
# # condition = "exp"
# condition = "control"
# neuron = "AVA"
# path = "/Volumes/Seagate Por/csv_1/_Aim1_AVA_duty_cycle/"

experiment = "duty_cycle"
condition = "exp"
# condition = "control"
neuron = "AIB"
path = "/Volumes/Seagate Por/csv_1/_AIBC-3_Aim1_AIB_duty_cycle/"

exp_vel, exp_acc, exp_curve1, exp_behaviors, pauses, turns, body_angle_mag = load_data(experiment, condition, neuron, path= path)