#%%
#get eigenworms using PCA 
import copy

import numpy as np
import cv2
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from shapely.geometry import LineString
import pickle
import copy
import numpy as np
import pandas as pd

import sys
import matplotlib.pylab as plt
import cv2

import glob 

####extract behavioral features from 
#json file
import os 
import json
import numpy as np
from scipy.interpolate import interp1d, griddata
import math
import copy
import h5py 
import sys 
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/')
from FileHandlers.OSHandler import OSHandler
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from get_data.read_neuro_beh_data_high_fps import get_exp_features as get_exp_features_fps
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification
from beh_classification.beh_classifications import is_pausing, is_forward, is_reversing,is_turning, get_reorientation_types, is_looping 

from beh_classification.img_mask_specific_features import get_is_looping_from_mask_dir_confocal
from get_data.flavell_raw_nir_extraction.convert_nir_to_confocal import convert_nir_to_confocal,convert_confocal_to_nir,  evenly_sample_nir
from beh_classification.process_midlines.midline_utils import compute_spline_lengths, convert_coordinates_to_angles, get_midlines_from_worm_tracer_csv, resample_midline, convert_angles_to_coordinates
from beh_classification.process_midlines.midline_utils import save_as_csv, load_from_csv
from beh_classification.process_midlines.eigenworm_utils import get_PCA_components_from_midlines
# from beh_analysis.behavior_features import convert_angles_to_coordinates, get_beh_classification
# from get_data.read_neuro_beh_data import get_processed_beh_features
from get_data.read_neuro_beh_data import get_exp_features
from scipy.interpolate import splprep, splev
import numpy as np
from sklearn.decomposition import PCA

import numpy as np
from scipy import interpolate


def create_spline(points, n_spline_pts, smoothness = 0 ):
    # Convert the points list into separate X and Y arrays
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # Perform spline interpolation
    tck, u = splprep([x, y], s=smoothness, per =0)

    # Evaluate the spline on a finer parameterization
    u_fine = np.linspace(0, 1, num=n_spline_pts)
    spline_x, spline_y = splev(u_fine, tck)

    # Plot the original points and the spline
    # plt.plot(x, y, 'ro', label='Original Points')
    # plt.plot(spline_x, spline_y, 'b-', label='Spline')
    # plt.legend()
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()
    return spline_x, spline_y, 

def get_tail_is_from_body_angles(body_angles):
    n_frames, n_pts = body_angles.shape
    tails = np.concatenate([body_angles, np.zeros((n_frames, 1))*np.nan], axis =1)

    tail_is = []
    for frame in range(n_frames):
        tail_i = np.min(np.argwhere(np.isnan(tails[frame, :])).flatten())-1
        tail_is.append(tail_i)
    return np.array(tail_is)
def get_tail_is(oriented_midlines_track):
    #nan_is = np.argwhere(np.isnan(oriented_midlines_track[:,:,0]))
    # np.isnan(oriented_midlines_track[:,:,0])
    n_frames, n_pts, n_dim = oriented_midlines_track.shape
    tails = np.concatenate([oriented_midlines_track[:,:,0], np.zeros((n_frames, 1))*np.nan], axis =1)

    tail_is = []
    for frame in range(n_frames):
        tail_i = np.min(np.argwhere(np.isnan(tails[frame, :])).flatten())-1
        tail_is.append(tail_i)
    return np.array(tail_is)


def resample_midline_and_get_body_angle(midline, midline_length, midline_length_thresh, max_n_pts= 101, smoothness = 0):
    if midline_length>= midline_length_thresh: 
        n_midline_pts = max_n_pts
    else: 
        n_midline_pts = math.floor((midline_length/midline_length_thresh)*max_n_pts)
    spline_x, spline_y, tck = create_spline(midline, n_midline_pts,smoothness )
    midline_spline = np.vstack([spline_x, spline_y]).T
    body_angle, mean_body_angle = convert_coordinates_to_angles(midline_spline)
    return body_angle, mean_body_angle

def convert_all_coordinates_to_angles(midline_timeseries, midline_lengths, midline_length_thresh, n_pts= 101, smoothness = 0):
    n_frames =  midline_timeseries.shape[0]
    body_angles = np.zeros((n_frames, n_pts))*np.nan
    mean_body_angles = np.zeros((n_frames,))*np.nan
    for frame in range(n_frames):
        body_angle, mean_body_angle = resample_midline_and_get_body_angle(midline_timeseries[frame],
                                                      midline_lengths[frame],
                                                      midline_length_thresh,
                                                      max_n_pts= n_pts, 
                                                      smoothness = smoothness)
        n_body_angle_pts = body_angle.shape[0]
        body_angles[frame, :n_body_angle_pts ] = body_angle
        mean_body_angles[frame] = mean_body_angle
    return body_angles, mean_body_angles




def check_midlines_length_cutoff(midline_lengths, length_cutoff,  mask_save_dir, n_imgs_to_show = 5):
    frames = np.unique(np.argwhere(np.abs(midline_lengths-length_cutoff)<1)[:,0])
    sample_frames = np.random.choice(frames, size=n_imgs_to_show, replace=False)
    img_filenames = [os.path.join(mask_save_dir, str(frame).zfill(5)+".png") for frame in sample_frames]
    for img in img_filenames:
        fig, ax = plt.subplots()
        print(img)
        plt.imshow(cv2.imread(img))
        plt.title(str(length_cutoff) + "\n"+os.path.basename(img))
        
        
def plot_eigenworms(body_angles, tail_is, midline_save_dir, n_pts_cutoff = 47, n_components = 8, close_plot = True):
    to_keep = np.argwhere(tail_is>=n_pts_cutoff).flatten()
    body_angles = copy.deepcopy(body_angles)[to_keep, :n_pts_cutoff]
    components, var_explained = get_PCA_components_from_midlines(body_angles, n_components=n_components)

    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, np.cumsum(var_explained).shape[0]), np.cumsum(var_explained))
    ax.set_xlabel("PC")
    ax.set_ylabel("variance explained")
    # ax.set_ylim([0,1.2])

    plt.savefig(os.path.join(midline_save_dir,"explained_var.jpeg"))
    
    fig, axs = plt.subplots(n_components)
    # print()
    for i in range(n_components):
        axs[i].scatter(np.linspace(0,1,num = components.shape[1]),components[i,:])
        axs[i].set_title("cutoff" + str(n_pts_cutoff) + " npts: " + str(to_keep.shape[0]) + "comp: "+ str(i)+ " var " + str(np.round(var_explained[i],2)))
        # if show_plot: 
        #     plt.show()
        # fig_list.append((fig, ax))
    OSHandler().make_new_folder(midline_save_dir, "")
    plt.savefig(os.path.join(midline_save_dir, "cutoff"+str(n_pts_cutoff)+ "eigenworms.jpeg"))
    if close_plot: 
        plt.close()




def plot_PCA_components(components, n_components = -1, show_plot = False, title = ""):
    if n_components ==-1:
        n_components = len(components)
    fig_list = []
    for i in range(n_components):
        fig, ax = plt.subplots()

        ax.scatter(np.linspace(0,1,num = components.shape[1]),components[i,:])
        ax.set_title(title +"component: "+ str(i))
        if show_plot: 
            plt.show()
        fig_list.append((fig, ax))
    
    return fig_list

def get_mean_squared_error(selected_eigenvectors, X_new):
    mean_vector = np.nanmean(X_new, axis=0)
    X_new_transformed = np.dot(X_new - mean_vector, selected_eigenvectors)
    X_new_reconstructed = np.dot(X_new_transformed, selected_eigenvectors.T) + mean_vector

    mse = np.nanmean((X_new - X_new_reconstructed) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse,X_new_transformed,  X_new_reconstructed


# def get_eigenworms(body_angles, tail_is, n_pts_cutoff = 47, n_components = 5):
#     n_frames = body_angles.shape[0]
#     to_use_i = np.argwhere(tail_is>=n_pts_cutoff).flatten()
#     body_angles_to_use = copy.deepcopy(body_angles)[to_use_i, :n_pts_cutoff]
#     components, var_explained = get_PCA_components_from_midlines(body_angles_to_use, n_components=n_components)
#     return components, var_explained

def reconstruct_midlines(body_angles, tail_is, components = None, n_pts_cutoff = 47, n_components = 5, tail_i_thresh = 30):
    
    if components is None: 
        components, var_explained = get_eigenworms(body_angles, tail_is, 
                                                n_pts_cutoff = n_pts_cutoff,
                                                n_components = n_components
                                                )
    n_frames = body_angles.shape[0]

    eigen_worm_contributions = np.zeros((n_frames, n_components))
    reconstructed_body_angles = np.zeros((n_frames, n_pts_cutoff))
    for frame in range(n_frames):

        tail_i = min(tail_is[frame], n_pts_cutoff)
        if tail_i<tail_i_thresh:
            reconstructed_body_angle = np.nan
            reconstructed_body_angles[frame, :] = np.nan
            continue
        x = body_angles[frame, :tail_i]
        
        contributions = np.dot(components[:, :tail_i], x) / np.linalg.norm(components[:, :tail_i], axis=1)**2
        eigen_worm_contributions[frame, :] = contributions
        reconstructed_body_angle = np.dot(components.T, contributions)
        reconstructed_body_angles[frame, :] = reconstructed_body_angle
        
    return components, eigen_worm_contributions, reconstructed_body_angles






def crop_and_interpol_eigenworms(eigenworms_TA, n_pts_cutoff = 45, fraction_to_crop = 0.85):
    n_org_pts, n_components = eigenworms_TA.shape
    n_cropped_pts = int(n_org_pts*fraction_to_crop)
    eigenworms_TA_cropped = copy.deepcopy(eigenworms_TA[:n_cropped_pts, :])
    eigenworms_TA_interpol = np.zeros((n_pts_cutoff, n_components))

    # Define the original and new x values
    original_x = np.arange(n_cropped_pts)
    new_x = np.linspace(0, n_cropped_pts - 1, n_pts_cutoff)


    for i in range(n_components):
        interpolator = interp1d(original_x, eigenworms_TA_cropped[:, i], kind='linear', fill_value="extrapolate")
        eigenworms_TA_interpol[:, i] = interpolator(new_x)
        
    return eigenworms_TA_interpol
    ####
    

def interpolate_data_at_timepoints(data, evenly_spaced_times, frame_times):#actual_times_csv):
    
#        frame_times = pd.read_csv(actual_times_csv, sep=',', usecols=['frame_time']).to_numpy().flatten()

    interpolated_data = griddata(frame_times, data,evenly_spaced_times, method='linear')
    return interpolated_data


def get_midline_lengths1(coords):
    
    # Initialize an array to store curve lengths for each time step
    curve_lengths = []

    # Iterate over each time step in the time series
    for t in range(coords.shape[0]):
        # Extract the 2D coordinates for the current time step
        current_coords = coords[t]

        # Calculate cumulative distance along the curve
        cumulative_distance = np.cumsum(np.sqrt(np.sum(np.diff(current_coords, axis=0)**2, axis=1)))
        cumulative_distance = np.insert(cumulative_distance, 0, 0)  # Insert a 0 at the beginning

        # Interpolate to re-sample the curve at uniform intervals
        n_samples = 100  # Number of points for re-sampling
        uniform_distance = np.linspace(0, cumulative_distance[-1], n_samples)
        interp_func = interp1d(cumulative_distance, current_coords, axis=0)
        uniform_coords = interp_func(uniform_distance)

        # Calculate the length using the re-sampled coordinates
        diffs = np.diff(uniform_coords, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        curve_length = np.sum(distances)
        
        # Append the curve length for this time step
        curve_lengths.append(curve_length)

    # Convert the list to a NumPy array for further processing
    curve_lengths = np.array(curve_lengths)
    return curve_lengths

def get_midline_lengths2(coords):
    curve_lengths = []
    for t in range(coords.shape[0]):
        # Extract the 2D coordinates for the current time step
        current_coords = coords[t]

        # Create a LineString object using the current coordinates
        line = LineString(current_coords)
        
        # Calculate the length of the curve
        curve_length = line.length
        
        # Append the curve length for this time step
        curve_lengths.append(curve_length)

    # Convert the list to a NumPy array for further processing
    curve_lengths = np.array(curve_lengths)
    return curve_lengths


if __name__ == "__main__":
#%%

    exp_date = "2023-03-07-01"
    # convert_all_angle_to_coordinates(midline_timeseries,  n_pts= 101, smoothness = 0)
    json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
    h5_dir = "/Volumes/Extreme SSD/data/data_processed_high_fps/h5/"
    beh_data_fps, neural_data, neuroID_to_key = get_exp_features_fps(exp_date, 
                                                                json_dir = json_dir, 
                                                                h5_dir = h5_dir, 
                                                                h5_suffix = "_output_data_dicts.h5" 
                                                                )

    # speed_stage = beh_data_fps["speed_stage"]
    # nir_speed_stage = beh_data_fps["nir_speed_stage"]
    # body_angles_fps = beh_data_fps['nir_body_angle_absolute']
    # body_angles_fps = body_angles_fps-np.nanmean(body_angles_fps, axis = 1)[:, None]
    # midlines_fps = convert_angles_to_coordinates(body_angles_fps)#, distances=[])
    # midlines_fps = midlines_fps[1:]

    json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
    h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
    dt, T, beh_data_confocal, _, _ = get_exp_features(exp_date, 
                                                                json_dir = json_dir, 
                                                                h5_dir = h5_dir)
    behavior_features = get_behavior_features(beh_data_confocal)
    beh_data_confocal.update(behavior_features)

    bin_img_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/bin_imgs/"
    

    
    n_frames_confocal = 1600
    nir_to_confocal = beh_data_fps["nir_to_confocal"]
    n_frames_nir = nir_to_confocal.shape[0]-1
    is_looping_bin_img_confocal = get_is_looping_from_mask_dir_confocal(n_frames_nir, 
                                          n_frames_confocal
                                          ,nir_to_confocal, bin_img_dir, area_to_perimeter_ratio_thres = 1, area_to_perimeter = 6)

    behavior_classification_confocal = get_behavior_classification(beh_data_confocal, n_looping_pts_thres = 0.3, looping = is_looping_bin_img_confocal)
    z_confocal = behavior_classification_confocal["is_fwd"]+ 2*behavior_classification_confocal["is_rev"]+ 4*behavior_classification_confocal["is_pause"]+3*behavior_classification_confocal["is_turn"]-1     
    z_nir = convert_confocal_to_nir(beh_data_fps['nir_to_confocal'], z_confocal, indices_to_skip = set([]))[1:]

    #%%
    raw_nir_midlines_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/results/"
    mask_img_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/masked_imgs"

    ##################
    ##################
    midlines1 = get_midlines_from_worm_tracer_csv(raw_nir_midlines_dir)
    # midline_lengths = get_all_midline_lengths(midlines1)
    # n_frames = midline_lengths.shape[0]-1
    # frames = np.random.randint(0, n_frames, 9).tolist()
    # n_frames = len(frames)
    # for ds in np.arange(0.001, 0.1, 0.01):
    #     fig, ax = plt.subplots()
    #     midline_lengths, midlines, indices  = compute_spline_lengths(midlines1, ds=ds, smoothing = 0 )
    #     ax.hist(midline_lengths, bins = 1000)
    #     ax.set_title("midline_lengths " + str(ds))
        
        
    #     fig, ax = plt.subplots(3,3)
    #     ax[0,0].set_title(str(ds))
    #     for i, frame in enumerate(frames):
    #         row, col = divmod(i, 3)
    #         img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
    #         ax[row, col].imshow(img)
    #         ax[row, col].scatter(*midlines[frame].T, s = 0.5)
    #         ax[row, col].scatter(*midlines[frame, indices[frame], :].T, s = 10)
    #     # plt.savefig(os.path.join(midline_save_dir, str(frame)+"_midline.jpeg"))

    #     print()
    #     plt.close()
    # plt.show()
    # for ds in np.arange(0.001, 0.4, 0.05):
    #     fig, ax = plt.subplots()
    #     midline_lengths, midlines, indices  = compute_spline_lengths(midlines1, ds=ds, smoothing = 0 )
    #     ax.hist(midline_lengths, bins = 1000)
    #     ax.set_title("midline_lengths " + str(ds))
    
    #     fig, ax = plt.subplots(3,3)
    #     ax[0,0].set_title(str(ds))
    #     for i, frame in enumerate(frames):
    #         row, col = divmod(i, 3)
    #         img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
    #         ax[row, col].imshow(img)
    #         ax[row, col].scatter(*midlines[frame].T, s = 0.5)
    #         ax[row, col].scatter(*midlines[frame, indices[frame], :].T, s = 10)

    # plt.show()
    # plt.close()    

    ##################
    #######################chose ds = 0.02

    '''
    - Mask out: 
    - 1013 - 1124 - loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
    - 7646 - 7737 - but mark as turn (this is okay just doesn follow head superrr well )...by 7754 it catches up 
    - 12437-12617 - loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
    - 14325 very short midline  - this is fine if only conside first 3/4 of worm..can keep 
    - 16679-16824(def - 16858 - mask out 
        - Mask out huge changes in stage velcoity
        - Makes out when area too small) 

    '''


            
    ########################
    ########################
    ########################
    ########################
    # midline_save_dir = "/Users/friederikebuck/Desktop/MBL/project/figs/eigenworm_length_cuttoff_masked/"
    # ds = 0.02 
    # target_length = 80
    # n_components = 8
    # comp_fig_axs = [plt.subplots() for i in range(n_components)]
    # comp_fig_axs1 = [plt.subplots() for i in range(n_components)]
    # n_frames = midlines1.shape[0]
    # frames = np.random.randint(0, n_frames, 9).tolist()
    # for target_length in range(50,125, 10):


    #     midline_lengths, splines, indices_length_thresh  = compute_spline_lengths(midlines1, ds=ds, smoothing = 0 , target_length = target_length)
        
    #     # fig, ax = plt.subplots()
    #     # ax.hist(midline_lengths, bins = 1000)
    #     # ax.set_title("midline_lengths " + str(ds))
    #     fig, ax = plt.subplots(3,3)
    #     ax[0,0].set_title(str(target_length))
    #     for i, frame in enumerate(frames):
    #         row, col = divmod(i, 3)
    #         img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
    #         ax[row, col].imshow(img)
    #         ax[row, col].scatter(*splines[frame].T, s = 0.5)
    #         ax[row, col].scatter(*splines[frame, indices_length_thresh[frame], :].T, s = 10)
    #     fig.savefig(os.path.join(midline_save_dir, "example_midlines_"+str(target_length)+ "all_comps.jpeg"))
    #     cropped_midlines = []
    #     mean_angles = []
    #     midline_angles = []


    #     for frame in range(n_frames):
    #         midline_cropped = splines[frame, 0:indices_length_thresh[frame]+1]
    #         resampled_midline = resample_midline(midline_cropped,num_resample_pts = int(50*(target_length/120 )))
    #         cropped_midlines.append(resampled_midline[None, : , :])
    #         mean_subtracted_angle, mean_angle = convert_coordinates_to_angles(resampled_midline)
    #         midline_angles.append(mean_subtracted_angle[None, :])
    #         mean_angles.append(mean_angle)


    #     cropped_midlines = np.concatenate(cropped_midlines, axis = 0)
    #     midline_angles = np.concatenate(midline_angles, axis = 0)
    #     mean_angles = np.array(mean_angles)
        
    #     for i, frame in enumerate(frames):
    #         row, col = divmod(i, 3)
    #         # img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
    #         # ax[row, col].imshow(img)
    #         ax[row, col].scatter(*cropped_midlines[frame].T, s = 0.5, alpha = 0.5, c = "violet")
    #         ax[row, col].scatter(*convert_angles_to_coordinates(midline_angles[frame]+mean_angles[frame]).T, s = 0.5, alpha = 0.5)
            
    #     fig.savefig(os.path.join(midline_save_dir, "example_midlines_"+str(target_length)+ "all_comps.jpeg"))
    #     nan_is = np.argwhere(np.isnan(midline_lengths)).flatten()
    #     frames_to_mask = np.union1d(frames_to_mask, nan_is)
    #     to_keep = np.isin(np.arange(n_frames), frames_to_mask, invert=True)
    #     # non_nan_is = np.argwhere(~np.isnan(midline_lengths)).flatten()
        
    #     components, var_explained = get_PCA_components_from_midlines(midline_angles[to_keep, :], n_components=n_components)
        
    #     fig, ax = plt.subplots(3,3)
    #     ax[0,0].set_title(str(target_length))
    #     for i, frame in enumerate(frames):
    #         row, col = divmod(i, 3)
    #         # img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
    #         # ax[row, col].imshow(img)
    #         # reconstruct_midline = components
    #         contributions = np.dot(components, midline_angles[frame, :]) / np.linalg.norm(components, axis=1)**2
            
    #         reconstructed_body_angle = np.dot(components.T, contributions)
    #         # ax[row, col].scatter(*reconstructed_body_angle[frame].T, s = 0.5, alpha = 0.5, c = "violet")
    #         ax[row, col].scatter(*convert_angles_to_coordinates(reconstructed_body_angle+mean_angles[frame]).T, s = 0.5, alpha = 0.5)
    #         ax[row, col].scatter(*cropped_midlines[frame].T, s = 0.5, alpha = 0.5, c = "violet")
    #         ax[row, col].scatter(*convert_angles_to_coordinates(midline_angles[frame]+mean_angles[frame]).T, s = 0.5, alpha = 0.5)
        
    #     fig.savefig(os.path.join(midline_save_dir, "example_reconstructed_midlines"+str(target_length)+ "all_comps.jpeg")) 

    #     fig, ax = plt.subplots()
    #     ax.scatter(np.arange(0, np.cumsum(var_explained).shape[0]), np.cumsum(var_explained))
    #     ax.set_xlabel("PC")
    #     ax.set_ylabel("variance explained")

    #     OSHandler().make_new_folder(midline_save_dir, "")
    #     fig, axs = plt.subplots(n_components)
        
    #     for i in range(n_components):
    #         axs[i].scatter(np.linspace(0,1,num = components.shape[1]),components[i,:])
    #         axs[i].set_title("target_length" + str(target_length) + "comp: "+ str(i)+ " var " + str(np.round(var_explained[i],2)))
            
    #         # fig1, ax1 = plt.subplots()
    #         ax1 = comp_fig_axs[i][1]
    #         midline_comp = convert_angles_to_coordinates(components[i,:], distances = [])#np.ones(components[i,:].shape[0])*target_length)
    #         ax1.scatter(*midline_comp.T)
    #         ax1.plot(*midline_comp.T, label = str(target_length) )
    #         # ax1.set_title( "target_length" + str(target_length) + "comp: "+ str(i)+ " var explained: " + str(np.round(var_explained[i],2)))
    #         ####
    #         ax1 = comp_fig_axs1[i][1]
    #         resampled_comp = resample_midline1(midline_comp, num_resample_pts =  50, final_length = target_length)
    #         # resampled_comp = resample_midline(midline_comp[i,:], num_resample_pts =  50*(target_length/120))#, final_length = target_length)
    #         ax1.scatter(*resampled_comp.T)
    #         ax1.plot(*resampled_comp.T, label = str(target_length) )
    #         # fig1.savefig(os.path.join(midline_save_dir, "target_length"+str(target_length)+ "comp"+ str(i)+".jpeg"))
    #     fig.savefig(os.path.join(midline_save_dir, "target_length"+str(target_length)+ "all_comps.jpeg"))
    # for i, (fig, ax) in enumerate(comp_fig_axs):
    #     ax.legend()
    #     fig.savefig(os.path.join(midline_save_dir, "comp_with_dist"+ str(i)+".jpeg"))

    # for i, (fig, ax) in enumerate(comp_fig_axs1):
    #     ax.legend()
    #     fig.savefig(os.path.join(midline_save_dir, "comp_resampled_"+ str(i)+".jpeg"))
    # plt.show()

    # for i in range(n_components):
    #     axs[i].scatter(np.linspace(0,1,num = components.shape[1]),components[i,:])
    #     axs[i].set_title("target_length" + str(target_length) + "comp: "+ str(i)+ " var " + str(np.round(var_explained[i],2)))
        
    #     # fig1, ax1 = plt.subplots()
    #     ax1 = comp_fig_axs[i][1]
    #     midline_comp = convert_angles_to_coordinates(components[i,:], distances = [])#np.ones(components[i,:].shape[0])*target_length)
    #     ax1.scatter(*midline_comp.T)
    #     ax1.plot(*midline_comp.T, label = str(target_length) )
    #     # ax1.set_title( "target_length" + str(target_length) + "comp: "+ str(i)+ " var explained: " + str(np.round(var_explained[i],2)))
    #     ####
    #     ax1 = comp_fig_axs1[i][1]
    #     resampled_comp = resample_midline1(midline_comp, num_resample_pts =  50, final_length = target_length)
    #     # resampled_comp = resample_midline(midline_comp[i,:], num_resample_pts =  50*(target_length/120))#, final_length = target_length)
    #     ax1.scatter(*resampled_comp.T)
    #     ax1.plot(*resampled_comp.T, label = str(target_length) )




    # ######################
    # #####################
    # #####################
    # #####################
    # ####get contributions 
    # for frame in range(n_frames):
    #     contributions = np.dot(components, midline_angles[frame, :]) / np.linalg.norm(components, axis=1)**2
    #     reconstructed_body_angle = np.dot(components.T, contributions)

    # reconstructed_body_angles = reconstructed_body_angles-np.nanmean(reconstructed_body_angles, axis = 1)[:, None]
    # reconstructed_body_angles = reconstructed_body_angles+mean_body_angles[:, None]
    # n_frames = midline_lengths.shape[0]
    # reconstructed_midlines = convert_angles_to_coordinates(reconstructed_body_angles, distances = np.ones((n_frames,n_pts_cutoff))*(midline_lengths[:, None]/n_pts_cutoff) )

    # fig, ax = plt.subplots()
    # ax.scatter(np.arange(0, np.cumsum(var_explained).shape[0]), np.cumsum(var_explained))
    # ax.set_xlabel("PC")
    # ax.set_ylabel("variance explained")


    ########################
    ########################
    #######################
    ###ds = 0.02 
    ###target_length = 85
    ########################
            
    ds = 0.02 
    # target_length = 115#85#100#
    for target_length in [85]:#, 100, 115]:
        frames_to_mask = [
        list(range(1013, 1124)),  #  loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
        list(range(7646, 7737)),  #  loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
        list(range(12437, 12617)),  #  loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
        list(range(16679, 16858)), #issue with stage -->  doesnt align to head super well
        ]
        frames_to_mask = np.concatenate([np.array(f) for f in frames_to_mask])
        n_components = 8

        n_frames = midlines1.shape[0]
        frames = np.random.randint(0, n_frames, 9).tolist()

        midline_lengths, splines, indices_length_thresh  = compute_spline_lengths(midlines1, ds=ds, smoothing = 0 , target_length = target_length)

        cropped_midlines = []
        mean_angles = []
        midline_angles = []
        for frame in range(n_frames):
            midline_cropped = splines[frame, 0:indices_length_thresh[frame]+1]
            resampled_midline = resample_midline(midline_cropped,num_resample_pts = 50)#int(50*(target_length/120 )))
            cropped_midlines.append(resampled_midline[None, : , :])
            mean_subtracted_angle, mean_angle = convert_coordinates_to_angles(resampled_midline)
            midline_angles.append(mean_subtracted_angle[None, :])
            mean_angles.append(mean_angle)
        cropped_midlines = np.concatenate(cropped_midlines, axis = 0)
        midline_angles = np.concatenate(midline_angles, axis = 0)
        mean_angles = np.array(mean_angles)
        
        midline_lengths1, splines1, indices_length_thresh1  = compute_spline_lengths(cropped_midlines, ds=ds, smoothing = 0 , target_length = target_length)
        #plt.hist(midline_lengths1[to_keep])
        nan_is = np.argwhere(np.isnan(midline_lengths)).flatten()
        length_buffer = 1
        too_short_is = np.argwhere(midline_lengths<(target_length-length_buffer)).flatten()
        nan_is = np.union1d(nan_is, too_short_is)
        
        frames_to_mask = np.union1d(frames_to_mask, nan_is)
        to_keep = np.isin(np.arange(n_frames), frames_to_mask, invert=True)

        components, var_explained = get_PCA_components_from_midlines(midline_angles[to_keep, :], n_components=n_components)

        fig, ax = plt.subplots()
        ax.scatter(np.arange(0, np.cumsum(var_explained).shape[0]), np.cumsum(var_explained))
        ax.set_xlabel("PC")
        ax.set_ylabel("variance explained")
        fig.savefig("/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"explained_var.jpeg")

        for i in range(n_components):
            fig, ax = plt.subplots()
            
            midline_comp = convert_angles_to_coordinates(components[i,:], distances = [])#np.ones(components[i,:].shape[0])*target_length)
            ax.scatter(*midline_comp.T)
            ax.set_title("target_length" + str(target_length) + "comp: "+ str(i)+ " var " + str(np.round(var_explained[i],2)))
            fig.savefig("/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/comp_midline_"+str(i)+"_target_length_"+str(target_length)+"explained_var.jpeg")
            
        # ax.set_ylim([0,1.2])


            
        centroids = np.nanmean(cropped_midlines, axis = 1) 
        centroids2 = np.nanmean(splines, axis = 1) 
        contributions_all = []


        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            "purple", # pause
            "navy" #nan
            ]    
            
        

        beh = [
            "fwd", 
            "rev",
            "turn", 
            "pause"
        ]

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=4)
        midline_save_dir = "/Users/friederikebuck/Desktop/MBL/project/figs/eigenworm_masked/"
        masked = np.logical_not(to_keep)
        # for i, frame in enumerate(frames):
        plot_masks = False
        reconstructed_body_angles =[]
        reconstructed_midlines = []
        for frame in range(n_frames):
            if frame%350==0:
                print("frame", frame)
            
            
            
            contributions = np.dot(components, midline_angles[frame, :]) / np.linalg.norm(components, axis=1)**2
            contributions_all.append(contributions)
            reconstructed_body_angle = np.dot(components.T, contributions)
            reconstructed_body_angles.append(reconstructed_body_angle)
            
            reconstructed_midline = convert_angles_to_coordinates(reconstructed_body_angle+mean_angles[frame],
                                                                distances = np.ones_like(reconstructed_body_angle)*(target_length/(reconstructed_body_angle.shape[0])))[:, [1,0]]
            
            reconstructed_midline = reconstructed_midline-np.nanmean(reconstructed_midline, axis = 0) + centroids[frame]
            reconstructed_midlines.append(reconstructed_midline)
                                                                #   (midline_lengths[frame]/(reconstructed_body_angle.shape[0])))[:, [1,0]] + centroids[frame]
            if plot_masks:
                fig, ax = plt.subplots()
                # row, col = divmod(i, 3)

                img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
                ax.imshow(img)
                ax.scatter(*splines[frame].T, s = 0.5, alpha = 0.5, c = "violet", label = "fitted spline")
                ax.scatter(*cropped_midlines[frame].T, s = 0.5, alpha = 0.5, label = "cropped spline")
                head_i = 2
                ax.scatter(*reconstructed_midline[head_i].T, s = 20)
                # ax.scatter(*convert_angles_to_coordinates(midline_angles[frame]+mean_angles[frame]).T, s = 0.5, alpha = 0.5)
            
                if not np.isnan(z_nir[frame]): 
                    ax.scatter(*reconstructed_midline.T, s = 2, alpha = 0.5, c = palette[int(z_nir[frame])],  label = "reconstructed midline")
                    loopflav = str(bool(looping_flavell_nir[frame+1]))
                    beh_str = str(beh[int(z_nir[frame])])
                else: 
                    ax.scatter(*reconstructed_midline.T,s = 10,  alpha = 0.5, label = "reconstructed midline", c = "darkblue")
                    beh_str = ""
                    loopflav  = ""
                txt = ["confocal frame: " + str(beh_data_fps["nir_to_confocal"][frame])]
                txt =txt + ["beh: " + beh_str] + ["looping_img: "+str(is_looping_bin_img[frame]),  "loopflav: "+loopflav] + ["mask: "+str(masked[frame]), "a_to_ps: "+str(a_to_ps[frame])]
                text = "\n".join(txt)
                ax.text(0, 100, text, fontsize=10,  color='green')
                ax.legend()
                fig.savefig(os.path.join(midline_save_dir, str(frame)+"_midline.jpeg"), dpi=400)
                fig.savefig(os.path.join(midline_save_dir, str(frame)+"_midline.jpeg"), dpi=400)
                plt.close()
            
        reconstructed_body_angles_nir =  np.concatenate([np.array(angles)[None, :] for angles in reconstructed_body_angles])
        reconstructed_midlines =  np.concatenate([np.array(midline)[None, :, :] for midline in reconstructed_midlines])
        contributions_nir = np.concatenate([np.array(contrs)[None, :] for contrs in contributions_all])
        # all_eigen_worm_contributions_resampled = np.concatenate(all_eigen_worm_contributions_resampled, axis = 1)[time_bins:]
        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"contributions_nir.csv"
        np.savetxt(csv_name,  contributions_nir, delimiter=",", fmt='%f')


        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"eigenworm_components.csv"
        np.savetxt(csv_name,  components, delimiter=",", fmt='%f')

        nir_to_confocal = beh_data_fps["nir_to_confocal"][1:]
        time_bins = 10
        nir_to_confocal_resampled = np.repeat(np.arange(0,1601), time_bins)
        # nir_to_confocal_resampled = np.repeat(np.arange(0,1599), time_bins)


        # beh_confocal =  evenly_sample_nir(nir_to_confocal, z_nir, time_bins = time_bins)[:-1*time_bins]#[time_bins:]# first index is where nir_to_confocal is 0 which dont want 
        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/beh_confocal.csv"
        np.savetxt(csv_name,  z_confocal, delimiter=",", fmt='%f')
        
        
        # beh_resampled =  evenly_sample_nir(nir_to_confocal, z_nir, time_bins = time_bins)[:-1*time_bins]#[time_bins:]# first index is where nir_to_confocal is 0 which dont want 
        # csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/beh_resampled.csv"
        # np.savetxt(csv_name,  beh_resampled, delimiter=",", fmt='%f')
        
        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/beh_nir.csv"
        np.savetxt(csv_name,  z_nir, delimiter=",", fmt='%f')

        masked_nir = copy.deepcopy(masked)
        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"masked_nir.csv"
        np.savetxt(csv_name,  masked_nir, delimiter=",", fmt='%f')
        masked_resampled = evenly_sample_nir(nir_to_confocal, masked_nir, time_bins = time_bins)[:-1*time_bins]

        # csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"masked_resampled.csv"
        # np.savetxt(csv_name,  masked_resampled, delimiter=",", fmt='%f')


        # fig, ax = plt.subplots()
        # plt.plot(beh_resampled)

        # beh_masked = copy.deepcopy(beh_resampled)
        # beh_masked[np.argwhere(masked_resampled)] = np.nan
        # plt.plot(beh_masked)
        all_eigen_worm_contributions_resampled = []
        for comp_i in range(n_components):
            # fig, ax =plt.subplots()
            # eigen_worm_contribution_resampled = evenly_sample_nir(nir_to_confocal, eigen_worm_contributions[:, comp_i], time_bins = time_bins)[time_bins:]
            eigen_worm_contribution_resampled = evenly_sample_nir(nir_to_confocal, contributions_nir[:, comp_i], time_bins = time_bins)
            all_eigen_worm_contributions_resampled.append(eigen_worm_contribution_resampled[:, None])
            

            eigenw_confocal_org = convert_nir_to_confocal(nir_to_confocal, contributions_nir[:, comp_i])
            eigenw_confocal_resampled = convert_nir_to_confocal(nir_to_confocal_resampled, eigen_worm_contribution_resampled)
            # ax.plot(eigenw_confocal_org)
            # ax.plot(eigenw_confocal_resampled)#worked! 

        all_body_angles_resampled = []
        for i in range(reconstructed_body_angles_nir.shape[1]):
            # fig, ax =plt.subplots()
            # eigen_worm_contribution_resampled = evenly_sample_nir(nir_to_confocal, eigen_worm_contributions[:, comp_i], time_bins = time_bins)[time_bins:]
            reconstructed_body_angles_resampled = evenly_sample_nir(nir_to_confocal, reconstructed_body_angles_nir[:, i], time_bins = time_bins)
            all_body_angles_resampled.append(reconstructed_body_angles_resampled[:, None])
            


        all_eigen_worm_contributions_resampled = np.concatenate(all_eigen_worm_contributions_resampled, axis = 1)[:-1*time_bins]#[time_bins:]
        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"eigen_worm_resampled1.csv"
        np.savetxt(csv_name,  all_eigen_worm_contributions_resampled, delimiter=",", fmt='%f')
        
        all_body_angles_resampled = np.concatenate(all_body_angles_resampled, axis = 1)[:-1*time_bins]#[time_bins:]
        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"body_angles.csv"
        np.savetxt(csv_name,  all_body_angles_resampled, delimiter=",", fmt='%f')
        
        

        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/nir_to_confocal.csv"
        np.savetxt(csv_name,  nir_to_confocal, delimiter=",", fmt='%f')
        

        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/processed_midlines/2023-03-07-01/target_length"+str(target_length)+"all_midlines_coords_nir.csv"
        save_as_csv(reconstructed_midlines, csv_name)


        # all_body_angles_resampled1 = evenly_sample_nir1(nir_to_confocal, reconstructed_body_angles_nir, time_bins = time_bins)[:-1*time_bins]
        all_midlines_coords_full = []
        for i in range(reconstructed_midlines.shape[1]):
            all_midlines_coords =[]
            for j in range(reconstructed_midlines.shape[2]):
                # fig, ax =plt.subplots()
                # eigen_worm_contribution_resampled = evenly_sample_nir(nir_to_confocal, eigen_worm_contributions[:, comp_i], time_bins = time_bins)[time_bins:]
                midlines_coord = evenly_sample_nir(nir_to_confocal, reconstructed_midlines[:, i, j], time_bins = time_bins)
                all_midlines_coords.append(midlines_coord[:, None])
            all_midlines_coords = np.concatenate(all_midlines_coords, axis = 1)
            all_midlines_coords_full.append(all_midlines_coords[:, None, :])
        all_midlines_coords_full= np.concatenate(all_midlines_coords_full, axis = 1)

        all_midlines_coords_full = all_midlines_coords_full[:-1*time_bins]
        csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/processed_midlines/2023-03-07-01/target_length"+str(target_length)+"all_midlines_coords_resampled.csv"
        save_as_csv(all_midlines_coords_full, csv_name)


        print()
    #plt.show() `