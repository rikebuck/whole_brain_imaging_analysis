import os
import pandas as pd
import glob
import numpy as np
from scipy import interpolate
from sklearn.decomposition import PCA
import sys
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from beh_classification.process_midlines.midline_utils import compute_spline_lengths, convert_coordinates_to_angles, resample_midline, convert_angles_to_coordinates

def get_PCA_components_from_midlines(processed_midlines, n_components=6):
    pca = PCA(n_components=n_components)
    pca.fit(processed_midlines)
    components = pca.components_
    var_explained = pca.explained_variance_ratio_
    return components, var_explained 



def get_cropped_midlines_eigenworms_and_body_angles(
                                                    midlines, 
                                                    target_length, 
                                                    frames_to_mask,
                                                    ds = 0.02, 
                                                    n_components = 8, 
                                                    smoothing = 0, 
                                                    num_resample_pts = 50
                                                    ):
    n_frames = midlines.shape[0]
    midline_lengths, full_splines, indices_length_thresh  = compute_spline_lengths(midlines, ds=ds, smoothing = smoothing , target_length = target_length)
    cropped_midlines = []
    mean_angles = []
    midline_angles = []

    for frame in range(n_frames):
        midline_cropped = full_splines[frame, 0:indices_length_thresh[frame]+1]
        resampled_midline = resample_midline(midline_cropped,num_resample_pts = num_resample_pts)#int(50*(target_length/120 )))
        cropped_midlines.append(resampled_midline[None, : , :])
        cropped_body_angles, mean_angle = convert_coordinates_to_angles(resampled_midline)
        midline_angles.append(cropped_body_angles[None, :])
        mean_angles.append(mean_angle)
    cropped_midlines = np.concatenate(cropped_midlines, axis = 0)
    midline_angles = np.concatenate(midline_angles, axis = 0)
    mean_angles = np.array(mean_angles)
    centroids = np.nanmean(cropped_midlines, axis = 1) 
    # ##confirm length is okay
    # midline_lengths1, splines1, indices_length_thresh1  = compute_spline_lengths(cropped_midlines, ds=ds, smoothing = 0 , target_length = target_length)
    # plt.hist(midline_lengths1[to_keep])
    # ##
    nan_is = np.argwhere(np.isnan(midline_lengths)).flatten()
    length_buffer = 1
    too_short_is = np.argwhere(midline_lengths<(target_length-length_buffer)).flatten()
    nan_is = np.union1d(nan_is, too_short_is)
    
    frames_to_mask = np.union1d(frames_to_mask, nan_is)
    to_keep = np.isin(np.arange(n_frames), frames_to_mask, invert=True)
    if  np.argwhere(to_keep).flatten().shape[0]<10:
        components, var_explained = None, None
    else:
        components, var_explained = get_PCA_components_from_midlines(midline_angles[to_keep, :], n_components=n_components)
    return full_splines, to_keep, cropped_midlines,centroids,  midline_angles, mean_angles,  components, var_explained

# def get_eigen_worm_contributions(mean_subtracted_midline_angles, components):
#     contributions_all = []
#     n_frames = mean_subtracted_midline_angles.shape[0]
#     for frame in range(n_frames):
#         contributions = np.dot(components, mean_subtracted_midline_angles[frame, :]) / np.linalg.norm(components, axis=1)**2
#         contributions_all.append(contributions[None, : ])
    
#     contributions_all = np.concatenate(contributions_all, axis = 0 )
#     return contributions_all

def get_contributions_and_reconstruct_body_angles(components, 
                                                midline_angles,
                                                mean_angles, 
                                                centroids, 
                                                target_length, 
                                                ):
    contributions_all = []
    reconstructed_body_angles =[]
    reconstructed_midlines = []
    n_frames = midline_angles.shape[0]
    for frame in range(n_frames):
        
        contributions = np.dot(components, midline_angles[frame, :]) / np.linalg.norm(components, axis=1)**2
        contributions_all.append(contributions)
        reconstructed_body_angle = np.dot(components.T, contributions)
        reconstructed_body_angles.append(reconstructed_body_angle)
        
        reconstructed_midline = convert_angles_to_coordinates(reconstructed_body_angle+mean_angles[frame],
                                                            distances = np.ones_like(reconstructed_body_angle)*(target_length/(reconstructed_body_angle.shape[0])))[:, [1,0]]
        
        reconstructed_midline = reconstructed_midline-np.nanmean(reconstructed_midline, axis = 0) + centroids[frame]
        reconstructed_midlines.append(reconstructed_midline)
    reconstructed_body_angles =  np.concatenate([np.array(angles)[None, :] for angles in reconstructed_body_angles])
    reconstructed_midlines =  np.concatenate([np.array(midline)[None, :, :] for midline in reconstructed_midlines])
    contributions = np.concatenate([np.array(contrs)[None, :] for contrs in contributions_all])
    return reconstructed_body_angles, reconstructed_midlines, contributions