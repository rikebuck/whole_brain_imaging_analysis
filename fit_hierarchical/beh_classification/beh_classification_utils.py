
import os
import pickle
import copy
import numpy as np
import pandas as pd

import sys
import matplotlib.pylab as plt
import cv2

# sys.path.append('/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/')
# sys.path.append('/home/fbuck/Large_Plate_Worm_Tracking/LargePlateWormTracker')
# from VectorandImageHandlers.NPCurveHandler import NPCurveHandler
# from VectorandImageHandlers.NPMatrixFunctions import multirange

#####from BehAnalyser/format_tracker_output/update_turning.py
def multirange(start_indices, end_indices, fsize):
    result = np.zeros(fsize, dtype=int)
    start_indices = start_indices.flatten().astype(np.int32)
    end_indices = end_indices.flatten().astype(np.int32)
    for i in range(start_indices.shape[0]):
        start = start_indices[i]
        end = end_indices[i]
        result[start:end] = 1
    return result

def get_start_end_is_of_ones_in_binary_array(binary_array):
    '''
    get frames in which consecutive chunk of good midlines or a "midline_chunk" starts and ends
    '''
    binary_array_buffered = np.concatenate([np.zeros(1),binary_array, np.zeros(1)])
    chunk_start_is = np.argwhere(np.diff(binary_array_buffered)==1).flatten()
    chunk_end_is = np.argwhere(np.diff(binary_array_buffered)==-1).flatten()

    return chunk_start_is, chunk_end_is
    
def get_is_long_enough_track(bin_event_track, min_duration_thres, n_frames):
    event_start_is, event_end_is = get_start_end_is_of_ones_in_binary_array(bin_event_track)
    event_durations  = event_end_is - event_start_is
    
    sufficient_length_events_is = np.argwhere(event_durations>min_duration_thres)
    new_event_starts = event_start_is[sufficient_length_events_is]
    new_event_ends = event_end_is[sufficient_length_events_is]
    updated_events_bin_track = multirange(new_event_starts, new_event_ends, n_frames)
    
    too_short_events_is = np.argwhere(event_durations<=min_duration_thres)
    too_short_event_starts = event_start_is[too_short_events_is]
    too_short_event_ends = event_end_is[too_short_events_is]
    short_events_bin_track  = multirange(too_short_event_starts, too_short_event_ends, n_frames)
    return short_events_bin_track, updated_events_bin_track
    
def calculate_HT_angles(arr1, arr2):
    magnitude_arr1 = np.linalg.norm(arr1, axis=1)
    magnitude_arr2 = np.linalg.norm(arr2, axis=1)

    # Handle zero-length vectors
    mask = (magnitude_arr1 * magnitude_arr2) != 0
    cos_angles = np.empty_like(magnitude_arr1)
    cos_angles[mask] = np.sum(arr1[mask] * arr2[mask], axis=1) / (magnitude_arr1[mask] * magnitude_arr2[mask])
    cos_angles[~mask] = np.nan

    angles_radians = np.arccos(cos_angles)
    angles_degrees = np.degrees(angles_radians)
    
    return angles_degrees



def convert_angles_to_coordinates(angles, distances=[]):
    if len(angles.shape)<2:
        angles = angles[None, :]
    if np.array(distances).shape[0]==0:
        distances = np.ones_like(angles)  # Assuming unit lengths for simplicity
    x = np.cumsum(distances * np.cos(angles), axis = 1)
    y = np.cumsum(distances * np.sin(angles), axis = 1)
    coordinates = np.concatenate((x[:, :, None], y[:, :, None]), axis = 2 )
    coordinates = np.squeeze(coordinates)
    return coordinates



