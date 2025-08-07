import numpy as np 
# %%
import os
# import keypoint_moseq as kpms
import numpy as np
import matplotlib.pyplot as plt

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
from get_data.read_neuro_beh_data import get_exp_features, load_json
import sys 
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from beh_classification.process_midlines.midline_utils import save_as_csv, load_from_csv, convert_coordinates_to_angles

from load_data_fncs import load_all_data, get_exp_dates

import numpy as np
import glob 
import cv2
from pathlib import Path
import os
import sys
import copy 
import traceback
from rslds_motor_state_class_fncs import get_motor_state_start_end_is_rslds_start_end_specific
sys.path.append("/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/")
from make_grid_vid_fncs_2 import get_clip_windows_from_start_end_is, create_video_grid ,make_vid_clip, create_video_grid_color


def get_frame_start_end_from_dir(full_jpeg_dir,frame_from_jpeg_dir_fnc,  sorted_img_file_names = None):
    if sorted_img_file_names is  None: 
        img_file_names = glob.glob(os.path.join(full_jpeg_dir,"*.jpeg"))
        sorted_img_file_names =  sorted(img_file_names, key = frame_from_jpeg_dir_fnc)
    # if frame_start is None: 
    frame_start = frame_from_jpeg_dir_fnc(sorted_img_file_names[0])# int(Path(sorted_img_file_names[0]).stem.split("_")[1])
   
    frame_end = frame_from_jpeg_dir_fnc(sorted_img_file_names[-1]) #int(Path(sorted_img_file_names[0]).stem.split("_")[-1])
    return frame_start, frame_end
frame_from_jpeg_dir_1 = lambda img_file_name: int(Path(img_file_name).stem.split("_")[1])
frame_from_jpeg_dir_2 = lambda img_file_name: int(Path(img_file_name).stem)

                                            

def load_frames_from_full_png_dir(full_jpeg_dir, frame_from_jpeg_dir_fnc,  frame_start = None, frame_end = None, frame_size = (120,120), color = True):
    img_file_names = glob.glob(full_jpeg_dir)#os.path.join(full_jpeg_dir,"*.png"))
    # print(img_file_names)
    # print("img_file_names[0]", img_file_names[0])
    sorted_img_file_names =  sorted(img_file_names, key = frame_from_jpeg_dir_fnc)#lambda img_file_name: int(Path(img_file_name).stem.split("_")[1]))
    # print("sorted_img_file_names", sorted_img_file_names)
    if frame_start is None: 
        frame_start, _ = get_frame_start_end_from_dir(full_jpeg_dir,frame_from_jpeg_dir_fnc,  sorted_img_file_names = sorted_img_file_names)
    if frame_end is None: 
        _, frame_end = get_frame_start_end_from_dir(full_jpeg_dir, frame_from_jpeg_dir_fnc, sorted_img_file_names = sorted_img_file_names)
    n_frames = int(frame_end-frame_start)
    if not color: 
        frames =  np.zeros((n_frames, frame_size[0], frame_size[1]))
    else: 
        frames =  np.zeros((n_frames, frame_size[0], frame_size[1], 3))
    # print(full_jpeg_dir)
    # print(sorted_img_file_names)
    dir_frame_start = frame_from_jpeg_dir_fnc(sorted_img_file_names[0]) #int(Path(sorted_img_file_names[0]).stem.split("_")[1])
    frame_start_i =  int(frame_start-dir_frame_start)
    frame_end_i = int(frame_start_i+n_frames)
    for i, frame_i in enumerate(range(frame_start_i, frame_end_i)):
        try:
            img_file_name = sorted_img_file_names[frame_i]
         
            if not color: 
                img = cv2.imread(img_file_name)[:,:,0]
            else: 
                img = cv2.imread(img_file_name)[:,:,:]
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            # Catching a more general exception if specific ones are not known
            print(f"A general error occurred: {e}")
            traceback.print_exc()
            if not color: 
                img = np.zeros(frame_size)
            else: 
                img = np.zeros((frame_size[0], frame_size[1], 3))
                    
        frames[i] = img
    return frames, frame_start, frame_end

def get_track_start_end_is_from_date_to_start_ends(date_to_start_ends, i = 0):
    track_to_date = {}
    all_tracks =[]
    all_starts = []
    all_ends =[]
    # i = 0
    for date, (starts, ends) in date_to_start_ends.items():
        all_tracks.append(np.ones(starts.shape[0])*i)
        all_starts.append(starts)
        all_ends.append(ends)
        track_to_date[i] = date
        i+=1
    all_tracks = np.concatenate(all_tracks)[:, None]
    all_starts = np.concatenate(all_starts)[:, None]
    all_ends = np.concatenate(all_ends)[:, None]
    return np.concatenate([all_tracks, all_starts, all_ends], axis = 1), track_to_date, i

def annotate_frames(frames, event_bin, font_scale = 0.4): #copy frames and annotate with beh class; velocity; curvature
    frames = copy.deepcopy(frames)
    annotated_frames = np.zeros_like(frames)
    for i, (frame, beh1)in enumerate(zip(frames,event_bin)):
        
                # Define the dimensions of the black area (e.g., 100 pixels wide and 50 pixels high)
        black_width = 30
        black_height = 30

        # Define the top-left and bottom-right points of the rectangle
        pt1 = (0, 0)  # Top-left corner of the image
        pt2 = (black_width, black_height) # Bottom-right corner of the black area

        # Draw a filled black rectangle
        frame = cv2.rectangle(frame, pt1, pt2, (0, 0, 0), -1) 
            


        if beh1: 
            dot_radius = 5
            # dot_color = (255, 0, 255)
            dot_color = (255, 255, 255)
            pos = (60, 60)
            frame = cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            
        
        annotated_frames[i] = frame
    return annotated_frames
def get_event_mat_from_track_start_end_is(track_start_end_is):
    track_start_end_is = track_start_end_is.astype('int')
    n_tracks = int(np.max(track_start_end_is[:, 0]))+1
    n_frames = int(np.max(track_start_end_is[:, 2]))+1

    
    event_mat = np.zeros((n_tracks,n_frames ))
    for track, start, end in track_start_end_is:
        event_mat[track, start:end] = 1
    return event_mat

def get_frames_and_annotate(clip_start_ends, event_mat,  full_jpeg_dirs, frame_from_jpeg_dir_fnc, frame_size = (120,120), color = True ):
    all_annotated_frames = []
    all_frames = []
    clip_start_ends = clip_start_ends.astype('int')
    # clip_start_ends_all_tracks = clip_start_ends_all_tracks.astype('int')
    
    for i, (track , start, end) in enumerate( clip_start_ends):
        # track_all_tracks, start_all_tracks, end_all_tracks = clip_start_ends_all_tracks[i]
        
        jpeg_dir = full_jpeg_dirs[i]
        frames, _ ,_ = load_frames_from_full_png_dir(jpeg_dir, frame_from_jpeg_dir_fnc,  frame_start = start, frame_end = end, frame_size = frame_size, color = color)
        # velocity = feature_to_vals["speed"][i]
        # event_mat = feature_to_vals["pause"][i]
        
        # label_to_beh = {}
        # annotated_frames = annotate_frames(frames, label_to_beh, beh, velocity, body_angle_mag)

        annotated_frames =  annotate_frames(frames, event_mat[track, start:end], font_scale = 0.4)
        all_annotated_frames.append(annotated_frames)
        all_frames.append(frames)
        
        # track_to_annotated_frames.append((frames, annotated_frames))
    return all_annotated_frames, all_frames
def filter_single_frames(track_start_ends, dur_min = 2):
    # print(track_start_ends.shape)
    durations = track_start_ends[:, 2]- track_start_ends[:, 1]
    dur_i = np.argwhere(durations>=dur_min).flatten()
    return track_start_ends[dur_i, :]



def get_grid_vids(save_dir_name, 
                  state_to_track_start_end_is,
                  exp_date_to_jpeg_dir,
                  track_to_date,
                  state_to_title_label,
                  max_clip_len,  
                  vid_fps, n_post_frames, n_pre_frames, vid_name_supp =""):
    
    for state, track_start_end_is in state_to_track_start_end_is.items():
        # motor_state, (start_state, end_state) = state_to_lbl[state]
        if track_start_end_is.size == 0: 
            continue

        clip_start_ends = get_clip_windows_from_start_end_is(track_start_end_is, n_pre_frames, n_post_frames, max_n_clips = 25)
        clipped_jpeg_dirs = [exp_date_to_jpeg_dir[track_to_date[track]] for track in clip_start_ends[:,0]]
        event_mat = get_event_mat_from_track_start_end_is(track_start_end_is)
        
        
        all_annotated_frames, all_frames = get_frames_and_annotate(clip_start_ends, event_mat,  clipped_jpeg_dirs, frame_from_jpeg_dir_1, frame_size =  (146,193), color = True )
        annotated_grid_frames = create_video_grid_color(all_annotated_frames, max_clip_len)
        vid_name = os.path.join(save_dir_name,f"[{state_to_title_label[state]}_ex_grid_{vid_name_supp}.mp4")
        make_vid_clip(annotated_grid_frames, vid_name, fps = vid_fps, quality=7)


