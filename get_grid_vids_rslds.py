# %%
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
import sys
sys.path.append("/Users/friederikebuck/Downloads/worm notes/worm_code/")
T = 1599
_, _, full_beh_classification, full_beh_data = load_all_data()
import ssm
import pickle 
comp = "mac"
if comp == "hpc":
  project_dir = "/ru-auth/local/home/fbuck/scratch/WholeBrainImaging/keypt_moseq/projects/_1/"#sys.argv[1]+"_1"#/Volumes/FB1/moseq_project_dir/WholeBrainImagingAnalysis_1/"
  video_dir = "/ru-auth/local/home/fbuck/scratch/WholeBrainImaging/outputs_hpc/" #"/Users/friederikebuck/Desktop/MBL/project/outputs/"  #when al subdirs with vids or keypoints are stored 
  video_clip_dir = "/ru-auth/local/home/fbuck/scratch/WholeBrainImaging/outputs_hpc/vid_clips/"

else: 
#   project_dir = "/Volumes/FB1/moseq_project_dir_and_outputs/_1/"
# moseq_project_dir = ""
    project_dir = "/Volumes/FB1/moseq_project_dir_and_outputs/_1/"
    project_dir = "/Volumes/FB_data_ana/moseq_project_dir_and_outputs/_1/"
    video_dir = "/Users/friederikebuck/Desktop/MBL/project/outputs/" #"/Users/friederikebuck/Desktop/MBL/project/outputs/"  #when al subdirs with vids or keypoints are stored 
    video_clip_dir = "/Users/friederikebuck/Desktop/MBL/project/vid_clips/"
    save_dir = f"/Volumes/FB_data_ana/moseq_project_dir_and_outputs_figs/"

    # os.makedirs(save_dir, exist_ok=True)


# with open('/Users/friederikebuck/Desktop/MBL/project/keypt_moseq_coordinates_not_downsampled.pickle', 'rb') as handle:
#    coordinates =  pickle.load(handle)
   
   
kappas = np.logspace(3,7,5)
kappas = [kappas[0]]
decrease_kappa_factor = 10
num_ar_iters = 50
num_full_iters = 200
time_bin = 10
prefix = 'my_kappa_scan'
# for kappa in kappas: #kappas[:2]
kappa = kappas[0]
print(f"Fitting model with kappa={kappa}")
model_name = f'{prefix}-{kappa}'
# fig_save_dir = os.path.join(save_dir, model_name)

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


print(np.array(Y).shape)

print(len(neural_labels))



slds = global_worm
# get estimated latents, states
q_x = q.mean_continuous_states
q_z = [slds.most_likely_states(q_x[w], Y[w]) for w in range(len(q_x))]

z = [beh_classification["is_fwd"][0:1599]+ 2*beh_classification["is_rev"][0:1599]+ 4*beh_classification["is_pause"][0:1599]+3*beh_classification["is_turn"][0:1599]-1 for beh_classification in full_beh_classification]



# %%


# %%

### get state mat (ie 0-8 dependineg on rsdls stae end state and motor state)
### (later) label ethogram based on state ( tbh just mark single miss state + neural activity)
### from stae mat get state start ends 
###from state start ends get clip start ends  
###from track get jpeg dir
###create jpeg dir 

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
from make_grid_vid_fncs import get_clip_windows_from_start_end_is, create_video_grid ,make_vid_clip, create_video_grid_color


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
    print(img_file_names)
    print("img_file_names[0]", img_file_names[0])
    sorted_img_file_names =  sorted(img_file_names, key = frame_from_jpeg_dir_fnc)#lambda img_file_name: int(Path(img_file_name).stem.split("_")[1]))
    print("sorted_img_file_names", sorted_img_file_names)
    if frame_start is None: 
        frame_start, _ = get_frame_start_end_from_dir(full_jpeg_dir,frame_from_jpeg_dir_fnc,  sorted_img_file_names = sorted_img_file_names)
    if frame_end is None: 
        _, frame_end = get_frame_start_end_from_dir(full_jpeg_dir, frame_from_jpeg_dir_fnc, sorted_img_file_names = sorted_img_file_names)
    n_frames = int(frame_end-frame_start)
    if not color: 
        frames =  np.zeros((n_frames, frame_size[0], frame_size[1]))
    else: 
        frames =  np.zeros((n_frames, frame_size[0], frame_size[1], 3))
    print(full_jpeg_dir)
    print(sorted_img_file_names)
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
    print(track_start_ends.shape)
    durations = track_start_ends[:, 2]- track_start_ends[:, 1]
    dur_i = np.argwhere(durations>=dur_min).flatten()
    return track_start_ends[dur_i, :]

jpeg_dir_basename = "/Users/friederikebuck/Desktop/MBL/project/outputs/"
save_dir_name = "/Users/friederikebuck/Desktop/MBL/project/outputs_grid_vids_colored_by_motor_state/"

os.makedirs(save_dir_name, exist_ok=True)
# exp_to_jpeg_dir = lambda exp_date: os.path.join(jpeg_dir_basename,str(exp_date),"imgs", "masked_imgs_resampled_timebin_1_target_length85", "*png")
exp_to_jpeg_dir = lambda exp_date: os.path.join(jpeg_dir_basename,str(exp_date),"imgs1", "annotated_masked_imgs_resampled_timebin_1_target_length110", "*png")


time_bin = 1

json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
beh_data_dir_parent = "/Users/friederikebuck/Desktop/MBL/project/outputs/"
        
        
# with open(f'/Users/friederikebuck/Desktop/MBL/project/beh_syllables_rslds_{model_name}_all_exps.pickle', 'rb') as handle: ##off by 10..figureo ut in which directio n
#     a = pickle.load(handle)

#     exp_dates, exp_date_to_syllables, date_to_discrete_rslds_states, exp_date_to_beh,syllables, \
#     syllables_all_dates, rslds_states_all_dates, syllables_all_dates_flat, \
#     rslds_states_all_dates_flat, beh_all_dates_flat, beh_all_dates = a
    

directories = glob.glob(json_dir+"*.json") #get filename/paths of all the json files
exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates
exp_date_to_jpeg_dir = {exp_date: exp_to_jpeg_dir(exp_date) for exp_date in exp_dates} ###for not wusing 10sx eesmapled..s hould use confocal..? go from rslds to 20 


motor_state_to_rslds_state_to_exp_date_to_beh_start_end_is = {}
motor_states = [0,1,2]

fps = 1.66
n_pre_frames = int(fps*10)
n_post_frames = int(fps*30)
vid_fps = 6
max_clip_len = n_pre_frames+n_post_frames

for motor_state in motor_states:#[0,1,2]:
    
    
    rslds_state_to_exp_date_to_beh_start_end_is =  get_motor_state_start_end_is_rslds_start_end_specific(motor_state, exp_dates, z, q_z,  rslds_states = [0,1,2])
   
    motor_state_to_rslds_state_to_exp_date_to_beh_start_end_is[motor_state] = rslds_state_to_exp_date_to_beh_start_end_is
    i = 0 
    for rslds_pair, date_to_start_ends in rslds_state_to_exp_date_to_beh_start_end_is.items():
        print("motor_state", motor_state, "rslds_pair", rslds_pair)
        track_start_end_is, track_to_date, i = get_track_start_end_is_from_date_to_start_ends(date_to_start_ends, i = i )
        track_start_end_is = filter_single_frames(track_start_end_is, dur_min = 2)
        if track_start_end_is.size == 0: 
            continue

        clip_start_ends = get_clip_windows_from_start_end_is(track_start_end_is, n_pre_frames, n_post_frames, max_n_clips = 25)
        clipped_jpeg_dirs = [exp_date_to_jpeg_dir[track_to_date[track]] for track in clip_start_ends[:,0]]
        event_mat = get_event_mat_from_track_start_end_is(track_start_end_is)
        all_annotated_frames, all_frames = get_frames_and_annotate(clip_start_ends, event_mat,  clipped_jpeg_dirs, frame_from_jpeg_dir_1, frame_size =  (146,193), color = True )
        # grid_frames = create_video_grid(all_frames, max_clip_len)
        # grid_frames = create_video_grid_color(all_frames, max_clip_len)
        annotated_grid_frames = create_video_grid_color(all_annotated_frames, max_clip_len)
        vid_name = os.path.join(save_dir_name,f"{motor_state}_{rslds_pair}_ex_grid.mp4")
        print("vid_name", vid_name)
        make_vid_clip(annotated_grid_frames, vid_name, fps = vid_fps, quality=7)
        i+=1
        


# %%
clipped_jpeg_dirs

# %%
# @##Check 10x is 

import numpy as np

# Assuming 'your_file.csv' is a CSV file with numerical data
data = np.loadtxt('/Users/friederikebuck/Desktop/MBL/project/outputs/2022-06-28-01/csvs1/behavior_resampled_timebin_10.csv', delimiter=',')
print(data.shape)

data = np.loadtxt('/Users/friederikebuck/Desktop/MBL/project/outputs/2022-06-28-01/csvs1/behavior_resampled_timebin_1.csv', delimiter=',')
print(data.shape)


