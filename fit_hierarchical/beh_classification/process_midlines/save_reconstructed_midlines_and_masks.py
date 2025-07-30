#get eigenworms using PCA 
import copy

import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import pickle
import copy
import numpy as np
import pandas as pd

import sys
import os 
import copy
import sys 

from functools import partial
sys.path.append('/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/')
from FileHandlers.OSHandler import OSHandler
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from get_data.read_neuro_beh_data_high_fps import get_exp_features as get_exp_features_fps
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification
from beh_classification.beh_classifications import is_pausing, is_forward, is_reversing,is_turning, get_reorientation_types, is_looping 

# from beh_analysis.behavior_features import convert_angles_to_coordinates, get_beh_classification
# from get_data.read_neuro_beh_data import get_processed_beh_features
from get_data.read_neuro_beh_data import get_exp_features 
# from get_data.flavell_raw_nir_extraction.convert_nir_to_confocal import convert_nir_to_confocal
from beh_classification.img_mask_specific_features import get_is_looping_from_mask_dir_confocal
from get_data.flavell_raw_nir_extraction.convert_nir_to_confocal import convert_nir_to_confocal,convert_confocal_to_nir,  evenly_sample_nir
from beh_classification.process_midlines.midline_utils import compute_spline_lengths, get_midlines_from_worm_tracer_csv, convert_coordinates_to_angles, resample_midline, convert_angles_to_coordinates
from beh_classification.process_midlines.eigenworm_utils import get_PCA_components_from_midlines, get_contributions_and_reconstruct_body_angles, get_cropped_midlines_eigenworms_and_body_angles
from save_img_utils.save_resampled_imgs import save_confocal_midlines_and_masks, evenly_sample_nir_and_save, evenly_sample_feature 

from save_img_utils.save_img_utils import get_text_nir_midlines, save_masked_imgs_and_annotations
from beh_classification.process_midlines.midline_utils import save_as_csv, load_from_csv
'''load in high nir data sampled at higher fps 
I was using this more before we got the raw imgs; now that we have the raw images the only really relative feature here is 
nir_to_confocal = beh_data_fps["nir_to_confocal"]
which maps the frames from the nir fps to the frames in the confocal fps

need to ignore  when beh_data["nir_to_confocal"] == 0
'''
exp_date = "2023-03-07-01"

raw_nir_midlines_dir = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/results/"
mask_img_dir = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/masked_imgs"

json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
h5_dir = "/Volumes/Extreme SSD/data/data_processed_high_fps/h5/"
beh_data_fps, neural_data, neuroID_to_key = get_exp_features_fps(exp_date, 
                                                            json_dir = json_dir, 
                                                            h5_dir = h5_dir, 
                                                            h5_suffix = "_output_data_dicts.h5" 
                                                            )
n_frames_confocal = 1600
nir_to_confocal = beh_data_fps["nir_to_confocal"].astype('int32')[1:]
n_frames_nir = nir_to_confocal.shape[0]




''''
intuition for nir_to_confocal: 

nir_to_confocal is a (n_nir_timestamps, ) with values 0-1600 indicating which timesteps of nir are avergage together to get confocal timesetep
values of 0 should be ignored (so np.argwhere(nir_to_confocal==i) are the timsetepsh that sould be coutns towards confocal timetep i-1)

'''
speed_stage_confocal = beh_data_fps["speed_stage"]
speed_stage_nir = beh_data_fps["nir_speed_stage"][1:]

print(nir_to_confocal)
fig, ax = plt.subplots()
ax.plot(nir_to_confocal)
ax.set_title("nir to confocal")

fig, ax = plt.subplots()
plt.plot(speed_stage_confocal)
plt.plot(convert_nir_to_confocal(nir_to_confocal, speed_stage_nir)[1:])






'''get self-intersection frames to be incorporated in to beh classificaion (for turns )'''
bin_img_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/bin_imgs/"

is_looping_bin_img_confocal, a_to_ps = get_is_looping_from_mask_dir_confocal(n_frames_nir, 
                                        n_frames_confocal
                                        ,nir_to_confocal, bin_img_dir, area_to_perimeter_ratio_thres = 1, area_to_perimeter = 6)

'''
load in data sampled at confocal fps
'''
json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
dt, T, beh_data_confocal, _, _ = get_exp_features(exp_date, 
                                                            json_dir = json_dir, 
                                                            h5_dir = h5_dir)
behavior_features = get_behavior_features(beh_data_confocal)
beh_data_confocal.update(behavior_features)

behavior_classification_confocal = get_behavior_classification(beh_data_confocal, n_looping_pts_thres = 0.3, looping = is_looping_bin_img_confocal)
z_confocal = behavior_classification_confocal["is_fwd"]+ 2*behavior_classification_confocal["is_rev"]+ 4*behavior_classification_confocal["is_pause"]+3*behavior_classification_confocal["is_turn"]-1     
z_confocal = z_confocal.astype('int32')
z_nir = convert_confocal_to_nir(nir_to_confocal, z_confocal, indices_to_skip = set([])).astype('int32')
z_nir = z_nir.astype('int32')
z_nir[nir_to_confocal==0] = 4
beh = [
    "fwd", 
    "rev",
    "turn", 
    "pause", 
    "nan"
]

beh_strs = [beh[i] for i in z_nir]
beh_strs_confocal = [beh[i] for i in z_confocal]


'''
save frames, midlines, splines, contributions for all nir frames 
'''
mask_annotations = [
list(range(1013, 1124)),  #  loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
list(range(7646, 7737)),  #  loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
list(range(12437, 12617)),  #  loopy and really doesnt caputure loopiness well.. can abstract awway as turn if want to.. 
list(range(16679, 16858)), #issue with stage -->  doesnt align to head super well
]
mask_annotations = np.concatenate([np.array(f) for f in mask_annotations])


midlines = get_midlines_from_worm_tracer_csv(raw_nir_midlines_dir)

ds = 0.02 
n_components = 8
num_resample_pts = 50


for target_length in [100]:#, 100, 115]:
    frames_to_mask = copy.deepcopy(mask_annotations)
    full_splines, to_keep, cropped_midlines,centroids,  midline_angles, mean_angles,  components, var_explained =  get_cropped_midlines_eigenworms_and_body_angles(
                                                        midlines, 
                                                        target_length, 
                                                        frames_to_mask,
                                                        ds = ds, 
                                                        n_components = n_components, 
                                                        num_resample_pts = num_resample_pts
                                                        )


    fig, ax = plt.subplots()
    ax.scatter(np.arange(0, np.cumsum(var_explained).shape[0]), np.cumsum(var_explained))
    ax.set_xlabel("PC")
    ax.set_ylabel("variance explained")
    ax.set_title(f"eigenworm components var explained;\n target_length {target_length}")
    reconstructed_body_angles, reconstructed_midlines, contributions_nir = get_contributions_and_reconstruct_body_angles(
                                                                        components, 
                                                                        midline_angles,
                                                                        mean_angles, 
                                                                        centroids, 
                                                                        target_length
                                                                        )
    
    '''
    get midlines cropped to target length specified 
    get eigenworms (PCAs of worm postures = body angles along spline) + from cropped midline 
    save pics of worms with overlain
        - spline fitted to midline
        - midlines cropped to target length specified  
        - reconstructed mdilen from eigenworm 
    
    plot explained var from eigenworm contributions
    '''
    masked = np.logical_not(to_keep)
    get_text_fnc = partial(
                            get_text_nir_midlines, 
                            nir_to_confocal = nir_to_confocal, 
                            state_strs = beh_strs, 
                            masked = masked, 
                            
                                
                                )
 
    output_dir = f"/Users/friederikebuck/Desktop/MBL/project/figs/{exp_date}/eigenworm_masked_091724/"
    n_frames = z_nir.shape[0]

    frames = np.arange(0, n_frames)
    save_masked_imgs_and_annotations(
        mask_img_dir, 
        frames, 
        output_dir, 
        get_text_fnc, 
        reconstructed_midlines, 
        z_nir,
        save_mask_img = False,
        masked_img_output_dir = None, 
        imgs = None
    )
    
    
    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/target_length"+str(target_length)+"contributions_nir.csv"
    np.savetxt(csv_name,  contributions_nir, delimiter=",", fmt='%f')


    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/target_length"+str(target_length)+"eigenworm_components.csv"
    np.savetxt(csv_name,  components, delimiter=",", fmt='%f')


    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/beh_confocal.csv"
    np.savetxt(csv_name,  z_confocal, delimiter=",", fmt='%f')
    
    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed{exp_date}/beh_nir.csv"
    np.savetxt(csv_name,  z_nir, delimiter=",", fmt='%f')
    

    masked_nir = copy.deepcopy(masked)
    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/target_length"+str(target_length)+"masked_nir.csv"
    np.savetxt(csv_name,  masked_nir, delimiter=",", fmt='%f')
    
    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/2023-03-07-01/target_length"+str(target_length)+"all_midlines_coords_nir.csv"
    save_as_csv(reconstructed_midlines, csv_name)

    '''
    resample features, midlines, etc into new time frame via interpolation
    (either to nir-> confocal time frame 
    of 
    nir-> new timeframe (ie for moseq because different numbers ie nir frames contribute to a single confocal frame 
    ie vals, counts = np.unique(nir_to_confocal, return_counts)
    np.all(np.diff(counts[1:])==0) = False
    )
    '''
    

    output_dir = f"/Users/friederikebuck/Desktop/MBL/project/figs/{exp_date}/eigenworm_masked_091724_confocal/"
    masked_img_output_dir =  f"/Users/friederikebuck/Desktop/MBL/project/figs/{exp_date}/eigenworm_masked_091724_confocal_masked_imgs/"
    cropped_midlines_confocal = save_confocal_midlines_and_masks(nir_to_confocal, 
                                reconstructed_midlines,
                                z_confocal,
                                mask_img_dir, 
                                output_dir,
                                masked_img_output_dir,
                                beh_strs_confocal
                                )
        
    time_bins=10
    output_dir = f"/Users/friederikebuck/Desktop/MBL/project/figs/{exp_date}/eigenworm_masked_091724_resampled_time_bin{time_bins}/"
    masked_img_output_dir =  f"/Users/friederikebuck/Desktop/MBL/project/figs/{exp_date}/eigenworm_masked_091724_resampled_masked_imgstime_bin{time_bins}/"
    midlines_resampled, beh_resampled, mask_resampled = evenly_sample_nir_and_save(nir_to_confocal, 
                                reconstructed_midlines,
                                z_nir, 
                                masked,
                                #features_nir, 
                                mask_img_dir,
                                output_dir, 
                                
                                time_bins=time_bins, 
                                
                                    save_mask_img = False,
                masked_img_output_dir = None, 
                )
    
    midlines_resampled1 =  evenly_sample_feature(reconstructed_midlines, 
                          nir_to_confocal,
                          time_bins=10, )
    beh_resampled1 =  evenly_sample_feature(z_nir, 
                          nir_to_confocal,
                          time_bins=10, ).flatten().astype('int')
    mask_resampled1 = evenly_sample_feature(masked, 
                          nir_to_confocal,
                          time_bins=10, ).flatten().astype('int')
    
    print(np.all(midlines_resampled.astype('uint8')==midlines_resampled1))
    print(np.all(beh_resampled ==beh_resampled1)) #True
    print(np.all(mask_resampled.astype('int') ==mask_resampled1))
    
    
    time_bins=1
    output_dir = f"/Users/friederikebuck/Desktop/MBL/project/figs/{exp_date}/eigenworm_masked_091724_resampled_time_bin{time_bins}/"
    masked_img_output_dir =  f"/Users/friederikebuck/Desktop/MBL/project/figs/{exp_date}/eigenworm_masked_091724_resampled_masked_imgstime_bin{time_bins}/"
    midlines_resampled, beh_resampled, mask_resampled = evenly_sample_nir_and_save(nir_to_confocal, 
                                reconstructed_midlines,
                                z_nir, 
                                masked,
                                #features_nir, 
                                mask_img_dir,
                                output_dir, 
                                
                                time_bins=time_bins, 
                                
                                    save_mask_img = False,
                masked_img_output_dir = None, 
                )
    
    midlines_resampled1 =  evenly_sample_feature(reconstructed_midlines, 
                          nir_to_confocal,
                          time_bins=time_bins)
    beh_resampled1 =  evenly_sample_feature(z_nir, 
                          nir_to_confocal,
                          time_bins=time_bins, ).flatten().astype('int')
    mask_resampled1 = evenly_sample_feature(masked, 
                          nir_to_confocal,
                          time_bins=time_bins, ).flatten().astype('int')
 
    print(np.all(midlines_resampled.astype('uint8') ==midlines_resampled1))#figureout why midlines_resampled is type uint8
    print(np.all(beh_resampled == beh_resampled1)) # True
    print(np.all(z_confocal == beh_resampled1)) #True
    print(np.all(mask_resampled ==mask_resampled1)) #True


    time_bins=10
    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/target_length"+str(target_length)+"contributions_nir.csv"
    eigenworm_contributions_nir = np.loadtxt(csv_name, delimiter=",")
    eigenworm_contributions_resampled = evenly_sample_feature(eigenworm_contributions_nir, 
                          nir_to_confocal,
                          time_bins=time_bins)
    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/target_length{target_length}contributions_resampled_{time_bins}.csv"
    save_as_csv(  eigenworm_contributions_resampled, csv_name)
    
    
    time_bins=1
    eigenworm_contributions_resampled = evenly_sample_feature(eigenworm_contributions_nir, 
                          nir_to_confocal,
                          time_bins=time_bins)
    csv_name = f"/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/{exp_date}/target_length{target_length}contributions_resampled_{time_bins}.csv"
    save_as_csv( eigenworm_contributions_resampled, csv_name)
