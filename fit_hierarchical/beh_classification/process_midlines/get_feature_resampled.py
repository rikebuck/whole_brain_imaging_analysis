# %%
#%%
#get eigenworms using PCA 
import copy
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
# from shapely.geometry import LineString
import pickle
import copy
import numpy as np
import pandas as pd

import sys
import os 
import copy

from scipy.interpolate import interp1d

from functools import partial

'''change the directory below'''
import sys 
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')

from get_data.read_neuro_beh_data_high_fps import get_exp_features as get_exp_features_fps
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification

# from beh_analysis.behavior_features import convert_angles_to_coordinates, get_beh_classification
# from get_data.read_neuro_beh_data import get_processed_beh_features
from get_data.read_neuro_beh_data import get_exp_features 
# from get_data.flavell_raw_nir_extraction.convert_nir_to_confocal import convert_nir_to_confocal
from get_data.flavell_raw_nir_extraction.convert_nir_to_confocal import convert_nir_to_confocal,convert_confocal_to_nir,  evenly_sample_nir
from beh_classification.img_mask_specific_features import get_is_looping_from_mask_dir_confocal
from beh_classification.process_midlines.midline_utils import compute_spline_lengths, get_midlines_from_worm_tracer_csv, convert_coordinates_to_angles, resample_midline, convert_angles_to_coordinates
from beh_classification.process_midlines.midline_utils import save_as_csv, load_from_csv
from beh_classification.process_midlines.eigenworm_utils import get_PCA_components_from_midlines, get_contributions_and_reconstruct_body_angles, get_cropped_midlines_eigenworms_and_body_angles

from save_img_utils.save_img_utils import get_text_nir_midlines, save_masked_imgs_and_annotations
from save_img_utils.save_resampled_imgs import save_confocal_midlines_and_masks, evenly_sample_nir_and_save, evenly_sample_feature


# %%
def make_new_folder(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# %%
save_csvs = False
save_figs = False
save_imgs = True
save_mask_img = True

'''
change these directories
'''
base_save_dir = "/Users/friederikebuck/Desktop/MBL/project/outputs/"
base_code_dir = "/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/"
base_data_dir = "/Volumes/FB1/flavell_midlines/flavell_data_processed_midlines/noheat/"
# base_data_dir = "/Users/friederikebuck/Downloads/noheat/"
exp_params_yml ="/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/exp_date_params.yaml"

import glob
from pathlib import Path
def get_exp_dates(json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"):
    # vid_dirs = glob.glob(os.path.join(vid_parent_dir, "*"))
    # exp_dates=[Path(vid_dir).stem for vid_dir in vid_dirs]
    
    json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
    # h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"


    directories = glob.glob(json_dir+"*.json") #get filename/paths of all the json files
    exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates
    return exp_dates
    

vid_parent_dir = base_data_dir
exp_dates = get_exp_dates(json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/")
# exp_dates = [
#     '2022-07-15-12', 
#              '2022-06-14-07', 
#              '2022-06-14-01', 
#              '2022-08-02-01', 
#              '2023-01-09-28',
#              '2022-07-20-01',
#              '2023-01-23-21', 
#              '2022-06-14-13', 
#              '2022-07-15-06',
#              '2022-07-26-01', 
#              '2022-06-28-01',
#              '2023-01-23-08', 
#              '2023-01-23-01', 
#              '2023-01-17-01', 
#              '2023-03-07-01',
#              '2022-06-28-07', 
#              '2023-01-19-15',
#              '2023-01-19-22', 
#              '2023-01-19-01', 
#              '2023-01-19-08', 
#              '2023-01-23-15'
             
#              ]
for k, exp_date in enumerate(exp_dates): 
    print(k, exp_date)
    csv_save_dir = make_new_folder(os.path.join(base_save_dir,f"{exp_date}/csvs1/"))
    fig_save_dir = make_new_folder(os.path.join(base_save_dir, f"{exp_date}/figs1/"))
    img_save_dir = make_new_folder(os.path.join(base_save_dir, f"{exp_date}/imgs1/"))

    raw_nir_midlines_dir = os.path.join(base_data_dir, f"{exp_date}/results/") 
    mask_img_dir = os.path.join(base_data_dir, f"{exp_date}/masked_imgs/")
    bin_img_dir = os.path.join(base_data_dir, f"{exp_date}/bin_imgs/")
    mask_img_dir = bin_img_dir

    # %%
    json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
    h5_dir = "/Volumes/Extreme SSD/data/data_processed_high_fps/h5/"
    beh_data_fps, neural_data, neuroID_to_key = get_exp_features_fps(exp_date, 
                                                                json_dir = json_dir, 
                                                                h5_dir = h5_dir, 
                                                                h5_suffix = "_output_data_dicts.h5" 
                                                                )

    nir_to_confocal = beh_data_fps["nir_to_confocal"].astype('int32')[1:]

    csv_name = os.path.join(csv_save_dir, f"nir_to_confocal.csv")
    np.savetxt(csv_name,  nir_to_confocal, delimiter=",", fmt='%f')

    ''''
    intuition for nir_to_confocal: 

    nir_to_confocal is a (n_nir_timestamps, ) with values 0-1600 indicating which timesteps of nir are avergage together to get confocal timesetep
    values of 0 should be ignored (so np.argwhere(nir_to_confocal==i) are the timsetepsh that sould be coutns towards confocal timetep i-1)

    '''

    nir_features = [
       
        "nir_speed_stage",

       
        'nir_head_angle',
        'nir_pumping',
        'nir_velocity_stage',
        
        
    ]
    confocal_features = [
                          "speed_stage",
                          'head_angle',
                          'ventral_worm_curvature'
                         ]
    speed_stage_confocal = beh_data_fps["speed_stage"]
    speed_stage_nir = beh_data_fps["nir_speed_stage"][1:]
    
    # %%

    for feature in confocal_features:
        csv_name = os.path.join(csv_save_dir, f"{feature}_confocal.csv")
        save_as_csv(beh_data_fps[feature], csv_name)

    for time_bins in [1, 10]:

        for feature in nir_features: 
            # speed_resampled1 =  evenly_sample_feature(speed_stage_nir, 
            #                         nir_to_confocal,
            #                         time_bins=time_bins, )
            
            feature_resampled1 =  evenly_sample_feature(beh_data_fps[feature][1:],
                            nir_to_confocal,
                            time_bins=time_bins, )

                
                
            csv_name = os.path.join(csv_save_dir, f"{feature}_resampled_timebin_{time_bins}.csv")
            save_as_csv(feature_resampled1, csv_name)

