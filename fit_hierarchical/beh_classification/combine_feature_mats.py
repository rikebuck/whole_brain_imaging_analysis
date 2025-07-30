'''
combine exp feature mats 
'''
import os
import shutil
import sys 
import numpy as np
from pathlib import Path
import glob
import time
import matplotlib.pyplot as plt

sys.path.append('/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker')
sys.path.append('/home/fbuck/Large_Plate_Worm_Tracking/LargePlateWormTracker')
sys.path.append('/ru-auth/local/home/fbuck/store/LargePlateWormTracker')

from FileHandlers.YamlHandler import YamlHandler
from BehaviorAnalyzer.PlottingFunctions.plot_figs_single_exps import  get_plot_id_to_condition_to_colors
from BehaviorAnalyzer.analyse_tracks_all.load_and_save_feature_mats import load_feature_mat,get_feature_mat_pickle_name, save_feature_mat_pickles,filter_feature_mat_shaped_to_interval, filter_and_crop_feature_mat_all_to_interval
from FileHandlers.OSHandler import OSHandler
from FileHandlers.save_imgs_utils import save_jpeg_grid_as_pdf, save_jpegs_as_pdf
from FileHandlers.ZipHandler import ZipHandler
from BehaviorAnalyzer.get_params.PlottingParams import PlottingParams
from functools import partial
from BehaviorAnalyzer.plot_beh_analysis_figs import get_figs_list
from BehaviorAnalyzer.get_params.comp_specific_params import comp_specific_params
from BehaviorAnalyzer.ExpParams_arduino_protocol import get_exp_params
# from BehaviorAnalyzer.analyse_tracks_all.load_and_save_feature_mats import save_feature_mat_pickles,

from FileHandlers.OSHandler import OSHandler
def combine_feature_mats(
    event_label, 
    exp_configs, 
    valid_conditions, #set of okay conditions

    valid_strains = None,  #set of okay strains or none if all okay 
    # exp_ids = None,  #set of okay exp_ids or none if all okay 
    comp = "mac", 
    save_feature_mats = True, 
    check_saved_feature_mats = True,
    n_prior_s  = 15, 
    reshaped = False, 
    
):
    if comp =="mac":
        base_dir ="/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/"
        
    if comp =="linux":
        base_dir ="/home/fbuck/Large_Plate_Worm_Tracking/LargePlateWormTracker/"

    if comp == "hpc":
        base_dir = "/ru-auth/local/home/fbuck/store/LargePlateWormTracker/"
        
    comp_specific_params_yaml = os.path.join(base_dir,"BehaviorAnalyzer/yamls/comp_specific_params.yml")
    comp_params = comp_specific_params(comp_specific_params_yaml, comp)
    oshandler = OSHandler()
    all_mats = []
    pickle_files_to_n_tracks = {}
    all_plate_name_mats = []
    all_frame_num_mats = []
    all_trackID_mats = []
    for exp_yaml in exp_configs: 
        plotting_params = PlottingParams(os.path.join(comp_params.exp_yamls_dir, exp_yaml), comp = comp)
        params = get_exp_params(plotting_params, plotting_params.all_strains[0], n_prior_s)
        if valid_strains is not None: 
            strains = set(plotting_params.all_strains).intersection(valid_strains)
        else: 
            strains = plotting_params.all_strains
        all_conditions = plotting_params.all_conditions
        conditions = set(valid_conditions).intersection(set(all_conditions))
        print("conditions", conditions)
        for strain in strains: 
            for condition in conditions: 
                pickle_save_dir = oshandler.make_new_folder(os.path.join(plotting_params.save_dir,"pickles",  plotting_params.multiplot_pdf_suffix), "")
                exp_id = plotting_params.multiplot_pdf_suffix
                pickle_file = get_feature_mat_pickle_name(pickle_save_dir, exp_id, condition, strain)
                if save_feature_mats: 
                    
                # exp_yaml = "/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/BehaviorAnalyzer/yamls/exp_config_yamls/041424_RIMR_dutycycle.yml"##062124_2400uW_AIBpChrimson-1_duty_cycle.yml"
                    save_feature_mat_pickles(pickle_save_dir, n_prior_s, plotting_params,comp_params, check_saved_feature_mats = check_saved_feature_mats )
                    time.sleep(120)
                    
                feature_mats = load_feature_mat(pickle_save_dir, exp_id, condition, strain) 
                all_tracks_feature_mat, feature_to_index,all_feature_mats_reshaped, params, plate_num_to_plate_name = feature_mats
                if reshaped:
                    feature_mat = all_feature_mats_reshaped
                else:
                    feature_mat = all_tracks_feature_mat
                pickle_files_to_n_tracks[pickle_file] = feature_mat[feature_to_index["speed"]].shape[1]
                all_mats.append(feature_mat[feature_to_index[event_label]])
                all_frame_num_mats.append(feature_mat[feature_to_index["frame_time"]])
                all_trackID_mats.append(feature_mat[feature_to_index["trackID"]])
                
                plate_num_mat = feature_mat[feature_to_index["plate_num"]].astype('int')
                map_plate_num_to_plate_name = np.vectorize(lambda x: plate_num_to_plate_name.get(x, ""))
                plate_mat = map_plate_num_to_plate_name(plate_num_mat)
                all_plate_name_mats.append(plate_mat)
                
                
    return all_mats, all_plate_name_mats, all_frame_num_mats, all_trackID_mats, pickle_files_to_n_tracks
                


import numpy as np

def concatenate_with_padding(arrays):
    # Determine the max number of frames and the total number of tracks across all arrays
    max_frames = max(arr.shape[0] for arr in arrays)
    n_total_tracks = sum(arr.shape[1] for arr in arrays)
    extra_dims = arrays[0].shape[2:]  # Extract extra dimensions, assuming consistent shape beyond the first two

    # Check if any array contains strings
    if any(arr.dtype.kind in {'U', 'S'} for arr in arrays):  # 'U' and 'S' are dtype kinds for Unicode and ASCII strings
        fill_value = ""  # Use empty string if arrays contain strings
        output_array = np.full((max_frames, n_total_tracks) + extra_dims, fill_value, dtype=object)
    else:
        fill_value = np.nan  # Use NaN if arrays contain numbers
        output_array = np.full((max_frames, n_total_tracks) + extra_dims, fill_value)

    # Fill in each array's data into the output array
    track_start = 0
    for arr in arrays:
        n_frames, n_tracks = arr.shape[:2]
        output_array[:n_frames, track_start:track_start + n_tracks] = arr
        track_start += n_tracks

    return output_array



                # condition=  "atr0"#""atr0his0"
                # n_prior_s = 15

                # augmentation_to_params = plotting_params.augmentation_to_params
                # oshandler = OSHandler()
                # save_dir = plotting_params.save_dir
                # save_dir = oshandler.make_new_folder(os.path.join(save_dir, plotting_params.multiplot_pdf_suffix, "fwds_pause_state_analysis"), "")
                # save_pickle_suffix = plotting_params.multiplot_pdf_suffix+ "_n_prior_frames_"+str(n_prior_s)
                # all_strains = plotting_params.all_strains
                # strain = all_strains[0]
                # load_pickle = True
                # all_tracks_feature_mat, feature_to_index,all_tracks_feature_mat_reshaped, params, plate_num_to_plate_name = get_combined_features_mat_plotting_params(
                #                             plotting_params, 
                #                             strain
                #                             condition = condition,
                #                             load_pickle = load_pickle,

                #                             n_prior_s = n_prior_s, 
                #                             save_pickle_suffix = save_pickle_suffix, 
                                            
                #                             )
                
if __name__ == "__main__":
    exp_configs = [
        "041424_RIMR_dutycycle.yml", 
        "041724_RIMR_time_titration.yml"
    ]

    valid_conditions = [
        "atr0his0", "atr0", "his0atr0"
    ]

    # pickle_save_dir =
    speedmat1, all_plate_name_mats1, all_frame_num_mats1,all_trackID_mats1,  pickle_files_to_n_tracks = combine_feature_mats(
        "speed", 
        exp_configs, 
        valid_conditions, #set of okay conditions
        
        valid_strains = None,  #set of okay strains or none if all okay 
        comp = "mac", 
        save_feature_mats = True, 
        check_saved_feature_mats = True,
        n_prior_s  = 15, 
        reshaped = False, 
    )
    all_speed_mat = concatenate_with_padding(speedmat1)
    all_plate_name_mats = concatenate_with_padding(all_plate_name_mats1)
    all_frame_num_mats = concatenate_with_padding(all_frame_num_mats1)
    print()



    pausemat1, all_plate_name_mats1, all_frame_num_mats1,all_trackID_mats1,  pickle_files_to_n_tracks = combine_feature_mats(
        "pause", 
        exp_configs, 
        valid_conditions, #set of okay conditions
        
        valid_strains = None,  #set of okay strains or none if all okay 
        comp = "mac", 
        save_feature_mats = True, 
        check_saved_feature_mats = True,
        n_prior_s  = 15, 
        reshaped = False, 
    )
    all_pause_mat = concatenate_with_padding(pausemat1)

    turnmat1, all_plate_name_mats1, all_frame_num_mats1,all_trackID_mats1,  pickle_files_to_n_tracks = combine_feature_mats(
        "speed", 
        exp_configs, 
        valid_conditions, #set of okay conditions
        
        valid_strains = None,  #set of okay strains or none if all okay 
        comp = "mac", 
        save_feature_mats = True, 
        check_saved_feature_mats = True,
        n_prior_s  = 15, 
        reshaped = False, 
    )
    all_turn_mat = concatenate_with_padding(turnmat1)

    '''
    test by plotting ethgoram of individ and combines ** 

    '''
    from BehaviorAnalyzer.ethogram_functions.plot_ethogram_color_map import  plot_beh_speed_heat_map
    fig, speed_mat_etho, beh_etho = plot_beh_speed_heat_map(all_speed_mat, all_pause_mat, all_turn_mat, 

                                
                                )


    fig, speed_mat_etho, beh_etho = plot_beh_speed_heat_map(speedmat1[0], pausemat1[0],  turnmat1[0], 
                                        
                                )
    fig, speed_mat_etho, beh_etho = plot_beh_speed_heat_map(speedmat1[1], pausemat1[1],  turnmat1[1], 

                                        
                                )
    plt.show()

    print()




