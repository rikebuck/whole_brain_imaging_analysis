import numpy as np
from beh_classification.beh_classifications import is_pausing, is_forward, is_reversing,is_turning, get_reorientation_types, is_looping 
from beh_classification.behavior_features import get_behavior_features
from beh_classification.beh_classification_utils import convert_angles_to_coordinates

# def get_beh_classification_and_features(beh_data):
#     behavior_features = get_behavior_features(beh_data)
#     beh_data.update(behavior_features)
#     behavior_classification = get_behavior_classification(beh_data)
#     return beh_data, behavior_classification

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



def get_event_mat(behavior_classification):
    beh_class = [
        "is_turn", 
        "is_pause", 
        "is_rev", 
        "is_fwd"
    ]

    n_frames = behavior_classification["is_turn"].shape[0]
    beh_indices = [np.argwhere(np.array([behclass+str(behavior_classification[behclass][frame]).flatten()[0] 
                           for behclass in behavior_classification.keys()
                        ])
                )
     for frame in range(n_frames)]
    beh_event = [beh_class[beh_i] for beh_i in beh_indices]
    return beh_event

###from format_tracker_output/get_feature_mats.py 
def get_beh_classification_from_features(body_angle_absolute, midlines, velocity, body_angle_pts_org, n_looping_pts_thres = 0.5, looping = None):
    if looping is None: 
        looping = is_looping(body_angle_absolute, body_angle_pts_org, n_looping_pts_thres = n_looping_pts_thres)

    #get is turning
    is_turn, HT_angles, HM_vecs, TM_vecs, midpoints, heads, tails = is_turning(midlines, looping, min_duration_thres = 2)
    # is_pause = is_pausing(velocity)
    is_pause = np.zeros(is_turn.shape)
    is_pause[is_turn] = 0
    
    #get reversing 
    is_rev = is_reversing(velocity)
    is_rev[np.argwhere(is_pause)] = 0
    is_rev[np.argwhere(is_turn)] = 0
    
    #get fwds 

    
    #####if nessecary: 
    ####from BehaviorAnalyzer.format_tracker_output_mats.preprocess_tracks_mat.remove_single_frame_behs import remove_single_frame_behs
    ##### remove_single_frame_behs(all_tracks_feature_mat, feature_to_index, min_event_duration_thres)
   
    #this below shouldn't be nessecary? ...i think its something with the remove single frames 
    # rev_turn_overlap = np.where(np.logical_and(is_turn,is_rev))
    # is_turn[rev_turn_overlap] = 0

    is_revturn, is_purerev, is_pureturn = get_reorientation_types(is_rev, is_turn, max_transition_interval_frames = 6)
    is_turn[np.argwhere(is_revturn)] = 1
    is_rev_of_rev_turn = np.logical_and(is_revturn, is_rev)
    is_turn[np.argwhere(is_rev_of_rev_turn)] = 0
    
    is_fwd = is_forward(velocity)
    is_fwd[np.argwhere(is_pause)] = 0
    is_fwd[np.argwhere(is_turn)] = 0

    is_turn_of_rev_turn = np.logical_and(is_revturn, is_turn)
    # is_pure_rev = np.logical_and(np.logical_not(is_revturn), is_rev)
    # is_pure_turn = np.logical_and(np.logical_not(is_revturn), is_turn)
    ####if nessecary: 
    ##remove pausy tracks; 
    ##remove turny tracks: 

    ####if nessecary: 
    ##shift rows of matrix 
    ###covnert units from pixels to um or mm 
    ###crop border of feature mat 
    behavior_classification = {
        "is_turn": is_turn, 
        "is_pause": is_pause, 
        "is_rev": is_rev, 
        "is_fwd": is_fwd, 
        "is_revturn": is_revturn, 
        "is_purerev": is_purerev, 
        "is_pureturn": is_pureturn, 
        "is_rev_of_rev_turn":is_rev_of_rev_turn, 
        "is_turn_of_rev_turn": is_turn_of_rev_turn, 
        # "is_pure_rev": is_pure_rev, 
        # "is_pure_turn": is_pure_turn
        # "event_type":
    }
    return behavior_classification#,is_turn, HT_angles, HM_vecs, TM_vecs, midpoints, heads, tails
    
def get_behavior_classification(beh_data,  n_looping_pts_thres = 0.5, looping = None):
    # velocity = beh_data['velocity']
    
    # body_angles = beh_data['body_angle_absolute']
    # body_angles = body_angles-np.nanmean(body_angles, axis = 1)[:, None]
    # tail_is = get_tail_is(midlines)
    
    # midlines = convert_angles_to_coordinates(body_angles, distances=[])
    # centroids = np.nanmean(midlines, axis = 1)
    # midlines = midlines-centroids[:, None, :]

    #behavior_classification, long_enough_omegas, HT_angles, HM_vecs, TM_vecs, midpoints, heads, tails  = get_beh_classification_from_features(beh_data["body_angles"],
    behavior_classification = get_beh_classification_from_features(beh_data["body_angles"],
                                                                   beh_data["midlines"],
                                                                   beh_data["velocity"], 
                                                                   beh_data['body_angle_pts_org'], 
                                                                    n_looping_pts_thres =  n_looping_pts_thres , 
                                                                    looping = looping)
    return behavior_classification
    

# def get_behavior_features():
#     #     if "add_time_since_last_event_end" in augmentation_to_params:
#     #     for new_feature_label in augmentation_to_params["add_time_since_last_event_end"].keys():
#     #         event_labels = augmentation_to_params["add_time_since_last_event_end"][new_feature_label]
#     #         add_time_since_event_end_to_feature_mats(all_tracks_feature_mat,feature_to_index, event_labels = event_labels, new_feature_label = new_feature_label)
#     # if "add_time_since_last_event_start":
#     #     for new_feature_label in augmentation_to_params["add_time_since_last_event_start"].keys():
#     #         event_labels = augmentation_to_params["add_time_since_last_event_start"][new_feature_label]
#     #         add_time_since_event_start_to_feature_mats(all_tracks_feature_mat,feature_to_index, event_labels = event_labels, new_feature_label = new_feature_label)

#     # all_tracks_feature_mat, feature_to_index = flip_track_midlines_mat_based_on_eigenvalues(all_tracks_feature_mat, feature_to_index, eigenworms_txt, midline_smoothing_factor = midline_smoothing_factor)
    
#     # # if "add_body_angle_magnitude_to_track" in augmentation_to_params:
#     # #     add_body_angle_magnitude_to_track(all_tracks_feature_mat, feature_to_index)
 
#     # if "add_acceleration_track" in augmentation_to_params:
#     #     acceleration_timestep = augmentation_to_params["add_acceleration_track"]["acceleration_timestep"]
#     #     add_acceleration_to_track(all_tracks_feature_mat, feature_to_index, timestep = acceleration_timestep)
        
#     #     ####filter by acceleration ; speed etc 

#     # if "add_curvature_features" in augmentation_to_params:

#     #     angular_velocity_timestep = augmentation_to_params["add_not_dv_oriented_angular_velocity_to_track"]["angular_velocity_timestep"]
#     #     add_dv_oriented_angular_velocity_to_track(all_tracks_feature_mat, feature_to_index, timestep = angular_velocity_timestep)  
        
#     #     add_posture_area_to_track(all_tracks_feature_mat, feature_to_index)
        
#     #     add_body_angle_magnitude_to_track(all_tracks_feature_mat, feature_to_index)
        
#     #     # add_wavelength_to_track(all_tracks_feature_mat, feature_to_index)
#     #     add_wavelength_welch_to_track(all_tracks_feature_mat, feature_to_index)
   
#     #     # _, _, _, all_tracks_feature_mat, feature_to_index = get_my_Data_cluster_and_phi2_and_add_to_feature_mat(all_tracks_feature_mat,feature_to_index, costa_cluster_data_dir)#, add_to_feature_mat = False )
#     # else: 
#     #     print("!!!!not adding curvature feautres!!")
        

#     pass
    


# def get_eigen_worms():
#     pass 

# def get_wavelength():
#     pass

# def correct_midlines():
#     pass
