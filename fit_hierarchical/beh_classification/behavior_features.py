import numpy as np
from scipy.ndimage import gaussian_filter1d
import copy
import sys 
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from beh_classification.beh_classification_utils import convert_angles_to_coordinates
def get_behavior_features(beh_data, time_interval = 1, acceleration_sigma =2 ):
    velocity = beh_data['velocity']
    
    body_angles = beh_data['body_angle_absolute']
    body_angles = body_angles-np.nanmean(body_angles, axis = 1)[:, None]
    # tail_is = get_tail_is(midlines)
    
    midlines = convert_angles_to_coordinates(body_angles, distances=[])
    centroids = np.nanmean(midlines, axis = 1)
    midlines = midlines-centroids[:, None, :]

    # body_angle_pts_org = beh_data['body_angle_pts_org']
    
    acceleration  = get_acceleration(velocity, time_interval, sigma = acceleration_sigma )
    
    beh_features = {
        "acceleration": acceleration, 
        "body_angles": body_angles, 
        "midlines": midlines, 
        "centroids": centroids, 
        
        #"angular_velocity": beh_data["angular_velocity"], 
        # beh_data["worm_curvature"], 
        # beh_data["pumping"], 
        # beh_data["head_angle"], 
    }
    return beh_features
    
def get_acceleration(signed_speed, time_interval, sigma = 2 ): # this works..
     # Standard deviation for Gaussian kernel
    denoised_data = gaussian_filter1d(signed_speed, sigma=sigma)

    # Step 2: Compute the derivative of the denoised data
    acceleration = (denoised_data[time_interval:]- denoised_data[:-time_interval])/time_interval

    tmp = np.ones(denoised_data.shape[0])*np.nan
    tmp[1:] = acceleration
    acceleration = tmp
    return acceleration
####from  get_track_features/get_angular_velocity.py -- see how compares to flavell version (although dont know the time step)

def get_angular_velocity(centroids, time_interval): # this works.. 
    '''#inputs: 
        centroids: float vector shape (n_frames , 2) of x,y positions of centroid at given frame and track 
        time interval: integer of the time interval over which to calculate the angular velocity

    
    outputs: 
        angular_velocity: float vector shape ((n_frames - time_interval/2) , 2) of angular velocity
    '''
    
    # Calculate the differences between consecutive points
    dx = np.diff(centroids[:, 0])
    dy = np.diff(centroids[:, 1])

    # Calculate the angle between consecutive points and "unwrap" adjusts values so they are outisde [ -pi, pi]
    angles = np.unwrap(np.arctan2(dy, dx)) 

    # Calculate the change in angle between frames
    angular_change = np.diff(angles) 
    
    #sum up the change in angles over time_interval and divide by the time interval amount (convolve  with 1's to get rolling sum of size time_interval)
    angular_velocity = np.convolve(angular_change, np.ones(time_interval), mode='valid')/ time_interval

    return angular_velocity


def add_dv_oriented_angular_velocity_to_track(all_tracks_feature_mat, feature_to_index, timestep = 6):
    # init new feature into all_tracks_feature_mat and feature_to_index
    all_tracks_feature_mat, feature_to_index =  add_new_feature_to_features_mat(all_tracks_feature_mat, feature_to_index, "signed_angular_velocity")
    
    #add angular velocity mat to all_tracks_feature_mat 
    centroids_mat = all_tracks_feature_mat[feature_to_index["centroids"]]
    unsigned_angular_velocity = get_angular_velocity_mat_from_centroids_mat(centroids_mat, timestep)
    
    #orient angular velocity
    to_flip_dv = all_tracks_feature_mat[feature_to_index["was_flipped_dv"]]
    to_flip_tracks = np.argwhere(np.any(to_flip_dv==1, axis = 0)).flatten()
    signed_angular_velocity = flip_angular_velocity(unsigned_angular_velocity, to_flip_tracks)
    all_tracks_feature_mat[feature_to_index["signed_angular_velocity"]] = signed_angular_velocity
            
    return all_tracks_feature_mat, feature_to_index


def flip_angular_velocity(angular_velocity_mat, to_flip_tracks):
    signed_angular_velocity_mat = copy.deepcopy(angular_velocity_mat)
    signed_angular_velocity_mat[:,to_flip_tracks ] = angular_velocity_mat[:,to_flip_tracks]*-1
    return signed_angular_velocity_mat

def add_dv_oriented_angular_velocity_to_track(all_tracks_feature_mat, feature_to_index, timestep = 6):
    # init new feature into all_tracks_feature_mat and feature_to_index
    all_tracks_feature_mat, feature_to_index =  add_new_feature_to_features_mat(all_tracks_feature_mat, feature_to_index, "signed_angular_velocity")
    
    #add angular velocity mat to all_tracks_feature_mat 
    centroids_mat = all_tracks_feature_mat[feature_to_index["centroids"]]
    unsigned_angular_velocity = get_angular_velocity_mat_from_centroids_mat(centroids_mat, timestep)
    
    #orient angular velocity
    to_flip_dv = all_tracks_feature_mat[feature_to_index["was_flipped_dv"]]
    to_flip_tracks = np.argwhere(np.any(to_flip_dv==1, axis = 0)).flatten()
    signed_angular_velocity = flip_angular_velocity(unsigned_angular_velocity, to_flip_tracks)
    all_tracks_feature_mat[feature_to_index["signed_angular_velocity"]] = signed_angular_velocity
            
    return all_tracks_feature_mat, feature_to_index


 ###from get_track_features/get_curvature.py
def signed_curvature_np_array(processed_splines_mat):
    if len(processed_splines_mat.shape)==2:
        processed_splines_mat = copy.deepcopy(processed_splines_mat)[None,:,:]
    # Calculate the first and second derivatives of the spline
    #frames, n_midline_pts, x/y
    xmat = processed_splines_mat[:,:,0]
    ymat = processed_splines_mat[:,:,1]
    dx = np.gradient(xmat, axis = 1)
    dy = np.gradient(ymat, axis = 1)
    ddx = np.gradient(dx, axis = 1)
    ddy = np.gradient(dy, axis = 1)
    
    # Calculate the curvature using the derivative values
    curvature = (dx * ddy - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)
    return curvature

def get_curvature_for_track_i_from_feature_mat(processed_splines):#, all_feature_mats, feature_to_index, n_midline_pts, frame_start = -1, frame_end = -1, smoothness = 0):
    # processed_midlines_angles, processed_splines, mean_angles = process_midline_at_track_i(track_i, all_feature_mats, feature_to_index, n_midline_pts = n_midline_pts, frame_start=frame_start, frame_end =frame_end, smoothness = smoothness)
    curvature = signed_curvature_np_array(processed_splines)
    return curvature

####from get_track_features/get_body_angle_mag.py
def add_body_angle_magnitude_to_track(all_tracks_feature_mat, feature_to_index):
    max_index = max([index for feature, index in feature_to_index.items() ])
    angles_mat = all_tracks_feature_mat[feature_to_index["dv_oriented_angles"]]
    feature_to_index["body_angle_mag"] = max_index+1
    max_index +=1
    all_tracks_feature_mat.append([])
    n_frames, n_tracks = all_tracks_feature_mat[feature_to_index["speed"]].shape
    all_tracks_feature_mat[feature_to_index["body_angle_mag"]] = np.zeros((n_frames,n_tracks))*np.nan
    for track_i in range(n_tracks):
        angles = angles_mat[:, track_i, :]
        body_angle_mag = np.linalg.norm(angles, axis = 1)
        all_tracks_feature_mat[feature_to_index["body_angle_mag"]][:, track_i] = body_angle_mag
        
    return all_tracks_feature_mat, feature_to_index


