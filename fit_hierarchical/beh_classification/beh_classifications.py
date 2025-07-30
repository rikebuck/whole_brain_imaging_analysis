import sys
import numpy as np
import copy 
import cv2
import os
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from beh_classification.beh_classification_utils import calculate_HT_angles, get_is_long_enough_track


def is_pausing(velocity, speed_thres = 0.009 ): 
    is_pausing = np.zeros(velocity.shape)
    is_pausing[np.abs(velocity)<speed_thres] = 1
    return is_pausing

def is_reversing(velocity):
    is_rev = np.zeros(velocity.shape)
    rev_beh_frames = np.argwhere(velocity<0)
    is_rev[rev_beh_frames] = 1
    return is_rev

def is_forward(velocity):
    is_fwd = np.zeros(velocity.shape)
    fwd_beh_frames = np.argwhere(velocity>0)
    is_fwd[fwd_beh_frames] = 1
    return is_fwd 

def is_looping(body_angle_absolute, body_angle_pts_org, n_looping_pts_thres = 0.3):
    #if 0.5 the pts are looping then call looping 

    n_nan_pts = np.sum(np.isnan(body_angle_absolute), axis = 1)
    is_looping = n_nan_pts>body_angle_pts_org*n_looping_pts_thres
    if len(body_angle_absolute.shape)==1:
        is_looping = is_looping.flatten()
    return is_looping


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
def get_tail_pts(oriented_midlines_track, tail_is):
    # n_frames, n_pts, n_dim = oriented_midlines_track.shape
    # tails = np.zeros((n_frames, n_dim))
    # for frame in range(n_frames):
    #     tails[frame, :]  = oriented_midlines_track[frame, tail_is[frame], :]
        # Use advanced indexing to directly extract the tail points
    tails = oriented_midlines_track[np.arange(oriented_midlines_track.shape[0]), tail_is, :]
    return tails

def is_turning(oriented_midlines_track, 
               is_looping, 
               min_duration_thres = 3):
    '''
    is _turnign if head-->mdpt (HM) vector from head to midline of spline is great tthan then 
    '''
    
    tail_is = get_tail_is(oriented_midlines_track)
    heads = oriented_midlines_track[:,0,:]
    tails = get_tail_pts(oriented_midlines_track, tail_is)
    n_midline_pts = oriented_midlines_track.shape[1]
    # mdpt_i = int(n_midline_pts/2)
    mdpt_i = np.floor(tail_is/2).astype('int32')
    midpoints = oriented_midlines_track[np.arange(oriented_midlines_track.shape[0]), mdpt_i, :]
    n_frames = oriented_midlines_track.shape[0]

    HT_angles = np.zeros(n_frames)
    HM_vecs = heads-midpoints
    TM_vecs = tails-midpoints
        
    HT_angles = calculate_HT_angles(HM_vecs, TM_vecs)

    is_narrow_angle = abs(HT_angles) < 45
    HM_vec_lengths = np.linalg.norm(HM_vecs, axis =1)
    TM_vec_lengths = np.linalg.norm(TM_vecs, axis =1)
    length_buffer = 20
    is_HM_greater_than_TM = HM_vec_lengths<TM_vec_lengths+length_buffer
    
    meets_omega_criteria = np.logical_and(is_narrow_angle,is_HM_greater_than_TM)
    meets_omega_criteria[is_looping] = 1
    short_events_bin_track, long_enough_omegas = get_is_long_enough_track(meets_omega_criteria, min_duration_thres, n_frames)
    return long_enough_omegas, HT_angles, HM_vecs, TM_vecs, midpoints, heads, tails


def get_reorientation_types(is_reversing_org, is_turning_org, max_transition_interval_frames = 6):
    ##### an issue with this is that this will also count a turn-rev as a rev-turn###how to deal with these..? or turns insdie revs; maye dont count trunts that are inside revs? ie too short turns or somethign? 
    #coudl just compare entry of frst rev and revturn an dsee if correct..as well as last rev and lst turn..

    rev_mat = copy.deepcopy(is_reversing_org)
    turn_mat = copy.deepcopy(is_turning_org)
    if len(is_reversing_org.shape)==1: 
        rev_mat =  copy.deepcopy(rev_mat[:, None])
    if len(is_turning_org.shape)==1: 
        turn_mat =  copy.deepcopy(turn_mat[:, None])
    
    n_frames, n_tracks = rev_mat.shape
    rev_mat[np.isnan(rev_mat)] = 0
    turn_mat[np.isnan(turn_mat)] = 0  

    #get indices where rev ends 
    rev_mat_buffered = np.vstack((np.zeros((1, n_tracks)), rev_mat, np.zeros((1, n_tracks))))
    rev_boundaries = np.diff(rev_mat_buffered.T, axis = 1)
    rev_ends_full = np.argwhere(rev_boundaries==-1)#first entry 
    
    #get indices where turn starts
    turn_mat_buffered = np.vstack((np.zeros((1, n_tracks)), turn_mat, np.zeros((1, n_tracks))))
    turn_boundaries = np.diff(turn_mat_buffered.T, axis = 1)
    rev_mat_transition = np.zeros((rev_mat.shape))
    
    #populate a binary matrix, transition_frames, that is 1s at the time post rev with a buffer period 
    for i in range(max_transition_interval_frames):
        
        #trnasiton frames must be occur before or at the last frame..
        transition_frames = copy.deepcopy(rev_ends_full)
        transition_track_is = transition_frames[:,0]
        transition_frame_is = np.minimum(transition_frames[:,1].flatten()+ i, np.ones(transition_frames[:,1].shape).flatten()*(n_frames-1)).astype('int16')
        rev_mat_transition[transition_frame_is, transition_track_is] = 1
        
    #overlap matrix is 1 only hwen a turn occurs during the transiton period (a certain time after the end of the rev)
    overlap_mat = np.logical_and(rev_mat_transition, turn_mat)
    
    #from here couldnt you just append the overlap mat and the rev mat; then just take 
    rev_turn_mat = np.zeros(rev_mat.shape)
    pure_rev_mat = np.zeros(rev_mat.shape)
    pure_turn_mat = np.zeros(rev_mat.shape)
    
    for track_i in range(n_tracks):
        
        rev_starts = np.argwhere(rev_boundaries[track_i,:]==1).flatten()
        rev_ends = np.argwhere(rev_boundaries[track_i,:]==-1).flatten()
        turn_ends = np.argwhere(turn_boundaries[track_i,:]==-1).flatten()
        
        pure_rev_starts = []
        pure_rev_ends = []

        revturn_starts = []
        revturn_ends = []
        
        for i in range(rev_starts.shape[0]):
            rev_start = rev_starts[i]
            rev_end = rev_ends[i]
            
            transition_interval_end = min(rev_end+max_transition_interval_frames, n_frames)
            if np.any(overlap_mat[rev_start:transition_interval_end, track_i]):
                revturn_start = rev_start
                revturn_end_i = np.argmin(turn_ends[turn_ends>=rev_end]  )
                revturn_end = turn_ends[turn_ends>=rev_end][revturn_end_i]
                rev_turn_mat[revturn_start:revturn_end, track_i] = 1
                
                revturn_starts.append(revturn_start)
                revturn_ends.append(revturn_end)
            else:
                
                pure_rev_mat[rev_start:rev_end, track_i ] =1 
                
                pure_rev_starts.append(rev_start)
                pure_rev_ends.append(rev_end)
        
        

        
        pure_turn_mat = copy.deepcopy(turn_mat)
        pure_turn_mat[rev_turn_mat ==1] = 0

    rev_turn_mat.astype('float32')[np.isnan(is_reversing_org)] = np.nan
    pure_rev_mat.astype('float32')[np.isnan(is_reversing_org)] = np.nan
    pure_turn_mat.astype('float32')[np.isnan(is_reversing_org)] = np.nan
    
    rev_turn_mat = rev_turn_mat.astype('uint8')
    pure_rev_mat = pure_rev_mat.astype('uint8')
    pure_turn_mat = pure_turn_mat.astype('uint8')
    
    if len(is_reversing_org.shape)==1: 
        rev_turn_mat = rev_turn_mat.flatten()
        pure_rev_mat = pure_rev_mat.flatten()
        pure_turn_mat = pure_turn_mat.flatten()

    return rev_turn_mat, pure_rev_mat, pure_turn_mat