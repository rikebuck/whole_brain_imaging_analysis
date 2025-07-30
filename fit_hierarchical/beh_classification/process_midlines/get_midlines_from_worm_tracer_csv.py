import os
import copy
import sys
import glob
import pickle
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/Users/friederikebuck/Downloads/LargePlateWormTracker_with_worm_tracer')
sys.path.append('/ru-auth/local/home/fbuck/store/LargePlateWormTracker_with_worm_tracer')

from GenerateBehTrack.WormShape.Worm_Midline import Worm_Midline
from GenerateBehTrack.WormShape.WormMorphology import WormMorphology
from FileHandlers.VideoHandlers.ImgVideoHandler import FrameVideoHandler
from FileHandlers.OSHandler import OSHandler
from VectorandImageHandlers.DebugImgHandler import DebugImgHandler

def get_midlines_from_worm_tracer_csv(csv_dir):
    x_csv = glob.glob(os.path.join(csv_dir, "*_x.csv"))[0]
    y_csv = glob.glob(os.path.join(csv_dir, "*_y.csv"))[0]
    xs= pd.read_csv(x_csv).to_numpy() #cols are midlines rows and aframes 
    ys= pd.read_csv(y_csv).to_numpy()
    midlines = np.concatenate([xs[:,:,None], ys[:,:,None]], axis = 2)
    midlines = np.flip(midlines, axis =1)
    return midlines

def get_midlines_from_bin_imgs(bin_img_dir, params_yaml,
                               frame_file_extension = ".png", is_looping =None):

    vid_handler = FrameVideoHandler(bin_img_dir, frame_file_extension)
    worm_midline_obj = Worm_Midline(params_yaml)
    print("worm_midline_obj.n_midline_pts", worm_midline_obj.n_midline_pts)
    #worm_midline_obj.n_midline_pts = n_midline_pts
    n_midline_pts = worm_midline_obj.n_midline_pts

    get_is_looping = is_looping is None
    ##eventually delete 


    n_frames = vid_handler.get_frame_count()
    print("n_frames", n_frames)
    # self.is_looping = np.zeros(n_frames)
    if get_is_looping:
        is_looping = np.zeros(n_frames)
    good_midline_frames = np.zeros(n_frames)
    unoriented_midlines = np.zeros((n_frames, n_midline_pts, 2))
    bad_frame_count = 0 
    # if start_frame is not None:
    #     vid_handler.set_frame_to_read(start_frame)#binimg dir starts at 0 
    for frame_i in range(n_frames):
        if frame_i%500 ==0:
            print(frame_i)
        worm_mask_img = vid_handler.imgGrab()
        if len(worm_mask_img.shape)!=2:
            bad_frame_count+=1
            continue
        if get_is_looping:
            wormorphology = WormMorphology(frame_i, worm_mask_img, params_yaml)
            wormorphology.thresholded_and_closed_worm = worm_mask_img
            is_looping[frame_i] = wormorphology.detect_loop()
        if is_looping[frame_i]:
            continue

        # blurred = cv2.blur(worm_mask_img, (3, 3)) 
        # blurred_bin = (blurred>50).astype('bool').astype('uint8')
        # good_coords_midline_coords_frame, worm_midline_coords, _, _ , _ = worm_midline_obj.get_midline_from_worm_mask(blurred_bin)
        try: 
            good_coords_midline_coords_frame, worm_midline_coords, _, _ , _ = worm_midline_obj.get_midline_from_worm_mask(worm_mask_img)    
            worm_midline_coords = worm_midline_obj.sample_midline_coords_evenly()
        except Exception as e: 
            print("!!!in midlines exception!!!!!!!!, bin img di, frame_i  ", bin_img_dir, frame_i)
            print(e)
            worm_midline_coords = 0
            good_coords_midline_coords_frame = False
        unoriented_midlines[frame_i,:,:] = worm_midline_coords
        good_midline_frames[frame_i] = good_coords_midline_coords_frame
    return unoriented_midlines, good_midline_frames, is_looping

def orient_midlines_based_on_worm_tracer(unoriented_midlines, worm_tracer_midlines):
    flipped_unoriented_midlines = np.flip(unoriented_midlines, axis =1)
    unflipped_midlines_dists = np.sum(np.linalg.norm(unoriented_midlines - worm_tracer_midlines, axis =2),axis =1)
    flipped_midlines_dists = np.sum(np.linalg.norm(flipped_unoriented_midlines - worm_tracer_midlines, axis=2), axis =1)
    
    to_flip_indices = np.argwhere(unflipped_midlines_dists<flipped_midlines_dists).flatten()
    
    oriented_midlines = copy.deepcopy(unoriented_midlines)
    oriented_midlines[to_flip_indices, :, :] = flipped_unoriented_midlines[to_flip_indices]
    
    return oriented_midlines

def insert_worm_tracer_midlines(oriented_midlines,worm_tracer_midlines, frames_to_insert):

    oriented_midlines[frames_to_insert, :,:] = worm_tracer_midlines[frames_to_insert, :,:]
    return oriented_midlines

        
def get_midlines(bin_img_dir, csv_dir, save_dir, params_yaml, 
                 # plot_midlines_bool = True,
                 start_frame = None, 
                 is_looping = None, 
                 pickle_save_dir = None, 
                 use_worm_tracker_midlines_only = True):
    os_handler = OSHandler()
    
    worm_tracer_midlines = get_midlines_from_worm_tracer_csv(csv_dir)

    unoriented_midlines, good_midline_frames, is_looping = get_midlines_from_bin_imgs(bin_img_dir, params_yaml,
                                                                                     
                                                                                      #start_frame = start_frame, 
                                                                                      is_looping = is_looping)
    unoriented_midlines = unoriented_midlines[1:,:,:]
    good_midline_frames = good_midline_frames[1:]
    is_looping = is_looping[1:]
    tracker_converter_mats = [unoriented_midlines, good_midline_frames, is_looping]
    if pickle_save_dir is None: 
        pickle_save_dir = save_dir
    os_handler.make_new_folder(pickle_save_dir, "")
    pickle_file = os.path.join(pickle_save_dir, "unoriented_midlines.pickle")
    with open(pickle_file, 'wb') as handle:
        pickle.dump(tracker_converter_mats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved pickle, ",pickle_file )

    print("use_worm_tracker_midlines_only", use_worm_tracker_midlines_only)
    if use_worm_tracker_midlines_only: 
        oriented_midlines = worm_tracer_midlines
        print("!!!!!! using use_worm_tracker_midlines_only!!!!")
    else: 
        frames_to_insert = np.logical_or(is_looping, np.logical_not(good_midline_frames))
        oriented_midlines = insert_worm_tracer_midlines(unoriented_midlines,worm_tracer_midlines, frames_to_insert)
        oriented_midlines = orient_midlines_based_on_worm_tracer(oriented_midlines, worm_tracer_midlines)
    
    midline_info = [oriented_midlines, unoriented_midlines, good_midline_frames, is_looping]
    pickle_file = os.path.join(pickle_save_dir, "oriented_midlines.pickle")
    with open(pickle_file, 'wb') as handle:
        pickle.dump(midline_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved pickle, ",pickle_file )
 
    midlines_save_dir = None
    # if plot_midlines_bool:
        
    #     midlines_save_dir = os.path.join(save_dir,"midlines")
    #     os_handler.make_new_folder(midlines_save_dir, "")
    #     frame_start = 1
    #     plot_midlines(oriented_midlines, frame_start, bin_img_dir, ".png", midlines_save_dir, params_yaml)
    return oriented_midlines, good_midline_frames, is_looping, midlines_save_dir 

if __name__ =="__main__":
    
    csv_dir = "/Users/friederikebuck/Desktop/WormTracking/wormtracking_tobackup/c1_082623_AVA_ret-1_0_2_bin_png/c1_082623_AVA_ret-1_0_2_bin_png/results"
    params_yaml = "/Users/friederikebuck/Downloads/LargePlateWormTracker_with_worm_tracer/behavior_config_yamls/beh_config.yaml"
    get_midlines_using_skeletonization = True
    is_looping = None
    bin_img_dir = "/Users/friederikebuck/Desktop/WormTracking/wormtracking_tobackup/c1_082623_AVA_ret-1_0_2_bin_png/c1_082623_AVA_ret-1_0_2_bin_png"
    oriented_midlines, good_midline_frames, is_looping = get_midlines(bin_img_dir, csv_dir, save_dir, params_yaml, 
                 plot_midlines_bool = True,
                 start_frame = None, 
                 is_looping = None)
    img_dir = "/Users/friederikebuck/Desktop/WormTracking/wormtracking_tobackup/c1_082623_AVA_ret-1_0_2_bin_png/c1_082623_AVA_ret-1_0_2_bin_png/WormTracer_first_pass_results/background_subtracted/"
    worm_tracer_midlines = get_midlines_from_worm_tracer_csv(csv_dir)
    plot_midlines(worm_tracer_midlines, 486, bin_img_dir, ".png", csv_dir, params_yaml)

    
    

