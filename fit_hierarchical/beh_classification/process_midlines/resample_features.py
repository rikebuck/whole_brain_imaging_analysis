#%%
#get eigenworms using PCA 
import copy

import numpy as np
import cv2
import matplotlib.pyplot as plt

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from shapely.geometry import LineString
import pickle
import copy
import numpy as np
import pandas as pd

import sys
import matplotlib.pylab as plt
import cv2

import glob 
import matplotlib

####extract behavioral features from 
#json file
import os 
import json
import numpy as np
from scipy.interpolate import interp1d, griddata
import math
import copy
import h5py 
import sys 
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('/Users/friederikebuck/Desktop/WormTracking/LargePlateWormTracker/')
from FileHandlers.OSHandler import OSHandler
sys.path.append('/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/')
from get_data.read_neuro_beh_data_high_fps import get_exp_features as get_exp_features_fps
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification
from beh_classification.beh_classifications import is_pausing, is_forward, is_reversing,is_turning, get_reorientation_types, is_looping 
from get_data.flavell_raw_nir_extraction.load_raw_frames import get_is_looping_from_mask
from beh_classification.process_midlines.get_eigenworms import load_from_csv, save_as_csv, compute_spline_lengths
# from beh_analysis.behavior_features import convert_angles_to_coordinates, get_beh_classification
# from get_data.read_neuro_beh_data import get_processed_beh_features
from get_data.read_neuro_beh_data import get_exp_features
import numpy as np

import numpy as np
from scipy.interpolate import interp1d
from PIL import Image
import os
from PIL import Image, ImageDraw, ImageFont
# def draw_points_on_image(draw, points, point_radius=1, point_color=(255, 0, 0)):
#     """
#     Draws a 2D array of points on an image.

#     :param points: A 2D numpy array of shape (n_pts, 2) where each row is (x, y) coordinates.
#     :param image_size: Size of the image as (width, height).
#     :param point_radius: Radius of the points to draw.
#     :param point_color: Color of the points, default is red (R, G, B).
#     :param output_path: Path to save the output image.
#     """
#     # Create a blank white image
#     # image = Image.new("RGB", image_size, (255, 255, 255))
#     # draw = ImageDraw.Draw(image)

#     # Loop through each point and draw it as a small circle on the image
#     for point in points:
#         x, y = point
#         # Draw a circle centered at (x, y)
#         draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)
#     return draw
def evenly_sample_nir_and_save(nir_to_confocal, 
                               midlines_nir,
                               beh_nir, 
                               mask_nir,
                               #features_nir, 
                               mask_img_dir,output_dir, 
                               
                               time_bins=10, ):
    indices = nir_to_confocal.astype('int32')[1:]

    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize frame counter for saving files
    frame_counter = 0
    midlines_resampled =[]
    beh_resampled = []
    mask_resampled = []
    

    palette = ["coral",     # forward
        "lightblue", # reverse
        "darkgreen", # turn
        "purple", # pause
        "navy" #nan
        ]    
        
        
        
    beh_to_color = {
        i: tuple(int(255 * v) for v in matplotlib.colors.to_rgb(c)) for i, c in enumerate(palette)
    }
    beh= [
        "fwd", 
        "rev", 
        "turn", 
        "pause", 
        "nan"
        
    ]
        
    # Loop through each unique index value
    for idx in np.unique(indices):
        if idx%50 == 0 : 
            print("idx", idx)
        if idx == 11: 
            print()
        imgs = []
        frames = np.argwhere(indices == idx).flatten()
        for frame in frames: 
            img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
            imgs.append(img[None, :, :])
        imgs = np.concatenate(imgs, axis = 0)
        
        # Get the features corresponding to the current index (frames)
       
        # Define the original time points (0 to len(current_features)-1)
        original_time = np.arange(frames.shape[0])  # shape: (n_frames,)

        # Define the new time points (interpolated)
        # new_time = np.linspace(0, original_time - 1, time_bins)
        new_time = np.linspace(0, frames.shape[0]- 1, time_bins) 

        # Perform interpolation across all frames and points
        f_interp_imgs = interp1d(original_time, imgs, axis=0, kind='linear', fill_value="extrapolate")
        # Get the interpolated values for the new frames
        interpolated_imgs= f_interp_imgs(new_time)
        # for feature_nir in features_nir: 
            
        #     # Perform interpolation across all frames and points
        #     f_interp_feature = interp1d(original_time, feature_nir, axis=0, kind='linear', fill_value="extrapolate")
        #     # Get the interpolated values for the new frames
        #     interpolated_feature= f_interp_feature(new_time)
        #     interpolated_features.append(interpolated_features)
        
        f_interp_feature = interp1d(original_time, midlines_nir[frames], axis=0, kind='linear', fill_value="extrapolate")
        # Get the interpolated values for the new frames
        interpolated_midlines = f_interp_feature(new_time)
        
        f_interp_beh = interp1d(original_time, beh_nir[frames], axis=0, kind='linear', fill_value="extrapolate")
        interpolated_beh= f_interp_beh(new_time)
        
        f_interp_mask = interp1d(original_time, mask_nir[frames], axis=0, kind='linear', fill_value="extrapolate")
        interpolated_mask= f_interp_mask(new_time)
        
        midlines_resampled.append(interpolated_midlines)#[None, :, :])
        beh_resampled.append(interpolated_beh)#[None, :])
        mask_resampled.append(interpolated_mask)#[None, :])
        
        
        beh_is = copy.deepcopy(interpolated_beh)
        beh_is[np.isnan(beh_is)] = 4
        beh_is = beh_is.astype('int')
        
        
        spline_lengths, interpolated_midlines, _ = compute_spline_lengths(interpolated_midlines, ds=0.1, smoothing = 0)
        # Save each interpolated frame as an image
        for i, interpolated_frame in enumerate(interpolated_imgs):
            # Reshape frame to (n_pts, 2) and normalize the pixel values to [0, 255]
            frame_image = (interpolated_frame - np.min(interpolated_frame)) / (np.max(interpolated_frame) - np.min(interpolated_frame)) * 255
            frame_image = frame_image.astype(np.uint8)
            
            # Convert to an image and save (you may need to reshape the frame if it's a single channel or specific format)
            image = Image.fromarray(frame_image)
            
            draw = ImageDraw.Draw(image)
   
  
            text = f"confocal frame: {idx}; nir frame: {frames[0]+new_time[i]} \n beh: {beh[beh_is[i]]}; mask: {interpolated_mask[i]} mask_nir: {mask_nir[int(frames[0]+new_time[i])]} \n length(px): {spline_lengths[i]}"
            
            # You can adjust the font, position, and size
            font = ImageFont.load_default()  # Use default font
            text_position = (0, 100)  
            text_color = (255,) * 3  # White text for grayscale images
            
            # Draw the text on the image
            draw.text(text_position, text, font=font, fill=text_color)

            draw = draw_points_on_image(draw, interpolated_midlines[i],  point_radius=1, point_color=beh_to_color[beh_is[i]]#(255, 0, 0)
                                        )
            # Save image to disk
            image.save(os.path.join(output_dir, f"frame_{frame_counter:04d}.png"))
            frame_counter += 1

    print(f"Saved {frame_counter} interpolated frames in '{output_dir}'")
    midlines_resampled = np.concatenate(midlines_resampled, axis = 0 )
    beh_resampled = np.concatenate(beh_resampled, axis = 0 )
    mask_resampled = np.concatenate(mask_resampled, axis = 0 )
    
    return midlines_resampled, beh_resampled, mask_resampled
    
    
    
csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/beh_nir.csv"
beh_nir = np.loadtxt(csv_name, delimiter=",")

csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/nir_to_confocal.csv"
nir_to_confocal =  np.loadtxt(csv_name
                                , delimiter=",") 


target_length =85
csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/processed_midlines/2023-03-07-01/target_length"+str(target_length)+"all_midlines_coords_nir.csv"
midlines_nir = load_from_csv(csv_name, n_frames = None, n_pts = None)
csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/target_length"+str(target_length)+"masked_nir.csv"
masked_nir = np.loadtxt(csv_name, delimiter=",") 



mask_img_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/masked_imgs/"

# output_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/processed_midlines/2023-03-07-01/resample_imgs_target_length"+str(target_length)+"/"
# OSHandler().make_new_folder(output_dir, "")
# midlines_resampled, beh_resampled, mask_resampled = evenly_sample_nir_and_save(nir_to_confocal, 
#                                                                                     midlines_nir,
#                                                                                     beh_nir,
#                                                                                     masked_nir,
#                                                                                     mask_img_dir, 
#                                                                                     output_dir, time_bins=10)


# save_as_csv(midlines_resampled, os.path.join(output_dir, "midlines_resampled.csv")) 
# np.savetxt(os.path.join(output_dir, "beh_resampled.csv"),  beh_resampled, delimiter=",", fmt='%f')
# np.savetxt(os.path.join(output_dir, "mask_resampled.csv"),  mask_resampled, delimiter=",", fmt='%f')



def save_confocal_masks(nir_to_confocal, 
                            midlines_nir,
                            beh_confocal,
                            mask_img_dir, 
                            output_dir,
                            masked_img_output_dir,
                            
                            time_bins=1):

    indices = nir_to_confocal.astype('int32')[1:]

    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize frame counter for saving files
    frame_counter = 0
    midlines_resampled =[]
    # beh_resampled = []
    # mask_resampled = []
    

    palette = ["coral",     # forward
        "lightblue", # reverse
        "darkgreen", # turn
        "purple", # pause
        "navy" #nan
        ]    
        
            # z_confocal = behavior_classification_confocal["is_fwd"]+ 2*behavior_classification_confocal["is_rev"]+ 4*behavior_classification_confocal["is_pause"]+3*behavior_classification_confocal["is_turn"]-1     
        
        
    beh_to_color = {
        i: tuple(int(255 * v) for v in matplotlib.colors.to_rgb(c)) for i, c in enumerate(palette)
    }
    beh= [
        "fwd", 
        "rev", 
        "turn", 
        "pause", 
        "nan"
        
    ]
    beh_is = copy.deepcopy(beh_confocal)
    beh_is[np.isnan(beh_is)] = 4
    beh_is = beh_is.astype('int')
    # Loop through each unique index value
    for idx in np.unique(indices):
        if idx ==0: 
            continue
        if idx%50 == 0 : 
            print("idx", idx)
        if idx == 11: 
            print()
        imgs = []
        frames = np.argwhere(indices == idx).flatten()
        for frame in frames: 
            img = cv2.imread(os.path.join(mask_img_dir, str(frame+1).zfill(5)+".png"))
            imgs.append(img[None, :, :])
        imgs = np.concatenate(imgs, axis = 0)
        
        # Get the features corresponding to the current index (frames)
       
        # Define the original time points (0 to len(current_features)-1)
        original_time = np.arange(frames.shape[0])  # shape: (n_frames,)

        # Define the new time points (interpolated)
        # new_time = np.linspace(0, original_time - 1, time_bins)
        new_time = np.linspace(0, frames.shape[0]- 1, time_bins) 

        # Perform interpolation across all frames and points
        f_interp_imgs = interp1d(original_time, imgs, axis=0, kind='linear', fill_value="extrapolate")
        # Get the interpolated values for the new frames
        interpolated_imgs= f_interp_imgs(new_time)

        f_interp_feature = interp1d(original_time, midlines_nir[frames], axis=0, kind='linear', fill_value="extrapolate")
        # Get the interpolated values for the new frames
        interpolated_midlines = f_interp_feature(new_time)

        midlines_resampled.append(interpolated_midlines)#[None, :, :])
        
        spline_lengths, interpolated_midlines, _ = compute_spline_lengths(interpolated_midlines, ds=0.1, smoothing = 0)
        # Save each interpolated frame as an image
        for i, interpolated_frame in enumerate(interpolated_imgs):

            # Reshape frame to (n_pts, 2) and normalize the pixel values to [0, 255]
            frame_image = (interpolated_frame - np.min(interpolated_frame)) / (np.max(interpolated_frame) - np.min(interpolated_frame)) * 255
            frame_image = frame_image.astype(np.uint8)
            
            # Convert to an image and save (you may need to reshape the frame if it's a single channel or specific format)
            image = Image.fromarray(frame_image)
            image.save(os.path.join(masked_img_output_dir, f"frame_{frame_counter:04d}.png")) # save blank masekd image 
            
            draw = ImageDraw.Draw(image)
   
  
            text = f"confocal frame: {idx}; nir frame: {frames[0]+new_time[i]} \n beh: {beh[beh_is[idx-1]]}; \n length(px): {spline_lengths[i]}"
            
            # You can adjust the font, position, and size
            font = ImageFont.load_default()  # Use default font
            text_position = (0, 100)  
            text_color = (255,) * 3  # White text for grayscale images
            
            # Draw the text on the image
            draw.text(text_position, text, font=font, fill=text_color)

            draw = draw_points_on_image(draw, interpolated_midlines[i],  point_radius=1, point_color=beh_to_color[beh_is[idx-1]]#(255, 0, 0)
                                        )
            # Save image to disk
            image.save(os.path.join(output_dir, f"frame_{frame_counter:04d}.png"))
            frame_counter += 1

    print(f"Saved {frame_counter} interpolated frames in '{output_dir}'")
    # midlines_resampled = np.concatenate(midlines_resampled, axis = 0 )
    # beh_resampled = np.concatenate(beh_resampled, axis = 0 )
    # mask_resampled = np.concatenate(mask_resampled, axis = 0 )
    
output_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/processed_midlines/2023-03-07-01/confocal_imgs_target_length"+str(target_length)+"/"
masked_img_output_dir = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/processed_midlines/2023-03-07-01/confocal_masked_imgs/"
csv_name = "/Volumes/Extreme SSD/data/raw_nir_frames_fully_processed/2023-03-07-01/beh_confocal.csv"

beh_confocal = np.loadtxt(csv_name, delimiter=",")
save_confocal_masks(nir_to_confocal, 
                        midlines_nir,
                        beh_confocal,
                        mask_img_dir, 
                        masked_img_output_dir,
                    output_dir, time_bins=1)


# save_as_csv(midlines_resampled, os.path.join(output_dir, "midlines_resampled.csv")) 
# np.savetxt(os.path.join(output_dir, "beh_resampled.csv"),  beh_resampled, delimiter=",", fmt='%f')
# np.savetxt(os.path.join(output_dir, "mask_resampled.csv"),  mask_resampled, delimiter=",", fmt='%f')

