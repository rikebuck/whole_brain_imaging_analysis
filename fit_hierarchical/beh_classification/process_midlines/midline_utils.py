import os
import pandas as pd
import glob
import numpy as np
from scipy import interpolate


# def get_midline_length(midline_coords):
#     return np.sum(np.linalg.norm(np.diff(midline_coords, axis = 0),axis=1))

def save_as_csv(array, filename):
    # Reshape the array from (n_frames, n_pts, n_dim) to (n_frames, n_pts * n_dim)
    reshaped_array = array.reshape(array.shape[0], -1)  # (n_frames, n_pts * 2)
    
    # Save the reshaped array as a CSV file
    np.savetxt(filename, reshaped_array, delimiter=",")
    
def load_from_csv(filename, n_frames = None, n_pts = None, n_dim = 2):
    # Load the CSV file
    reshaped_array = np.loadtxt(filename, delimiter=",")
    if n_frames is None: 
        n_frames = reshaped_array.shape[0]
    if n_pts is None: 
        n_pts = int(reshaped_array.shape[1]/2)
    # Reshape back to the original shape (n_frames, n_pts, 2)
    original_shape_array = reshaped_array.reshape(n_frames, n_pts, n_dim)
    
    return original_shape_array

def convert_coordinates_to_angles(midline):

    diffs = midline[1:,:]-midline[0:-1,:]
    angles = np.arccos(diffs[:,1]/np.sqrt(np.sum(diffs**2,axis=1)))*np.sign(diffs[:,0])
    angles = np.unwrap(angles)
    mean_angle = np.mean(angles)
    mean_subtracted_angle = angles-np.mean(angles)
    return mean_subtracted_angle, mean_angle

def get_all_midline_lengths(midlines):
    
    return np.sum(np.linalg.norm(np.diff(midlines, axis = 1),axis=2), axis = 1)
    
def get_midlines_from_worm_tracer_csv(csv_dir, 
                                      is_midline_flipped = False,
                                       midline_flipped_frames =None 
                                      ):
    x_csv = glob.glob(os.path.join(csv_dir, "*_x.csv"))[0]
    y_csv = glob.glob(os.path.join(csv_dir, "*_y.csv"))[0]
    xs= pd.read_csv(x_csv).to_numpy() #cols are midlines rows and aframes 
    ys= pd.read_csv(y_csv).to_numpy()
    midlines = np.concatenate([xs[:,:,None], ys[:,:,None]], axis = 2)
    if is_midline_flipped:
        midlines = np.flip(midlines, axis =1)
    if midline_flipped_frames is not None: 
        for flipped_frames in midline_flipped_frames:
            midlines[flipped_frames] = np.flip( midlines[flipped_frames], axis =1)
    return midlines

def compute_spline_lengths(xy_coords, ds=0.1, smoothing = 0, target_length = 120*2/3):
    """
    Computes the length of splines for each frame of a time series of 2D coordinates.
    
    Parameters:
    - xy_coords: np.ndarray of shape (n_frames, n_pts, 2), where n_frames is the number
                 of frames, n_pts is the number of points per frame, and 2 represents the
                 x and y coordinates.
    - ds: float, step size for spline sampling.
    
    Returns:
    - spline_lengths: np.ndarray of length n_frames, containing the spline length for
                      each frame. Returns NaN for frames with insufficient valid points.
    """
    n_frames = xy_coords.shape[0]
    
    # List to store the lengths of the spline for each frame
    spline_lengths = []
    interpolated_midlines = []
    indices = []
    for frame in range(n_frames):
        # Extract xy coordinates for the current frame
        xy_frame = xy_coords[frame]
        
        # Remove rows where either x or y is NaN
        mask = ~np.isnan(xy_frame).any(axis=1)
        xy_frame_clean = xy_frame[mask]
        u_fine = np.arange(0, 1, ds)
        # print("xy_frame_clean.shape", xy_frame_clean.shape)
        # Proceed only if there are enough valid points
        if len(xy_frame_clean) > 1:  # Need at least 2 points to fit a spline
            # Fit the spline for the current frame
            tck, _ = interpolate.splprep([xy_frame_clean[:, 0], xy_frame_clean[:, 1]], s=smoothing)
            
            # Generate finely spaced u values for sampling the spline
            
            
            # Evaluate the spline at the fine intervals
            x_fine, y_fine = interpolate.splev(u_fine, tck)
            
            # Compute the spline length
            distances = np.sqrt(np.diff(x_fine)**2 + np.diff(y_fine)**2)
            length = np.sum(distances)
            cumulative_distances = np.cumsum(distances)
            index = np.searchsorted(cumulative_distances, target_length)
            if index >= len(x_fine):
                index = len(x_fine)-1
        else:
            length = np.nan  # Not enough points to compute the spline length
            x_fine, y_fine = np.zeros(u_fine.shape)*np.nan, np.zeros(u_fine.shape)*np.nan
            index = np.nan
            
        # Store the length for this frame
        spline_lengths.append(length)
        interpolated_midlines.append( np.concatenate([x_fine[:, None], y_fine[:, None]], axis = 1)[None, :, :])
        indices.append(index)
    interpolated_midlines = np.concatenate(interpolated_midlines, axis = 0)
    
    # Convert to array for easy manipulation
    return np.array(spline_lengths), interpolated_midlines, indices


def resample_midline(cropped_spline, num_resample_pts=50, final_length=None):
    """
    Resamples a given 1D or 2D spline to a specified number of points, 
    optionally adjusting to a final length.

    Parameters:
    - cropped_spline: np.ndarray, either 1D (shape (n_pts,)) or 2D (shape (n_pts, 2)) array of coordinates.
    - num_resample_pts: int, number of points to resample the spline.
    - final_length: float, the desired final length of the spline. If None, no adjustment for length is done.

    Returns:
    - resampled_midline: np.ndarray of shape (num_resample_pts, 2), resampled coordinates.
    """
    
    # Ensure the input is at least 2D for processing
    cropped_spline = np.atleast_2d(cropped_spline)
    
    if cropped_spline.shape[1] == 1:  # In case input is 1D, treat as a 2D array of coordinates
        cropped_spline = np.hstack([cropped_spline, np.zeros_like(cropped_spline)])
    
    # Get the x and y coordinates from the cropped spline
    x_coords = cropped_spline[:, 0]
    y_coords = cropped_spline[:, 1]
    
    # Fit a spline to the input coordinates
    tck_cropped, u = interpolate.splprep([x_coords, y_coords], s=0)
    
    # Resample the spline to `num_resample_pts` evenly spaced points
    u_resampled = np.linspace(0, 1, num_resample_pts)
    x_resampled, y_resampled = interpolate.splev(u_resampled, tck_cropped)
    
    # Resampled midline before applying final length adjustment
    resampled_midline = np.vstack([x_resampled, y_resampled]).T
    
    if final_length is not None:
        # Compute the current length of the resampled spline
        distances = np.sqrt(np.diff(x_resampled)**2 + np.diff(y_resampled)**2)
        current_length = np.sum(distances)
        
        # Scale the resampled midline to match the specified final length
        scale_factor = final_length / current_length
        resampled_midline[:, 0] *= scale_factor
        resampled_midline[:, 1] *= scale_factor
    
    return resampled_midline

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
