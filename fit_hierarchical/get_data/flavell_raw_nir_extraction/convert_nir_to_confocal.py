import numpy as np
def convert_nir_to_confocal(nir_to_confocal, feature_nir):
    indices = nir_to_confocal.astype('int32')
    features = feature_nir

    sums = np.bincount(indices, weights=features, minlength=1601)
    # Compute the count of each index
    counts = np.bincount(indices, minlength=1601)

    # Calculate the mean for each index
    feature_confocal  = sums / counts

    # Handle cases where count might be zero to avoid division by zero
    feature_confocal[np.isnan(feature_confocal)] = 0  
    return feature_confocal


def convert_confocal_to_nir(nir_to_confocal, feature_confocal, indices_to_skip = set([0])):#set([0])):
        
    # indices = nir_to_confocal.astype('int32')
    # # # indices1 = nir_to_confocal.astype('int32')-1
    # # # indices1[indices1==-1] = np.nan
    # # # indices[np.argwhere(~np.isnan(indices1)).flatten()]
    # # features = np.zeros(feature_confocal.shape[0]+1)*np.nan
    # # features[1:] = feature_confocal
    # features = np.zeros(feature_confocal.shape[0]+1)*np.nan
    # features[:-1] = feature_confocal

    # output_array = features[indices]
    # output_array[indices ==0] = np.nan
    # # output_array = output_array[indices!=0]
    # return output_array#features_nir

    indices = nir_to_confocal.astype('int32')
    features_nir = np.zeros(nir_to_confocal.shape[0])*np.nan
    for idx in np.unique(indices):
        if idx == 0 : 
            continue
        frames = np.argwhere(indices == idx).flatten()
        features_nir[frames] = feature_confocal[idx-1]
    return features_nir
    
        



def evenly_sample_nir(nir_to_confocal, feature_nir, time_bins = 10):
    indices = nir_to_confocal.astype('int32')
    features = feature_nir
    # Define the number of time bins for interpolation
    # Replace with your desired number of time bins
    # Create an empty list to store the interpolated feature arrays
    interpolated_features = []

    # Loop through each unique index value
    for idx in np.unique(indices):
        # Get the features corresponding to the current index
        current_features = features[indices == idx]
        
        # Define the original time points (0 to len(current_features)-1)
        original_time = np.arange(len(current_features))
        
        # Define the new time points (interpolated)
        new_time = np.linspace(0, len(current_features) - 1, time_bins)
        
        # Perform interpolation
        # f_interp = interp1d(original_time, current_features, kind='linear', fill_value="extrapolate")
        # interpolated_values = f_interp(new_time)
        
        # interpolated_values = griddata(original_time, current_features,new_time, method='linear')
        f_interp =interp1d(original_time, current_features, kind='linear', fill_value="extrapolate")  #griddata(original_time, current_features,new_time, method='linear')
        interpolated_values = f_interp(new_time)
        # Get the interpolated feature values

        
        # Append the interpolated values to the list
        interpolated_features.append(interpolated_values)

    # Convert the list of arrays into a single numpy array
    interpolated_features = np.concatenate(interpolated_features)
    return interpolated_features