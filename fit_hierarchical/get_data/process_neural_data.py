import numpy as np
from scipy.ndimage import gaussian_filter1d

def get_derivative_of_neural_activity(neural_data, dframes = 1 ):
    #neural_data shape timestepsxn_neurons
    return neural_data[1:, :]- neural_data[:-1, :]

def smooth_trace(raw_trace, sigma = 2):
    denoised_data = gaussian_filter1d(raw_trace, sigma=sigma)
    return denoised_data


##get data 
# def get_neural_activity_and_labels_denoised(neural_data):  # compare data all versus unlabeled stuff 
#     Y = []
#     X = []
#     sigma = 2
    
#     for neuron, trace in neural_data.items():
#         X.append(neuron)
#         denoised_data = gaussian_filter1d(trace, sigma=sigma)
#         Y.append(np.array(denoised_data)[:, None])
#     X = np.array(X)
#     Y = np.concatenate(Y, axis = 1)
#     return X, Y
def get_neural_activity_and_labels(neural_data, denoised = False, sigma = 1):  # compare data all versus unlabeled stuff 
    Y = []
    X = []
    for neuron, trace in neural_data.items():
        X.append(neuron)
        if denoised: 
            denoised_data = gaussian_filter1d(trace, sigma=sigma)
            Y.append(np.array(denoised_data)[:, None])
        else: 
            Y.append(np.array(trace)[:, None])
    X = np.array(X)
    Y = np.concatenate(Y, axis = 1)
    return X, Y


def get_neural_activity_from_labels(neural_data, labels, neuroID_to_key, denoised = False, sigma = 1):  # compare data all versus unlabeled stuff 
    Y = []
    X = []
    for label in labels:
        X.append(label)
        
        neuron_key  = neuroID_to_key[label]
        trace = neural_data[neuron_key]
        if denoised: 
            denoised_data = gaussian_filter1d(trace, sigma=sigma)
            Y.append(np.array(denoised_data)[:, None])
        else:
            Y.append(np.array(trace)[:, None])
        
    X = np.array(X)
    Y = np.concatenate(Y, axis = 1)
    return X, Y

# def get_neural_activity_denoised_from_labels(neural_data, labels, neuroID_to_key):  # compare data all versus unlabeled stuff 
#     Y = []
#     sigma = 2
#     X = []
#     for label in labels:
#         neuron_key  = neuroID_to_key[label]
#         trace = neural_data[neuron_key]
#         denoised_data = gaussian_filter1d(trace, sigma=sigma)
#         X.append(label)
#         Y.append(np.array(denoised_data)[:, None])
#     X = np.array(X)
#     Y = np.concatenate(Y, axis = 1)
#     return X, Y