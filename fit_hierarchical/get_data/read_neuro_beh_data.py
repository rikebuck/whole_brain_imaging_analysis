#%% 
####extract behavioral features from 
#json file
import os 
import json
import h5py 
import numpy as np
import glob
# import matplotlib.pyplot as plt

# import sys
# sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collabs/")
# from beh_classification.get_behavior_classifications import get_behavior_classification
# from visualize_data.beh_class_colored_plots import make_behavior_ethogram, color_ax_by_beh_class
# from get_data.process_neural_data import get_derivative_of_neural_activity, smooth_trace

def load_json(json_name):
    features = {}
    with open(json_name, 'r') as file:
        data = json.load(file)
        neuron_id_to_label = data["labeled"]
        labeled_trace_array = {(neuron_id_to_label[neuron_id]["neuron_class"], neuron_id_to_label[neuron_id]["label"]): data["trace_array"][int(neuron_id)-1] for neuron_id in data["labeled"].keys()}
        features["trace_array"]= labeled_trace_array
        features["avg_timestep"] = data["avg_timestep"]
    
    return features #neuron to trace array (combines AVAL and AVAR if exists.. ehh not sure thats a good idea tbh   )


def get_processed_beh_features(h5_file):
    #['angular_velocity', 'body_angle', 'body_angle_absolute', 'body_angle_all', 'head_angle', 'pumping', 'reversal_events', 'reversal_vec', 'velocity', 'worm_curvature']
    features = {}
    with h5py.File(h5_file, 'r') as file:
        for feature in file['behavior'].keys():
            features[feature] = file['behavior'][feature][()]
    
    return features

def get_exp_features(exp_date, 
                     json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/", 
                     h5_dir = "/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
                     ):
    
    json_name = os.path.join(json_dir, exp_date+".json" )
    h5_name = glob.glob(os.path.join(h5_dir, exp_date+"*data*.h5" ))[0]
    json_data = load_json(json_name)
    neural_data = json_data['trace_array']
    dt = json_data["avg_timestep"]#s 
    beh_data = get_processed_beh_features(h5_name)
    n_frames, n_pts = beh_data["body_angle_absolute"].shape
    beh_data["body_angle_pts_org"] = np.ones(beh_data["velocity"].shape[0])*n_pts
    neuroID_to_key = get_neuroID_to_key(neural_data)
    T = beh_data["velocity"].shape[0]
    return dt, T, beh_data, neural_data, neuroID_to_key

def get_neuroID_to_key(neural_data):
        
    neuro_class_to_key = {neuron_class: (neuron_class, label) for neuron_class, label in neural_data.keys()}
    neuro_label_to_key =  {label: (neuron_class, label) for neuron_class, label in neural_data.keys()}
    neuro_label_to_key.update(neuro_class_to_key)
    ID_to_key = neuro_label_to_key
    return ID_to_key


def get_neural_activity_and_labels(neural_data):  # compare data all versus unlabeled stuff 
    Y = []
    X = []
    for neuron, trace in neural_data.items():
        X.append(neuron)
        Y.append(np.array(trace)[:, None])
    X = np.array(X)
    Y = np.concatenate(Y, axis = 1)
    return X, Y


    

# if __name__ == "__main__":
#     exp_date = "2023-03-07-01"
#     json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
#     h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
#     _, T, beh_data, neural_data, neuroID_to_key = get_exp_features(exp_date, 
#                                                                 json_dir = json_dir, 
#                                                                 h5_dir = h5_dir)
    
#     X, Y = get_neural_activity_and_labels(neural_data)
#     behavior_classification = get_behavior_classification(beh_data)
    
#     timesteps = np.arange(0, T)
#     neuron = "AIB"
#     AIB = neuroID_to_key[neuron]
#     print(AIB)
#     trace = neural_data[AIB]
#     denoised_data = smooth_trace(trace, sigma = 1)

#     fig, ax = plt.subplots()
#     dt = 0.6
#     time = np.arange(0, T*dt, dt)
#     ax.plot(time, trace, c ="darkgray", label = "raw trace")
#     ax.plot(time, denoised_data, c = "darkslateblue",  label = "smoothed trace")
#     ax.set_ylabel("z scored GcAMP fluorescence")
#     ax.set_xlabel("time (s)")
#     ax.set_title(neuron + " GcAMP")
    
#     color_time_series = make_behavior_ethogram(behavior_classification["is_fwd"], behavior_classification["is_rev"], behavior_classification["is_pause"],behavior_classification["is_turn"])
#     ax = color_ax_by_beh_class(ax, time, color_time_series)
#     ax.legend()
#     plt.show()