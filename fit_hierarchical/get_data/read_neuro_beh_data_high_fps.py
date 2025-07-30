#%% 
####extract behavioral features from 
#json file
import os 
import json
import math
import copy
import h5py 
import numpy as np
import matplotlib.pyplot as plt
def load_json(json_name):
    features = {}
    # behavior_features = [
    #     "velocity", 
    #     "pumping", 
    #     "angular_velocity", 
    #     "head_curvature"
    #     ]
    with open(json_name, 'r') as file:
        data = json.load(file)
        neuron_id_to_label = data["labeled"]

        labeled_trace_array = {(neuron_id_to_label[neuron_id]["neuron_class"], neuron_id_to_label[neuron_id]["label"]): data["trace_array"][int(neuron_id)-1] for neuron_id in data["labeled"].keys()}

        features["trace_array"]= labeled_trace_array
        features["avg_timestep"] = data["avg_timestep"]
    
        # beh_features = {beh: data[beh] for beh in behavior_features}
        
    return features #neuron to trace array (combines AVAL and AVAR if exists.. ehh not sure thats a good idea tbh   )


def get_processed_beh_features(h5_file):
    #['angular_velocity', 'body_angle', 'body_angle_absolute', 'body_angle_all', 'head_angle', 'pumping', 'reversal_events', 'reversal_vec', 'velocity', 'worm_curvature']
    features = {}
    with h5py.File(h5_file, 'r') as file:
        for feature in file.keys():
            features[feature] = file[feature][()]
    return features

def get_exp_features(exp_date, 
                     json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/", 
                     h5_dir = "/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/", 
                     h5_suffix = "_output_data_dicts.h5" 
                     ):
    
    json_name = os.path.join(json_dir, exp_date+".json" )
    h5_name = os.path.join(h5_dir, exp_date+h5_suffix )
    json_data = load_json(json_name)
    neural_data = json_data['trace_array']
    # dt = json_data["avg_timestep"]#s 
    beh_data = get_processed_beh_features(h5_name)
    neuroID_to_key = get_neuroID_to_key(neural_data)
    # T = beh_data["velocity"].shape[0]
    # return dt, T, beh_data, neural_data, neuroID_to_key
    return beh_data, neural_data, neuroID_to_key

def get_neuroID_to_key(neural_data):
        
    neuro_class_to_key = {neuron_class: (neuron_class, label) for neuron_class, label in neural_data.keys()}
    neuro_label_to_key =  {label: (neuron_class, label) for neuron_class, label in neural_data.keys()}
    neuro_label_to_key.update(neuro_class_to_key)
    ID_to_key = neuro_label_to_key
    return ID_to_key

if __name__ == "__main__":
    exp_date = "2023-03-07-01"
    json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
    h5_dir = "/Users/friederikebuck/Desktop/MBL/project/"
    data_str = "2023-03-07-01"

    h5_file = os.path.join(h5_dir,data_str+"_output_data_dicts.h5")
    features = get_processed_beh_features(h5_file)
    dt, T, beh_data, neural_data, neuroID_to_key = get_exp_features(exp_date, 
                                                                json_dir = json_dir, 
                                                                h5_dir = h5_dir)

    timesteps = np.arange(0, T)
    AIB = neuroID_to_key["AIB"]

    f, (neural_ax, beh_ax) = plt.subplots(2,  sharex=True)#, sharey=True)
    neural_ax.plot(timesteps,neural_data[AIB])
    beh_ax.plot(timesteps,beh_data["velocity"], c = "gray")
    beh_ax.axhline(y=0)
    neural_ax.axhline(y=0)
    # align_axes(beh_ax, neural_ax)
    f.subplots_adjust(hspace=0)

