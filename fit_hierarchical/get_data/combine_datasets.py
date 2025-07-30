data_set_no_heat_strs = [

"2022-06-14-01",
"2022-06-14-07", 
"2022-06-14-13", 
"2022-06-28-01", 
"2022-06-28-07", 
"2022-07-15-06", 
"2022-07-15-12", 
"2022-07-20-01", 
"2022-07-26-01", 
"2022-08-02-01", 
"2023-01-23-08", 
"2023-01-23-15", 
"2023-01-23-21", 
"2023-01-19-01" ,
"2023-01-19-08", 
"2023-01-19-22" ,
# "2023-01-09-28", 
# "2023-01-17-01", 
"2023-01-19-15", 
"2023-01-23-01", 
"2023-03-07-01"

]
##combine worms 
def combine_data_sets(exp_dates,
                      json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/", 
                       h5_dir =  "/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"):
    neuroID_to_key_all = {}
    neural_data_all = {}
    beh_data_all= {}
    # dt, T, beh_data_all, neural_data_all, neuroID_to_key_all = get_exp_features(exp_dates[0], 
    #                                                     json_dir = json_dir, 
    #                                                     h5_dir = h5_dir)
    
    beh_data_keys = ['angular_velocity', 'head_angle', 'pumping',  'reversal_vec', 'velocity', 'worm_curvature'
        
    ]
    #skipped:  'body_angle','body_angle_absolute', 'body_angle_all','reversal_events', '
    for exp_date in exp_dates[1:]:
        dt, T, beh_data, neural_data, neuroID_to_key = get_exp_features(exp_date, 
                                                                json_dir = json_dir, 
                                                                h5_dir = h5_dir)
        print(beh_data["velocity"].shape)
        for key in beh_data_keys:

            if key in beh_data_all: 
                beh_data_all[key] = np.concatenate([beh_data_all[key],beh_data[key]] )
            else: 
                 beh_data_all[key] = beh_data[key]
        print(exp_date)
        print(beh_data_all["velocity"].shape)
        for key, val in neural_data.items():
            if key in neural_data_all: 
                neural_data_all[key] = np.concatenate([neural_data_all[key],neural_data[key]] )
            else: 
                neural_data_all[key] = neural_data[key]
        neuroID_to_key_all.update(neural_data_all)

    return  dt, T, beh_data_all, neural_data_all, neuroID_to_key_all

def combine_neuron_classes(neural_data):
    same_neuron_classes = {neuron:np.array([]) for neuron, ID in neural_data.keys() }
    for neuron, ID in neural_data.keys():
        same_neuron_classes[neuron]= np.concatenate([same_neuron_classes[neuron], neural_data[( neuron, ID)]])
    return same_neuron_classes
    
dt, T, beh_data_all, neural_data, neuroID_to_key = combine_data_sets(data_set_no_heat_strs,
                      json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/", 
                       h5_dir =  "/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/")


neural_data = combine_neuron_classes(neural_data)
