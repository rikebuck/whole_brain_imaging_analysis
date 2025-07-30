import copy
import socket
import os
import pickle 
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')

from scipy.ndimage import gaussian_filter1d
from get_data.read_neuro_beh_data import get_exp_features, load_json
from get_data.process_neural_data import get_derivative_of_neural_activity, get_neural_activity_from_labels, get_neural_activity_and_labels
from beh_classification.behavior_features import get_behavior_features
from beh_classification.get_behavior_classifications import get_behavior_classification
# from beh_classification.beh_classification_utils import get_start_end_is_of_ones_in_binary_array

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA
# from sklearn import svm
# from sklearn.svm import LinearSVC
# from sklearn.neighbors import NearestNeighbors



# def smooth_trace(raw_trace, sigma = 2):
#     denoised_data = gaussian_filter1d(raw_trace, sigma=sigma)
#     return denoised_data

# def shift_array_by_one(array):
#     _, n_dim = array.shape
#     tmp = np.ones((T, n_dim))*np.nan
#     tmp[1:] = array
#     array = tmp
#     return array

def get_exp_dates():
    
    host = socket.gethostname()[0:6]
    if host == 'Bennet' or host == 'bennet' or 'northwestern' in socket.gethostname():
        json_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/Neuropal_no_heat/"
        h5_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/processed_h5/"
        beh_data_dir_parent = "/Users/bennetsakelaris/Desktop/behavior/outputs_sparse/"
    else:
        json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
        h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
        beh_data_dir_parent = "/Users/friederikebuck/Desktop/MBL/project/outputs/"



    directories = glob.glob(json_dir+"*.json") #get filename/paths of all the json files
    exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates
    # for i in range(len(exp_dates)):
    #     print("Worm {}, date: {}".format(i, exp_dates[i]))
        
    return exp_dates
    
# def load_all_data():
#     # this just returns lists of traces, neural labels, and behavior for each worm
#     # Also includes dtrace
#     # be warned that all elements of the list have different shapes

#     host = socket.gethostname()[0:6]
#     if host == 'Bennet' or host == 'bennet' or 'northwestern' in socket.gethostname():
#         json_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/Neuropal_no_heat/"
#         h5_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/processed_h5/"
#         beh_data_dir_parent = "/Users/bennetsakelaris/Desktop/behavior/outputs_sparse/"
#     else:
#         json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
#         h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
#         beh_data_dir_parent = "/Users/friederikebuck/Desktop/MBL/project/outputs/"



#     directories = glob.glob(json_dir+"*.json") #get filename/paths of all the json files
#     exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates
#     for i in range(len(exp_dates)):
#         print("Worm {}, date: {}".format(i, exp_dates[i]))

#     full_traces = []
#     full_neural_labels = []
#     full_beh_classification = []
#     full_beh_data = []

#     for exp_date in exp_dates:
#         dt, T, beh_data, neural_data, neuroID_to_key = get_exp_features(exp_date, 
#                                                                     json_dir = json_dir, 
#                                                                     h5_dir = h5_dir)
#         behavior_features = get_behavior_features(beh_data)
#         beh_data.update(behavior_features)
#         behavior_classification = get_behavior_classification(beh_data)

#         beh_data_dir = os.path.join(beh_data_dir_parent, str(exp_date), "csvs1")
#         csv_name = os.path.join(beh_data_dir, "beh_confocal.csv")
#         beh_confocal = np.loadtxt(csv_name, delimiter=",")
#         z_beh = beh_confocal
        
#         behavior_classification = {
#         "is_fwd": beh_confocal==0, 
#         "is_rev": beh_confocal==1, 
#         "is_turn": beh_confocal==2, 
#         "is_pause":  beh_confocal==3,
#         }

#         neural_labels_orig, ftrace = get_neural_activity_and_labels(neural_data,  denoised = True)

#         dtrace = get_derivative_of_neural_activity(smooth_trace(ftrace.T,sigma=5).T, dframes = 1)
#         dtrace -= np.mean(dtrace,axis=0)
#         dtrace /= np.std(dtrace,axis=0)
#         traces = np.concatenate([ftrace[1:,: ],dtrace ], axis =1)
#         neural_labels = np.array(["F - "+ lbl[0] for lbl in neural_labels_orig]+
#                                 ["dF - "+ lbl[0] for lbl in neural_labels_orig])
        
#         full_traces.append(traces)
#         full_neural_labels.append(neural_labels)
#         full_beh_classification.append(behavior_classification)
#         full_beh_data.append(beh_data)

#     return full_traces, full_neural_labels, full_beh_classification, full_beh_data
        
        

def convert_coordinates_to_angles(midline):

    diffs = midline[1:,:]-midline[0:-1,:]
    angles = np.arccos(diffs[:,1]/np.sqrt(np.sum(diffs**2,axis=1)))*np.sign(diffs[:,0])
    angles = np.unwrap(angles)
    mean_angle = np.mean(angles)
    mean_subtracted_angle = angles-np.mean(angles)
    return mean_subtracted_angle, mean_angle

# def get_body_angle_mag_from_midline(midlines):
#     body_angle_mag_all = []
#     body_angles = []
#     for midline in midlines: 
        
#         mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
        
#         body_angle_mags = np.linalg.norm(mean_subtracted_angles)#, axis = 1)
#         body_angle_mag_all.append(body_angle_mags)
#         body_angles.append(mean_subtracted_angles)
    
#     return np.array(body_angle_mag_all), np.array(body_angles)

def load_from_csv_midline(filename, n_frames = None, n_tracks = None, n_pts = None, n_dim = 2):
    # Load the CSV file
    reshaped_array = np.loadtxt(filename, delimiter=",")
    if n_frames is None: 
        n_frames = reshaped_array.shape[0]
    if n_pts is None: 
        n_pts = int(reshaped_array.shape[1]/2)
    # Reshape back to the original shape (n_frames, n_pts, 2)
    original_shape_array = reshaped_array.reshape(n_frames,n_tracks,  n_pts, n_dim)
    
    return original_shape_array

def get_body_angle_mag_from_midline_all_tracks(midlines,ratio=1):
    
    body_angles_all = []
    n_frames, n_tracks, n_total_pts, _  = midlines.shape
    body_angle_mag_all = np.zeros( (n_frames, n_tracks))
    for frame in range(n_frames):
        body_angles = []
        for track in range(n_tracks):
            midline = midlines[frame,track, :, : ]
        
            mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
            n_pts = int(ratio*n_total_pts)
            body_angle_mags = np.linalg.norm(mean_subtracted_angles[:n_pts])#, axis = 1)
            body_angle_mag_all[frame, track] = body_angle_mags
            body_angles.append(mean_subtracted_angles)
        body_angles_all.append(body_angles)
    return np.array(body_angle_mag_all), np.array(body_angles_all)


def get_body_angle_mag_from_midline(midlines, ratio=1):
    body_angle_mag_all = []
    body_angles = []
    for midline in midlines: 
        mean_subtracted_angles, mean_angle = convert_coordinates_to_angles(midline)
        n_total_pts = mean_subtracted_angles.shape[0]
        n_pts = int(ratio*n_total_pts)
        body_angle_mags = np.linalg.norm(mean_subtracted_angles[:n_pts])#, axis = 1)
        body_angle_mag_all.append(body_angle_mags)
        body_angles.append(mean_subtracted_angles)
    return np.array(body_angle_mag_all), np.array(body_angles)

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
def smooth_trace(raw_trace, sigma = 2):
    denoised_data = gaussian_filter1d(raw_trace, sigma=sigma)
    return denoised_data

def shift_array_by_one(array):
    _, n_dim = array.shape
    tmp = np.ones((T, n_dim))*np.nan
    tmp[1:] = array
    array = tmp
    return array

def get_midline_and_curvature_exp_date(exp_date, time_bin=1, target_length = 80):
    midline_dir  = f"/Users/friederikebuck/Desktop/MBL/project/outputs/{exp_date}/csvs1"
    midline_txt = os.path.join(midline_dir,f"cropped_midlines_wormtracer_target_length_{target_length}_timebin_{time_bin}.csv" )
    midlines = load_from_csv(midline_txt)
    body_angle_mag, body_angles = get_body_angle_mag_from_midline(midlines)
    return midlines,body_angles,  body_angle_mag

def load_all_data():
    # this just returns lists of traces, neural labels, and behavior for each worm
    # Also includes dtrace
    # be warned that all elements of the list have different shapes
    host = socket.gethostname()[0:6]
    if host == 'Bennet' or host == 'bennet' or 'northwestern' in socket.gethostname():
        json_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/Neuropal_no_heat/"
        h5_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/processed_h5/"
        beh_data_dir_parent = "/Users/bennetsakelaris/Desktop/behavior/outputs_sparse/"
    else:
        json_dir = "/Users/friederikebuck/Desktop/MBL/project/data/Neuropal_no_heat/"
        h5_dir ="/Users/friederikebuck/Desktop/MBL/project/data/processed_h5/"
        beh_data_dir_parent = "/Users/friederikebuck/Desktop/MBL/project/outputs/"


    directories = glob.glob(json_dir+"*.json") #get filename/paths of all the json files
    exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates
    for i in range(len(exp_dates)):
        print("Worm {}, date: {}".format(i, exp_dates[i]))

    full_traces = []
    full_neural_labels = []
    full_beh_classification = []
    full_beh_data = []

    for exp_date in exp_dates:
        dt, T, beh_data, neural_data, neuroID_to_key = get_exp_features(exp_date, 
                                                                    json_dir = json_dir, 
                                                                    h5_dir = h5_dir)
        behavior_features = get_behavior_features(beh_data)
        beh_data.update(behavior_features)
        midlines, body_angles,  body_angle_mag = get_midline_and_curvature_exp_date(exp_date, time_bin=1, target_length = 80)
        beh_data["body_angle_mag"] = body_angle_mag
        beh_data["midlines"] = midlines
        beh_data["body_angles"] = body_angles
        
        behavior_classification = get_behavior_classification(beh_data)

        beh_data_dir = os.path.join(beh_data_dir_parent, str(exp_date), "csvs1")
        csv_name = os.path.join(beh_data_dir, "beh_confocal.csv")
        beh_confocal = np.loadtxt(csv_name, delimiter=",")
        z_beh = beh_confocal
        
        behavior_classification = {
        "is_fwd": beh_confocal==0, 
        "is_rev": beh_confocal==1, 
        "is_turn": beh_confocal==2, 
        "is_pause":  beh_confocal==3,
        }

        neural_labels_orig, ftrace = get_neural_activity_and_labels(neural_data,  denoised = True)

        dtrace = get_derivative_of_neural_activity(smooth_trace(ftrace.T,sigma=5).T, dframes = 1)
        dtrace -= np.mean(dtrace,axis=0)
        dtrace /= np.std(dtrace,axis=0)
        traces = np.concatenate([ftrace[1:,: ],dtrace ], axis =1)
        neural_labels = np.array(["F - "+ lbl[0] for lbl in neural_labels_orig]+
                                ["dF - "+ lbl[0] for lbl in neural_labels_orig])
        
        full_traces.append(traces)
        full_neural_labels.append(neural_labels)
        full_beh_classification.append(behavior_classification)
        full_beh_data.append(beh_data)

    return full_traces, full_neural_labels, full_beh_classification, full_beh_data
        


def load_all_data_but_pretend_its_all_one_worm():
    # This function concatenates everything into one worm and also returns a mask that tells you which neurons are and arent present
    # Thing to ponder: adding columns of NaNs between worms would probably help the rSLDS learn better
    full_traces, full_neural_labels, full_beh_classification, full_beh_data = load_all_data()

    #get list of all neurons recorded in at least one trial
    neural_labels_set = set()
    for nl in full_neural_labels:
        neural_labels_set = neural_labels_set.union(set(nl))
    neural_labels = np.sort(list(neural_labels_set))

    #now make the traces array, where the different recordings are concatenated and matched by neuron
    #note: a couple of worms have 15 extra timesteps. I am truncating those to make things easier down the line
    T = 1599
    traces = np.zeros((T*len(full_traces), neural_labels.shape[0]))*np.nan #initialize traces to nan 
    #build the traces matrix
    w=0
    for tr, labels, in zip(full_traces, full_neural_labels): #for each worm, w
        for i in range(tr.shape[1]): #for each neuron, i
            label = labels[i] #get neuron name
            idx = np.where(neural_labels==label)[0][0] #get index in full array
            traces[w*T:(w+1)*T, idx] = tr[0:T,i] #put the neuron's activity in the appropriate spot
        w+=1 #update worm index

    

    #set up behavior classification dict
    behavior_classification = dict()
    behavior_classification["is_turn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_pause"] = np.zeros(T*len(full_traces))
    behavior_classification["is_rev"] = np.zeros(T*len(full_traces))
    behavior_classification["is_fwd"] = np.zeros(T*len(full_traces))
    behavior_classification["is_revturn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_purerev"] = np.zeros(T*len(full_traces))
    behavior_classification["is_pureturn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_rev_of_rev_turn"] = np.zeros(T*len(full_traces))
    behavior_classification["is_turn_of_rev_turn"] = np.zeros(T*len(full_traces))

    #fill it in
    w=0
    for bc in full_beh_classification:
        for key in bc.keys():
            behavior_classification[key][w*T:(w+1)*T] = bc[key][1:(T+1)] # shifting by one bc of the trace
        w+=1 #update worm index

    #return mask of nan data
    mask =  (~np.isnan(traces)).astype(int)
    return traces, neural_labels, behavior_classification, mask

# removes neurons that were only recorded in fewer than [threshold] experiments
def remove_rare_neurons(traces, neural_labels, mask, threshold=10): 
    n_occurances = np.zeros(traces.shape[1]) #count the number of occurances for each neuron
    for n in np.arange(0, traces.shape[1]):
        n_occurances[n] = sum(~np.isnan(traces[np.arange(100, traces.shape[0], 1599),n]))
    print("Removing {} neurons that are not recorded in more than {} experiments".format(sum(n_occurances<=threshold)/2, threshold))
    print("{} neurons remaining".format(sum(n_occurances>threshold)/2))
    
    traces = traces[:, n_occurances>threshold]
    neural_labels = neural_labels[n_occurances>threshold]
    mask = mask[:, n_occurances>threshold]
    return traces, neural_labels, mask


def states_to_changepoints(z):
    assert z.ndim == 1
    z = np.array(z)
    return np.concatenate(([0], 1 + np.where(np.diff(z))[0], [z.size - 1]))

# def plot_2d_continuous_states(x, z,
#                               colors=None,
#                               ax=None,
#                               inds=(0,1),
#                               figsize=(2.5, 2.5),
#                               **kwargs):
#     if colors is None: 
#         palette = ["coral",     # forward
#                 "lightblue", # reverse
#                 "darkgreen", # turn
#                 "purple","red", "yellow", "black", "pink"] 
#         colors = palette
#     if ax is None:
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(111)

#     cps = states_to_changepoints(z)

#     # Color denotes our inferred latent discrete state
#     for cp_start, cp_stop in zip(cps[:-1], cps[1:]):
#         ax.plot(x[cp_start:cp_stop + 1, inds[0]],
#                 x[cp_start:cp_stop + 1, inds[1]],
#                  '-', color=colors[z[cp_start]],
#                 **kwargs)

# def plot_dynamic_landscape(model,k,
#     xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
#     alpha=0.3, ax=None, figsize=(3, 3), color=None):
#     assert model.D == 2
#     K = model.K
#     x = np.linspace(*xlim, nxpts)
#     y = np.linspace(*ylim, nypts)
#     X, Y = np.meshgrid(x, y)
#     xy = np.column_stack((X.ravel(), Y.ravel()))

#     # Get the probability of each state at each xy location
#     try:
#         z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
#     except:
#         z = np.argmax(xy.dot(model.transitions.Rs.T), axis=1)

#     if ax is None:
#         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#     dxydt_m = xy.dot(model.dynamics.As[k].T) + model.dynamics.bs[k] - xy
#     dxydt_m_norm = np.zeros((len(dxydt_m)))
#     for xt in range(len(dxydt_m_norm)):
#         dxydt_m_norm[xt] = np.linalg.norm(dxydt_m[xt,:])

#     ax.plot_trisurf(xy[:,0], xy[:,1], dxydt_m_norm, cmap = "coolwarm", linewidth=0, antialiased=False)

#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')

#     plt.tight_layout()

#     return ax

# def plot_most_likely_dynamics(model, test_x, test_z, pca_x, palette = None,  other_pcs = False, input_str=0, input_id=0, ax=None, lim = None, plot_trajectory=True, inds=(0,1), do_pca=True, velocity=False):
#     ind1,ind2 = inds
#     pca = PCA(n_components=pca_x.shape[1])
#     x_centered = pca_x - np.mean(pca_x, axis=0)
#     pca.fit(x_centered)  # X is your trajectory data matrix
#     X_pca = pca.transform(test_x - np.mean(pca_x, axis=0) )
    
#     K = model.K
#     # Define grid limits based on PCA-transformed trajectory
    
#     if palette is None: 
#         palette = ["coral",     # forward
#                 "lightblue", # reverse
#                 "darkgreen", # turn
#                 "purple","red", "yellow", "black", "pink"] 

        
#     if lim == None:
#         x_min, x_max = X_pca[:, ind1].min() - 3, X_pca[:, ind1].max() + 3
#         y_min, y_max = X_pca[:, ind2].min() - 3, X_pca[:, ind2].max() + 3
#         lim = [x_min, x_max, y_min, y_max]
#     else:
#         x_min = lim[0]
#         x_max = lim[1]
#         y_min = lim[2]
#         y_max = lim[3]

#     # Create a grid of points
#     x = np.linspace(x_min, x_max, 30)  # 30 grid points along x-axis
#     y = np.linspace(y_min, y_max, 30)  # 30 grid points along y-axis
#     X_grid, Y_grid = np.meshgrid(x, y)  # Create the grid

#     # Inverse transform to get high-dimensional coordinates of grid points
#     xy = np.column_stack(pca_x.shape[1]*[0*Y_grid.ravel()])
#     xy[:,ind1] = X_grid.ravel()
#     xy[:,ind2] = Y_grid.ravel()

#     if (np.abs(input_str) > 0) and other_pcs:
#         for i in range(xy.shape[1]):
#             if (i != ind2) and (i!= ind1):
#                 xy[:,i] = X_pca[:,i].mean() + 0*Y_grid.ravel()

#     if do_pca:
#         xy_high_dim = pca.inverse_transform(xy)  # Map back to original space
#     else:
#         xy_high_dim = xy

#     # Get the probability of each state at each xy location
#     try:
#         z = np.argmax(xy_high_dim.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)
#     except:
#         z = np.argmax(xy_high_dim.dot(model.transitions.Rs.T), axis=1)

#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(111)

#     velocity_magnitudes = np.zeros_like(X_grid.ravel())

#     if model.M == 0:
#         for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
#             next_step_high_dim = xy_high_dim @ A.T + b  # x_t+1 = Ax_t + b
#             # Map back to PCA space for visualization
#             if do_pca:
#                 next_step_pca = next_step_high_dim @ pca.components_.T
#             else:
#                 next_step_pca = next_step_high_dim

#             zk = z == k
#             if zk.sum(0) > 0:
#                 ax.quiver(xy[zk, ind1], xy[zk, ind2],
#                         next_step_pca[zk, ind1]-xy[zk,ind1], next_step_pca[zk, ind2]-xy[zk,ind2],
#                         color=palette[k % len(palette)], alpha=1)
#             velocity_magnitudes[zk] = np.linalg.norm(next_step_pca[zk] - xy[zk], axis=1)
#     else:
#         for k, (A, b, V) in enumerate(zip(model.dynamics.As, model.dynamics.bs, model.dynamics.Vs)):
#             next_step_high_dim = xy_high_dim @ A.T + b + input_str*V[:,input_id] # x_t+1 = Ax_t + b
#             # Map back to PCA space for visualization
#             if do_pca:
#                 next_step_pca = next_step_high_dim @ pca.components_.T
#             else:
#                 next_step_pca = next_step_high_dim

#             zk = z == k
#             if zk.sum(0) > 0:
#                 ax.quiver(xy[zk, ind1], xy[zk, ind2],
#                         next_step_pca[zk, ind1]-xy[zk,ind1], next_step_pca[zk, ind2]-xy[zk,ind2],
#                         color=palette[k % len(palette)], alpha=1)
#                 velocity_magnitudes[zk] = np.linalg.norm(next_step_pca[zk] - xy[zk], axis=1)
            
#     if velocity:
#         ax.cla()
#          # Reshape velocity magnitudes to match grid
#         velocity_magnitudes = velocity_magnitudes.reshape(30, 30)

#         if ax is None:
#             fig, ax = plt.subplots(figsize=(6, 6))

#         # Plot the velocity heatmap
#         heatmap = ax.pcolormesh(X_grid, Y_grid, velocity_magnitudes, cmap="Purples", shading='auto')
#         plt.colorbar(heatmap, ax=ax, label="Velocity Magnitude")
                
#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')

#     plt.tight_layout()
#     if plot_trajectory:
#         if do_pca:
#             plot_2d_continuous_states(X_pca, test_z, ax=ax, inds=inds)
#         else:
#             plot_2d_continuous_states(test_x, test_z, ax=ax, inds=inds)
#     ax.set_xlim(x_min+2, x_max-2)
#     ax.set_ylim(y_min+2, y_max-2)

#     return ax, lim

def moving_average_matrix(matrix, window_size=3):
    # Apply moving average to each column (i.e., smooth over time for each neuron)
    smoothed_matrix = np.copy(matrix)
    for i in range(matrix.shape[1]):  # Iterate over neurons (columns)
        smoothed_matrix[:, i] = np.convolve(matrix[:, i], np.ones(window_size)/window_size, mode='same')
    return smoothed_matrix

def load_data_from_folder(directories, json_dir):

    exp_dates = [directory.split("/")[-1].split(".")[0] for directory in directories] #extract the exp_dates

    full_traces = []
    full_neural_labels = []
    for exp_date in exp_dates:
        json_name = json_dir + exp_date + ".json"
        json_data = load_json(json_name)
        neural_data = json_data['trace_array']

        neural_labels_orig, ftrace = get_neural_activity_and_labels(neural_data,  denoised = True)

        dtrace = get_derivative_of_neural_activity(smooth_trace(ftrace.T,sigma=5).T, dframes = 1)
        dtrace -= np.mean(dtrace,axis=0)
        dtrace /= np.std(dtrace,axis=0)
        traces = np.concatenate([ftrace[1:,: ],dtrace ], axis =1)
        neural_labels = np.array(["F - "+ lbl[0] for lbl in neural_labels_orig]+
                                ["dF - "+ lbl[0] for lbl in neural_labels_orig])
        
        full_traces.append(traces)
        full_neural_labels.append(neural_labels)
    return full_traces, full_neural_labels


# def load_all_neural_data_heat():
#     json_dirs = {}
#     json_dirs["no_heat"] = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/Neuropal_no_heat/"
#     json_dirs["heat"] = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/Neuropal_heat/"
#     h5_dir = "/Users/bennetsakelaris/Documents/Obsidian Vault/worms/wormcode/data/processed_h5/"


#     directories = {}
#     directories["heat"] = glob.glob(json_dirs["heat"]+"*.json") #get filename/paths of all the json files
#     directories["no_heat"] = glob.glob(json_dirs["no_heat"]+"*.json") #get filename/paths of all the json files

#     traces = []
#     labels = []
#     t, l = load_data_from_folder(directories["no_heat"], json_dirs["no_heat"])
#     traces += t
#     labels += l
    
#     t, l = load_data_from_folder(directories["heat"], json_dirs["heat"])
#     traces += t
#     labels += l

#     return traces, labels


# def load_all_heat_data_but_pretend_its_all_one_worm():
#     # This function concatenates everything into one worm and also returns a mask that tells you which neurons are and arent present
#     # Thing to ponder: adding columns of NaNs between worms would probably help the rSLDS learn better
#     full_traces, full_neural_labels = load_all_neural_data_heat()

#     #get list of all neurons recorded in at least one trial
#     neural_labels_set = set()
#     for nl in full_neural_labels:
#         neural_labels_set = neural_labels_set.union(set(nl))
#     neural_labels = np.sort(list(neural_labels_set))
#     print(neural_labels)

#     #now make the traces array, where the different recordings are concatenated and matched by neuron
#     #note: a couple of worms have 15 extra timesteps. I am truncating those to make things easier down the line
#     T = 1599
#     traces = np.zeros((T*len(full_traces), neural_labels.shape[0]))*np.nan #initialize traces to nan 
#     #build the traces matrix
#     w=0
#     for tr, labels, in zip(full_traces, full_neural_labels): #for each worm, w
#         for i in range(tr.shape[1]): #for each neuron, i
#             label = labels[i] #get neuron name
#             idx = np.where(neural_labels==label)[0][0] #get index in full array
#             traces[w*T:(w+1)*T, idx] = tr[0:T,i] #put the neuron's activity in the appropriate spot
#         w+=1 #update worm index

#     #return mask of nan data
#     mask =  (~np.isnan(traces)).astype(int)
#     return traces, neural_labels, mask


def consecutive_counts(arr):
    # Identify positions where the element changes
    changes = np.diff(arr) != 0
    # Get the indices where the changes occur and add 0 at the start and len(arr) at the end
    indices = np.concatenate(([0], np.where(changes)[0] + 1, [len(arr)]))
    
    # Compute consecutive counts using the indices
    counts = np.diff(indices)
    
    # Elements are just the first element of each run
    elements = arr[indices[:-1]]
    
    # Create the result array with [element, count] for each unique run
    result = np.vstack((elements, counts)).T
    return result.astype(int)


def fix_signs(components):
    for i in range(components.shape[0]):
        if components[i].mean() < 0:
            components[i] *= -1
    return components
