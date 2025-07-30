from scipy.stats import circmean, circstd, mode

from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import circmean, circstd, mode
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from plot_donut_vs_phase import phase_response_curve2, plot_rev_pro2

def bin_phase_data(phases, data, num_bins=20, metric="mode"):
    """Bin the phase data and find the mode of the associated data in each bin."""
    # Define the bin edges
    bins = np.linspace(np.min(phases), np.max(phases), num_bins+1)  # Bins for phase data from -180 to 180 degrees
    binned_data = []
    binned_std = []

    # Find the bin indices for each phase
    bin_indices = np.digitize(phases, bins) - 1  # Get the correct bin for each phase

    # Loop over each bin and find the mode of the associated data
    for i in range(num_bins):
        # Mask the data corresponding to this bin
        bin_mask = bin_indices == i
        bin_data = data[bin_mask]
        
        # Compute the mode of the data in this bin
        if len(bin_data) > 0:  # Only compute mode if there is data in the bin
            if metric=="mode":
                mode_val, _ = mode(bin_data)
            else:
                mode_val = np.mean(bin_data)
                binned_std.append(np.std(bin_data))
            binned_data.append(mode_val)
        else:
            binned_data.append(np.nan)  # If no data in the bin, append NaN

    return bins[:-1], np.array(binned_data), np.array(binned_std)  #

def circular_stats(data, window_size=5):
    """Compute circular mean and standard deviation for smoothing PRC."""
    smoothed, std_dev = [], []
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        smoothed.append(circmean(data[start:end], high=180, low=-180))
        std_dev.append(circstd(data[start:end], high=180, low=-180))
    return np.array(smoothed), np.array(std_dev)

def detect_discontinuities(phases, smoothed_shifts, threshold=90):
    """Break the plot where the phase shift jumps suddenly (e.g., near ±180°)."""
    phase_diff = np.abs(np.diff(smoothed_shifts, prepend=smoothed_shifts[0]))
    mask = phase_diff > threshold  # Detect large jumps
    smoothed_shifts_masked = np.copy(smoothed_shifts)
    smoothed_shifts_masked[mask] = np.nan  # Break the line where jumps occur
    return smoothed_shifts_masked

def plot_phase_resp(phase, response, z=[], label=None, ax=None, color=None, cmap = None):
    
    if cmap is None: 
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
        
        
    phase= np.array(phase)
    response = np.array(response)
    idx = np.argsort(phase)
    sorted_phase = phase[idx]
    sorted_response = response[idx]

    sorted_response_mean, sorted_response_var = circular_stats(sorted_response, window_size=50)
    sorted_response_mean = detect_discontinuities(sorted_phase, sorted_response_mean)


    if ax == None:
        fig,ax = plt.subplots()
    l1 = ax.plot(sorted_phase, sorted_response_mean, label=label, color=color)[0]
    ax.fill_between(sorted_phase, sorted_response_mean - sorted_response_var, sorted_response_mean+sorted_response_var, alpha = 0.3, color=l1.get_color())
    

    if np.size(z)>0:
        # find disc state by phase
        sorted_z = z[idx]
        #bin data, find mode of each bin
        _, states,_ = bin_phase_data(sorted_phase, sorted_z)

        ax.imshow(np.array(states)[np.newaxis,:], extent=[-180,180, -185,-180], cmap=cmap, vmin=0, vmax=6)

    ax.set_xlabel("Phase")
    ax.set_ylabel("Response")
    ax.set_ylim(-185, 180)
    ax.set_xlim(-180, 180)
    return ax

def compute_phase(x, pca):
    """Compute phase from the first two PCs."""
    x=pca.transform(x[np.newaxis,:])
    return np.degrees(np.arctan2(x[:, 1], x[:, 0]))  # PC2 vs PC1

def apply_perturbation(slds, prefix, input, pert_strength, pert_time):
    """Apply a perturbation at a specific time and return new phase trajectory."""
    T = input.shape[0]
    pert_input = input.copy()
    pert_input[pert_time] += pert_strength  # Small perturbation
    
    # Generate perturbed trajectory
    _, x_pert, _ = slds.sample(T=T, prefix=prefix, input=pert_input)
    return compute_phase(x_pert)

def phase_response_curve(slds, init_cons, input, pca,beh_decoder,  offset=-1):
    """Compute the phase response curve (PRC) by applying perturbations at different phases."""
    T = input.shape[0]

    phase_shifts = []
    orig_thetas = []
    final_thetas = []
    final_behs = []
    for i in range(init_cons.shape[0]):
        prefix = [np.array([0]), np.array([init_cons[i]]), np.array([np.zeros(slds.N)])]
        theta_orig = compute_phase(init_cons[i], pca)
        _, x_pert, _ = slds.sample(T=T, prefix=prefix, input=input)
        final_behs.append(beh_decoder.predict(x_pert)[offset])
        theta_pert = compute_phase(x_pert[-1], pca)
        delta_theta = np.unwrap(np.radians(theta_pert - theta_orig))  # Convert to radians and unwrap
        delta_theta = np.degrees(delta_theta)  # Convert back to degrees
        delta_theta = (theta_pert - theta_orig + 180) % 360 - 180 # Keep in 0 to 2π
        phase_shifts.append(delta_theta[0])
        orig_thetas.append(theta_orig[0])
        final_thetas.append(theta_pert[0])


    return orig_thetas, phase_shifts, final_thetas, final_behs




def plot_rev_prob(phase, beh, z=[], label=None, ax=None, color=None, window_size=50, beh1=True, cmap = None):
    if cmap is None: 
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
        
    phase= np.array(phase)
    if beh1:
        beh = np.array(beh) == 1
    idx = np.argsort(phase)
    sorted_phase = phase[idx]
    sorted_beh= beh[idx]

    sorted_beh_mean, sorted_beh_var = circular_stats(sorted_beh, window_size=window_size)


    if ax == None:
        fig,ax = plt.subplots()
    l1 = ax.plot(sorted_phase, sorted_beh_mean, label=label, color=color)[0]
    ax.fill_between(sorted_phase, sorted_beh_mean - sorted_beh_var/5, sorted_beh_mean+sorted_beh_var/5, alpha = 0.3, color=l1.get_color())
    

    if np.size(z)>0:
        # find disc state by phase
        sorted_z = z[idx]
        #bin data, find mode of each bin
        _, states,_ = bin_phase_data(sorted_phase, sorted_z)

        ax.imshow(np.array(states)[np.newaxis,:], extent=[-180,180, -0.02,0], cmap=cmap, vmin=0, vmax=6, aspect="auto")

    ax.set_xlabel("Phase")
    ax.set_ylabel("Response")
    ax.set_ylim(-0.02,1.1)
    ax.set_xlim(-180, 180)
    return ax


def simulate_exp(input_strs, input_durs, neuron,input_dict, new_slds, q_x,pca,  z,  total_dur=5, thresh=1, remove_rev = False):

    if (len(input_durs) == 1) and (len(input_strs) == 1):
        pass
    else:
        if len(input_durs) == 1:
            input_durs = input_durs*len(input_strs)
        if len(input_strs) == 1:
            input_strs = input_strs*len(input_durs)

    inputs = np.zeros((total_dur,new_slds.M))
    input_id = input_dict[neuron]
    
    final_behs = []
    for i in range(len(input_strs)):
        inputs[:input_durs[i],input_id] = input_strs[i]
        results = phase_response_curve2(new_slds, np.concatenate(np.array(q_x[0:5])[:,10:]), inputs, pca, z, offset=-8, thresh=thresh, remove_rev=remove_rev)
        final_behs.append(results[3])
        
    return final_behs


def plot_results(idxs, inferred_phases, final_behaviors, combined, ax, colors, window_size= np.pi/4, num_bins=6, noise_level=0, plot_type="box", vert=True):
    if combined == False:
        for i in idxs:
            # plot_rev_pro2((270-np.degrees(inferred_phases[i] + np.pi+ noise_level*(-0.5+np.random.rand(inferred_phases[i].size))))%360-180, final_behaviors[i], ax=ax, color=colors[i+1], window_size=window_size, beh1=False, plot_type=plot_type, num_bins=num_bins)
            plot_rev_pro2(np.degrees(inferred_phases[i]), final_behaviors[i], ax=ax, color=colors[i+1], window_size=window_size, beh1=False, plot_type=plot_type, num_bins=num_bins)
    else:
        # plot_rev_pro2((270-np.degrees(np.concatenate(inferred_phases[idxs]) + np.pi+ noise_level*(-0.5+np.random.rand(np.concatenate(inferred_phases[idxs]).size))))%360-180, np.concatenate(final_behaviors[idxs]), ax=ax, color=colors[-1], window_size=window_size, beh1=False, plot_type=plot_type, num_bins=num_bins, vert=vert)
        plot_rev_pro2(np.degrees(np.concatenate(inferred_phases[idxs])), np.concatenate(final_behaviors[idxs]), ax=ax, color=colors[-1], window_size=window_size, beh1=False, plot_type=plot_type, num_bins=num_bins, vert=vert)


def bin_phase_data(phases, data, num_bins=20, metric="mode"):
    """Bin the phase data and find the mode of the associated data in each bin."""
    # Define the bin edges
    bins = np.linspace(np.min(phases), np.max(phases), num_bins+1)  # Bins for phase data from -180 to 180 degrees
    binned_data = []
    binned_std = []

    # Find the bin indices for each phase
    bin_indices = np.digitize(phases, bins) - 1  # Get the correct bin for each phase

    # Loop over each bin and find the mode of the associated data
    for i in range(num_bins):
        # Mask the data corresponding to this bin
        bin_mask = bin_indices == i
        bin_data = data[bin_mask]
        
        # Compute the mode of the data in this bin
        if len(bin_data) > 0:  # Only compute mode if there is data in the bin
            if metric=="mode":
                mode_val, _ = mode(bin_data)
            else:
                mode_val = np.mean(bin_data)
                binned_std.append(np.std(bin_data))
            binned_data.append(mode_val)
        else:
            binned_data.append(np.nan)  # If no data in the bin, append NaN

    return bins[:-1], np.array(binned_data), np.array(binned_std)  #

def circular_stats(data, window_size=5):
    """Compute circular mean and standard deviation for smoothing PRC."""
    smoothed, std_dev = [], []
    for i in range(len(data)):
        start = max(0, i - window_size//2)
        end = min(len(data), i + window_size//2 + 1)
        smoothed.append(circmean(data[start:end], high=180, low=-180))
        std_dev.append(circstd(data[start:end], high=180, low=-180))
    return np.array(smoothed), np.array(std_dev)

def detect_discontinuities(phases, smoothed_shifts, threshold=90):
    """Break the plot where the phase shift jumps suddenly (e.g., near ±180°)."""
    phase_diff = np.abs(np.diff(smoothed_shifts, prepend=smoothed_shifts[0]))
    mask = phase_diff > threshold  # Detect large jumps
    smoothed_shifts_masked = np.copy(smoothed_shifts)
    smoothed_shifts_masked[mask] = np.nan  # Break the line where jumps occur
    return smoothed_shifts_masked

def plot_phase_resp(phase, response, z=[], label=None, ax=None, color=None, cmap = None):
    if cmap is None: 
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
        
    
    phase= np.array(phase)
    response = np.array(response)
    idx = np.argsort(phase)
    sorted_phase = phase[idx]
    sorted_response = response[idx]

    sorted_response_mean, sorted_response_var = circular_stats(sorted_response, window_size=50)
    sorted_response_mean = detect_discontinuities(sorted_phase, sorted_response_mean)


    if ax == None:
        fig,ax = plt.subplots()
    l1 = ax.plot(sorted_phase, sorted_response_mean, label=label, color=color)[0]
    ax.fill_between(sorted_phase, sorted_response_mean - sorted_response_var, sorted_response_mean+sorted_response_var, alpha = 0.3, color=l1.get_color())
    

    if np.size(z)>0:
        # find disc state by phase
        sorted_z = z[idx]
        #bin data, find mode of each bin
        _, states,_ = bin_phase_data(sorted_phase, sorted_z)

        ax.imshow(np.array(states)[np.newaxis,:], extent=[-180,180, -185,-180], cmap=cmap, vmin=0, vmax=6)

    ax.set_xlabel("Phase")
    ax.set_ylabel("Response")
    ax.set_ylim(-185, 180)
    ax.set_xlim(-180, 180)
    return ax

def compute_phase(x, pca):
    """Compute phase from the first two PCs."""
    x=pca.transform(x[np.newaxis,:])
    return np.degrees(np.arctan2(x[:, 1], x[:, 0]))  # PC2 vs PC1

def apply_perturbation(slds, prefix, input, pert_strength, pert_time):
    """Apply a perturbation at a specific time and return new phase trajectory."""
    T = input.shape[0]
    pert_input = input.copy()
    pert_input[pert_time] += pert_strength  # Small perturbation
    
    # Generate perturbed trajectory
    _, x_pert, _ = slds.sample(T=T, prefix=prefix, input=pert_input)
    return compute_phase(x_pert)

def phase_response_curve(slds, init_cons, input, pca,beh_decoder,  offset=-1):
    """Compute the phase response curve (PRC) by applying perturbations at different phases."""
    T = input.shape[0]

    phase_shifts = []
    orig_thetas = []
    final_thetas = []
    final_behs = []
    for i in range(init_cons.shape[0]):
        prefix = [np.array([0]), np.array([init_cons[i]]), np.array([np.zeros(slds.N)])]
        theta_orig = compute_phase(init_cons[i], pca)
        _, x_pert, _ = slds.sample(T=T, prefix=prefix, input=input)
        final_behs.append(beh_decoder.predict(x_pert)[offset])
        theta_pert = compute_phase(x_pert[-1], pca)
        delta_theta = np.unwrap(np.radians(theta_pert - theta_orig))  # Convert to radians and unwrap
        delta_theta = np.degrees(delta_theta)  # Convert back to degrees
        delta_theta = (theta_pert - theta_orig + 180) % 360 - 180 # Keep in 0 to 2π
        phase_shifts.append(delta_theta[0])
        orig_thetas.append(theta_orig[0])
        final_thetas.append(theta_pert[0])


    return orig_thetas, phase_shifts, final_thetas, final_behs




def plot_rev_prob(phase, beh, z=[], label=None, ax=None, color=None, window_size=50, beh1=True, cmap = None):
    if cmap is None: 
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
    phase= np.array(phase)
    if beh1:
        beh = np.array(beh) == 1
    idx = np.argsort(phase)
    sorted_phase = phase[idx]
    sorted_beh= beh[idx]

    sorted_beh_mean, sorted_beh_var = circular_stats(sorted_beh, window_size=window_size)


    if ax == None:
        fig,ax = plt.subplots()
    l1 = ax.plot(sorted_phase, sorted_beh_mean, label=label, color=color)[0]
    ax.fill_between(sorted_phase, sorted_beh_mean - sorted_beh_var/5, sorted_beh_mean+sorted_beh_var/5, alpha = 0.3, color=l1.get_color())
    

    if np.size(z)>0:
        # find disc state by phase
        sorted_z = z[idx]
        #bin data, find mode of each bin
        _, states,_ = bin_phase_data(sorted_phase, sorted_z)

        ax.imshow(np.array(states)[np.newaxis,:], extent=[-180,180, -0.02,0], cmap=cmap, vmin=0, vmax=6, aspect="auto")

    ax.set_xlabel("Phase")
    ax.set_ylabel("Response")
    ax.set_ylim(-0.02,1.1)
    ax.set_xlim(-180, 180)
    return ax


