
import scipy
import torch 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd, mode

from compute_donut_utils import compute_phase

def create_X_sequences(X, T):
    X_seq = []
    for i in range(len(X) - T):
        X_seq.append(X[i:i+T])  
    return torch.stack(X_seq)



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


def circular_moving_success_stats(theta, values, window_size):

    theta = np.radians(theta)
    smoothed_rates = np.zeros_like(values, dtype=float)
    smoothed_sem = np.zeros_like(values, dtype=float)

    for i, t in enumerate(theta):
        # Compute circular distances
        circular_diff = np.abs(np.angle(np.exp(1j * (theta - t))))  # Circular distance
        mask = circular_diff <= window_size / 2  # Select points within window

        if np.any(mask):
            local_values = values[mask]
            smoothed_rates[i] = np.nanmean(local_values)  # Compute success rate (fraction of 1s)
            smoothed_sem[i] = scipy.stats.sem(local_values)
        else:
            smoothed_rates[i] = values[i]  # Default to original if no neighbors
            smoothed_sem[i] = 0  # No variance in a single point

    return smoothed_rates, smoothed_sem


def plot_rev_pro2(phase, beh, z=[], label=None, ax=None, color=None, window_size=50, beh1=True, 
                  plot_type="box", num_bins=6, vert=True, cmap = None):
    if cmap is None: 
        palette = ["coral",     # forward
            "lightblue", # reverse
            "darkgreen", # turn
            "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
    phase= np.array(phase)

    phase = np.array(phase)
    if beh1:
        beh = np.array(beh) == 1  # Convert to boolean

    idx = np.argsort(phase)
    sorted_phase = phase[idx]
    sorted_beh = beh[idx]

    # Compute smoothed behavioral response
    sorted_beh_mean, sorted_beh_std = circular_moving_success_stats(sorted_phase, sorted_beh, window_size=window_size)
    #sorted_beh_std = np.sqrt(sorted_beh_mean*(1-sorted_beh_mean)/len(idx))

    #sorted_beh_mean = sorted_beh

    if ax is None:
        fig, ax = plt.subplots()

    if plot_type == "line":
        # Line plot with shaded standard deviation
        l1 = ax.plot(sorted_phase, sorted_beh_mean, label=label, color=color)[0]
        ax.fill_between(sorted_phase, sorted_beh_mean - sorted_beh_std , 
                        sorted_beh_mean + sorted_beh_std , alpha=0.3, color=l1.get_color())

    elif plot_type == "box":
        # Bin phase data
        bin_edges = np.linspace(-180, 180, num_bins + 1)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_indices = np.digitize(sorted_phase, bin_edges) - 1  # Assign each phase to a bin
        
        # Collect data in bins from the smoothed response
        binned_data = [sorted_beh_mean[bin_indices == i] for i in range(num_bins)]  
        
        # Remove empty bins
        valid_bins = [i for i, data in enumerate(binned_data) if len(data) > 0]
        bin_centers = bin_centers[valid_bins]
        binned_data = [binned_data[i] for i in valid_bins]
        print([(sum(~np.isnan(bd))) for bd in binned_data])

        boot_data = []
        for bd in binned_data:
            bin_boot = []
            for i in range(1000):
                n_samples = 4
                idx2 = np.arange(len(bd))
                np.random.shuffle(idx2)
                bin_boot.append(bd[idx2[:n_samples]].mean())
            boot_data.append(np.array(bin_boot))

        filtered_data = []
        filtered_positions = []

        ### Uncomment if you want to Filter out bins with fewer than X samples
        # for data, pos in zip(binned_data, bin_centers):
        #     if len(data) >= 8:
        #         filtered_data.append(data)
        #         filtered_positions.append(pos)

        filtered_data = binned_data
        filtered_positions=bin_centers

        # Plot boxplot using the smoothed behavioral response
        ax.boxplot(filtered_data, positions=filtered_positions, widths=(360 / num_bins) * 0.7, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.7, edgecolor="k"), showfliers=False,
                   medianprops=dict(color="black"), manage_ticks=False, vert=vert)
        

    # Optional: Plot discrete states
    if np.size(z) > 0:
        sorted_z = z[idx]
        _, states, _ = bin_phase_data(sorted_phase, sorted_z)
        ax.imshow(np.array(states)[np.newaxis, :], extent=[-180, 180, -0.02, 0], cmap=cmap, vmin=0, vmax=6, aspect="auto")

    if vert: 
        ax.set_xlabel("Phase")
        ax.set_ylabel("Rev. prob")
        ax.set_ylim(-0.02, 1.1)
        ax.set_xlim(-180, 180)
    else:
        ax.set_xlim(-0.02, 1.1)
        ax.set_ylim(-180, 180)

    if plot_type == "box":
        return ax, np.array([bin_i.mean() for bin_i in boot_data])
    else:
        return ax



def phase_response_curve2(slds, init_cons, input, pca, beh_decoder, offset=-1, thresh=3, remove_rev = False):
    T = input.shape[0]
    init_behs = np.concatenate(np.array(z[0:5])[:,10:]) # assuming init_cons == np.concatenate(np.array(q_x[0:5])[:,10:])

    phase_shifts = []
    orig_thetas = []
    final_thetas = []
    final_behs = []
    all_behs = []
    for i in range(init_cons.shape[0]):
        prefix = [np.array([0]), np.array([init_cons[i]]), np.array([np.zeros(slds.N)])]
        theta_orig = compute_phase(init_cons[i], pca)
        _, x_pert, y_pert = slds.sample(T=T, prefix=prefix, input=input)
        x_est = y_pert@(np.linalg.pinv(slds.emissions.Cs[0]).T) # get back neural activity ? oh no I think its get latents from neural actiity 
        if init_behs[i] == 1 and remove_rev:
            final_behs.append(np.nan)
        else:
            final_behs.append((beh_decoder.predict(x_est)[0:] == 1).sum()>=thresh)
        theta_pert = compute_phase(x_pert[-1], pca)
        delta_theta = np.unwrap(np.radians(theta_pert - theta_orig))  # Convert to radians and unwrap
        delta_theta = np.degrees(delta_theta)  # Convert back to degrees
        delta_theta = (theta_pert - theta_orig + 180) % 360 - 180 # Keep in 0 to 2π
        phase_shifts.append(delta_theta[0])
        orig_thetas.append(theta_orig[0])
        final_thetas.append(theta_pert[0])
        all_behs.append(beh_decoder.predict(x_est)[0:])


    return orig_thetas, phase_shifts, final_thetas, final_behs, all_behs

