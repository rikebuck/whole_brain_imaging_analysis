import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import sys
sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")

from visualize_data.get_inflection_pt_utils import get_neural_on_times



def get_neural_activity(Y, neuron, neural_labels, T = 5, len_list = 1594):
    y = np.array(Y)[:,5:]
    
    y = np.concatenate([y_i[T:T+len_list] for y_i in y])
    # Y = np.concatenate([Y_i[T:1599] for Y_i in Y])
    y[y==0] = np.nan #why mask out nuerosn that are o1
    return y[:, neural_labels == neuron].flatten()

def bin_activity_by_phase_1(phases, activity, neuron, n_bins = 50, fig = None, axs = None):
    

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(phases, bins) - 1
    mean_features = []
    std_features = []

    for i in range(n_bins):
        values_in_bin = activity[bin_indices == i]
        if len(values_in_bin) > 0:
            mean_features.append(np.nanmean(values_in_bin))
            std_features.append(np.nanstd(values_in_bin))
        else:
            mean_features.append(np.nan)
            std_features.append(np.nan)

    mean_features = np.array(mean_features)
    std_features = np.array(std_features)

    
    
    label = neuron.replace("F -", "")
    
    # --- Create figure and three subplots
    fig = plt.figure(figsize=(10, 3))
    ax_colorwheel = fig.add_subplot(1, 3, 1)               # Cartesian
    ax_radialhist = fig.add_subplot(1, 3, 2, polar=True)   # Polar
    ax_lineplot   = fig.add_subplot(1, 3, 3)               # Cartesian
    
    
    
    plot_color_wheel(ax_colorwheel, vmin = -np.pi, vmax = np.pi)

        
      # --- Plot 3: Line plot
    ax_lineplot = plot_phase_line_plot(fig, ax_lineplot, n_bins, mean_features,bin_centers,  label, 
                                       shading_max =  mean_features + std_features, 
                                       shading_min =  mean_features - std_features )

    
    
    
    #######plot radial hist
    plot_radial_hist(ax_radialhist,mean_features,  n_bins,bin_centers,  label)


    return fig

def plot_color_wheel(ax_colorwheel, vmin = -np.pi, vmax = np.pi):
    # --- Color mapping
    cmap = plt.cm.hsv
    # norm = Normalize(vmin=0, vmax=2 * np.pi)
    norm = Normalize(vmin=vmin, vmax= vmax)
    # --- Plot 1: Color wheel
    r = 4.5
    # theta = np.linspace(0, 2 * np.pi, 360)
    theta = np.linspace(-np.pi,np.pi, 360)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    colors = cmap(norm(theta))
    ax_colorwheel.scatter(x, y, c=colors, s=20)
    ax_colorwheel.set_aspect('equal')
    ax_colorwheel.set_title("Color Wheel")
    return ax_colorwheel

# def plot_phase_line_plot(fig, ax_lineplot, n_bins, phase_times, label, shading_max = None, shading_min = None):
def plot_phase_line_plot(fig, ax_lineplot, n_bins, counts,bin_centers, label, shading_max = None, shading_min = None):
    cmap = plt.cm.hsv
    norm = Normalize(vmin=-np.pi, vmax= np.pi)


    # --- Plot 3: Line plot
    
    # bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    # bin_centers = (bins[:-1] + bins[1:]) / 2


    # # counts, _ = np.histogram(phases[neuron_on_times], bins=bins)
    # counts, _ = np.histogram(phase_times, bins=bins)

    ax_lineplot.plot(bin_centers, counts, color='black')
    if shading_max is not None: 
        ax_lineplot.fill_between(bin_centers,
                   shading_min,
                   shading_max,
                    color='gray', alpha=0.3, label='±1 STD')
    # ax_lineplot.set_ylabel("n occurrences")
    ax_lineplot.set_xticks([])
    
    ax_lineplot.set_title(f"{label} on times Across Phase Bins")

    plt.tight_layout()
    # --- Add colorbar under line plot
    pos = ax_lineplot.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.01, pos.width, 0.02])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Phase (radians)")
    # cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2])
    # cbar.set_ticklabels([r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
    cbar.set_ticks([-np.pi, -0.5*np.pi, 0, np.pi/2, np.pi])
    cbar.set_ticklabels([ r"-$\pi$", r"-$\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])

    return ax_lineplot
def plot_radial_hist(ax_radialhist,counts,  n_bins,bin_centers, label):
    cmap = plt.cm.hsv
    norm = Normalize(vmin=0, vmax=2 * np.pi)

    

    bin_centers = (bin_centers + 2 * np.pi) % (2 * np.pi)
    # print("bin_centers_post", bin_centers/np.pi)
    
    # fig,ax = plt.subplots()
    # ax.plot(bin_centers/np.pi , c="gray")
    bar_colors = cmap(norm(bin_centers))
    
    # print(np.argwhere(bin_centers>=np.pi).flatten())
    # print(np.argwhere(bin_centers<np.pi).flatten())
    
    
    # bar_colors = np.concatenate([bar_colors[np.argwhere(bin_centers>=0).flatten()], bar_colors[np.argwhere(bin_centers<0).flatten()]])
    # bar_colors = np.concatenate([bar_colors[np.argwhere(bin_centers>=np.pi).flatten()], bar_colors[np.argwhere(bin_centers<np.pi).flatten()]])
    bar_colors = np.concatenate([bar_colors[np.argwhere(bin_centers<np.pi).flatten()], bar_colors[np.argwhere(bin_centers>=np.pi).flatten()]])

    # a = np.concatenate([np.argwhere(bin_centers<np.pi).flatten(), np.argwhere(bin_centers>=np.pi).flatten()])
    # print(a/np.pi)
    # fig,ax = plt.subplots()
    # ax.plot(a)
    
    ax_radialhist.bar(bin_centers, counts, width=(2 * np.pi / n_bins),
                    align='center', color=bar_colors, edgecolor='k')
    ax_radialhist.set_theta_zero_location("E")
    # ax_radialhist.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
    # ax_radialhist.set_xticklabels([r"0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"])
    
    # ax_radialhist.set_xticks([-np.pi, -0.5*np.pi, 0, np.pi/2, np.pi])
    # ax_radialhist.set_xticklabels([ r"-$\pi$", r"-$\pi/2$", r"0", r"$\pi/2$", r"$\pi$"])
    


    ax_radialhist.set_xticks([np.pi, 3*np.pi/2, 0, np.pi/2])  # same positions
    ax_radialhist.set_xticklabels([r"-$\pi$", r"-$\pi/2$", r"0", r"$\pi/2$"])

    ax_radialhist.set_title(f"{label} on times Across Phase Bins")
    

def plot_inflection_times_1(phases, neuron_on_times, neuron,  n_bins = 50):
    label = neuron.replace("F -", "")

    phase_times = phases[neuron_on_times]
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(phase_times, bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    
    # --- Create figure and three subplots
    fig = plt.figure(figsize=(10, 3))
    ax_colorwheel = fig.add_subplot(1, 3, 1)               # Cartesian
    ax_radialhist = fig.add_subplot(1, 3, 2, polar=True)   # Polar
    ax_lineplot   = fig.add_subplot(1, 3, 3)               # Cartesian
    
    plot_radial_hist(ax_radialhist,counts,  n_bins,bin_centers,  label)
    # plot_radial_hist(ax_radialhist,counts,  n_bins,bin_centers, label)
    

    plot_color_wheel(ax_colorwheel, vmin = -np.pi, vmax = np.pi)


    # --- Plot 3: Line plot

    ax_lineplot = plot_phase_line_plot(fig, ax_lineplot, n_bins, counts,bin_centers, label )
    
    

    
    # # plt.show()
    return fig

def plot_neuron_heatmap_1d(phase_concat, Y, neuronID,neural_labels,z,  T,  ax = None, num_bins=40, len_list = 1594):
    beh_labels = np.concatenate([z_i[T:T+len_list] for z_i in z])
    figs = []
    for neuron in [f"F - {neuronID}", f"dF - {neuronID}"]:
        
        activity = get_neural_activity(Y, neuron, neural_labels, T = 5, len_list = len_list)
        if not "dF" ==neuron[0:2]:
            smoothed_trace, neuron_on_times, _ = get_neural_on_times(activity, threshold = 0.5) # small bug given that concatenated but fine 

        

            palette = ["coral",     # forward
                "lightblue", # reverse
                "darkgreen", # turn
                #    "purple","red", "yellow", "black", "pink"
                ]    # pause
            neuron_on_times = np.array(neuron_on_times)
            start = max(0, neuron_on_times[0]-100)
            print(start)
            end = min(start+1000, activity.shape[0])

            #plto inflection pts on line graph to akes sure okay 
            fig, ax = plt.subplots(figsize = (12,3)) 
            cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
            # start, end = 7800, 9000

            # Plot behavioral labels as image (1 row)
            ax.imshow(beh_labels[None, start:end],
                    cmap=cmap,
                    aspect='auto',
                    extent=[start, end, np.nanmin(activity[start:end]), np.nanmax(activity[start:end])],
                    alpha=0.4)  # optionally make it transparent

            # Plot activity trace
            ax.plot(np.arange(start, end), activity[start:end], label='neural trace')
            ax.set_title(f"{neuron} on times and trace")
            # Plot neuron on-times
            neuron_mask = np.logical_and(neuron_on_times > start, neuron_on_times < end)
            ax.scatter(neuron_on_times[neuron_mask],
                    activity[neuron_on_times[neuron_mask]],
                    color='black', s=10, label='on-times')
            ax.set_xlim(start, end)
            figs.append(fig)


        
            fig = plot_inflection_times_1(phase_concat, neuron_on_times, neuron, n_bins = num_bins)
            figs.append(fig)
            
        activity -= np.nanmin(activity)
        fig = bin_activity_by_phase_1(phase_concat, activity, neuron, n_bins = num_bins)
        figs.append(fig)

    return figs
