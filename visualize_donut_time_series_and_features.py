import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

from visualize_donut_time_series_and_states import make_masks, mark_5frame_zero_windows, mark_5frame_zero_windows_return_single_frame, mark_5frame_zero_windows_with_start_phase_return_single_frame


###bin phase into 10 bins; for each bin get vel; cuvature; acceleration 


###plot feature trajecotries post rev for ones where end in phase ___ vs phase ___ 

###plot vids..? 

###plot rev features of revs that are followed by this vs not? 

#### 


# figs = plot_neuron_heatmap_1d(worm_angles_concat_neural, Y, neuron,neural_labels, np.array(z)[:,5:], ax = None, num_bins = 40)

def plot_feature_heat_maps_given_phase(T, phase_concat, feature1, feature2,feature1_label, feature2_label, z,   ax = None, n_bins=10, len_list = 1594):
    
    beh_labels = np.concatenate([z_i[T:T+len_list] for z_i in z])
    feature1_concat = np.concatenate([f1[T:T+len_list] for f1 in feature1])
    feature2_concat = np.concatenate([f2[T:T+len_list] for f2 in feature2])
    phases = phase_concat
    print(beh_labels.shape)
    print(feature1_concat.shape)
    print(feature2_concat.shape)
    print(phases.shape)
    fig, feature_matrix = phase_features_labels_heatmap(phases, [feature1_concat, feature2_concat],[feature1_label, feature2_label], n_bins=10, agg=np.nanmean)
    phase_binned_2d_histogram(phases, feature1_concat, feature2_concat, n_phase_bins=10, n_feat_bins=20)

    
# # def phase_feature_heatmap(phases, feature1, feature2,feature1_label, feature2_label, n_bins=10, agg=np.nanmean):
# def phase_feature_heatmap(phases, feature1, feature2,feature1_label, feature2_label, n_bins=10, agg=np.nanmean):
#     # Ensure phases are in [0, 2π]
#     phases = (phases + 2 * np.pi) % (2 * np.pi)

#     # Define bins
#     bins = np.linspace(0, 2 * np.pi, n_bins + 1)
#     bin_indices = np.digitize(phases, bins) - 1

#     # Container for features
#     feature_matrix = np.full((2, n_bins), np.nan)

#     for i in range(n_bins):
#         mask = bin_indices == i
#         feature_matrix[0, i] = agg(feature1[mask]) if np.any(mask) else np.nan
#         feature_matrix[1, i] = agg(feature2[mask]) if np.any(mask) else np.nan

#     # Plot heatmap
#     fig, ax = plt.subplots(figsize=(8, 2))
#     im = ax.imshow(feature_matrix, aspect='auto', cmap='bwr',
#                    extent=[0, 2*np.pi, 0, 2])  # y from 0 to 2 features
#     ax.set_yticks([0.5, 1.5])
#     ax.set_yticklabels([feature1_label, feature2_label])
#     ax.set_xticks(np.linspace(0, 2*np.pi, n_bins+1))
#     ax.set_xticklabels([f"{tick:.1f}" for tick in np.linspace(0, 2*np.pi, n_bins+1)])
#     ax.set_xlabel("Phase (radians)")
#     ax.set_title("Mean feature values per phase bin")

#     fig.colorbar(im, ax=ax, label="Mean feature value")

#     plt.tight_layout()
#     return fig, feature_matrix

# def phase_feature_heatmap(phases, feature1, feature2,feature1_label, feature2_label, n_bins=10, agg=np.nanmean):
from matplotlib.colors import TwoSlopeNorm
def phase_features_labels_heatmap(
                                  phases,
                                  features,
                                 
                                #   vmaxs,
                                  feature_labels,
                                   absmaxs, 
                                  n_bins=10,
                                  agg=np.nanmean, cmaps = None
                                  ):
    phases = (phases + 2 * np.pi) % (2 * np.pi)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    bin_indices = np.digitize(phases, bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    n_features = len(features)
    feature_matrix = np.full((n_features, n_bins), np.nan)

    for i in range(n_bins):
        mask = bin_indices == i
        for j, feature in enumerate(features):
            feature_matrix[j, i] = agg(feature[mask]) if np.any(mask) else np.nan

    if cmaps is None:
        cmaps = ['bwr'] * n_features  # Default to 'bwr'

    counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])


    fig, axs = plt.subplots(
        n_features + 1, 1,
        figsize=(8, 1.6 * (n_features + 1)),
        constrained_layout=True,
        gridspec_kw={'height_ratios': [1] + [1] * n_features}
    )

    # Plot counts on top
    ax_count = axs[0]
    # ax_count.bar(bin_centers, counts, width=(2 * np.pi) / n_bins, align='center', color='gray')
    ax_count.imshow(
            counts[None, :],
            aspect='auto',
            cmap=plt.get_cmap('plasma'),
            # norm=norm,
            extent=[0, 2 * np.pi, 0, 1]
        )
    ax_count.set_ylabel("Count")
    ax_count.set_xticks([])
    ax_count.set_title("Phase bin counts and feature values")
    
    if n_features == 1:
        axs = [axs]

    images = []
    norms = []
    for i, (row, cmap_name, label, absmax) in enumerate(zip(feature_matrix, cmaps, feature_labels, absmaxs, 
                                  )):
    # for i in range(1, len(feature_labels)):
        # row, cmap_name, label = list(zip(feature_matrix, cmaps, feature_labels))[i]
        ax = axs[i+1]
        # absmax = np.nanmax(np.abs(row))
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
        norms.append(norm)

        im = ax.imshow(
            row[None, :],
            aspect='auto',
            cmap=plt.get_cmap(cmap_name),
            norm=norm,
            extent=[0, 2 * np.pi, 0, 1]
        )
        images.append(im)
        
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=25, va='center')
        ax.set_xticks(np.linspace(0, 2 * np.pi, n_bins + 1))
        
        #         # Rainbow (phase angle) colorbar under each feature axis
        # pos = ax.get_position()
        # hsv_cbar_ax = fig.add_axes([
        #     pos.x0,
        #     pos.y0 - 0.025,  # spacing below the axis; adjust as needed
        #     pos.width,
        #     0.01             # short height for slim bar
        # ])
        # rainbow_cmap = plt.cm.hsv
        # phase_norm = Normalize(vmin=0, vmax=2*np.pi)
        # sm_rainbow = ScalarMappable(cmap=rainbow_cmap, norm=phase_norm)
        # sm_rainbow.set_array([])
        # cbar_rainbow = fig.colorbar(
        #     sm_rainbow,
        #     cax=hsv_cbar_ax,
        #     orientation='horizontal'
        # )
        # cbar_rainbow.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        # cbar_rainbow.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"3$\pi/2$", r"$2\pi$"])
        # cbar_rainbow.ax.tick_params(labelsize=6, pad=0.5)
        # cbar_rainbow.set_label("Phase", fontsize=7, labelpad=1)
        
        
        

        if i < n_features - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([f"{tick:.1f}" for tick in np.linspace(0, 2 * np.pi, n_bins + 1)])
            ax.set_xlabel("Phase (radians)")
            
        
    # Add colorbars for all features aligned horizontally at bottom of last axis
    for i, (im, norm, cmap_name) in enumerate(zip(images, norms, cmaps)):
        cbar = fig.colorbar(
            im,
            ax=axs[-1],
            orientation='horizontal',
            pad=0.1,
            aspect=30,
            shrink=0.9
        )
        cbar.set_label(f"{feature_labels[i]} value")
        

        
    # Color each bar in histogram by angle
    # .hist(worm_angles_concat[pure_rev_bin.astype('bool')])
    cmap = plt.cm.hsv  # or any cyclic colormap like 'twilight', 'hsv'
    norm = Normalize(vmin=0, vmax=2*np.pi)
    axs[-1].set_xticks([])


    pos = axs[-1].get_position()
    # cbar_ax = fig.add_axes([
    #     pos.x0,                # left edge matches axs[-1]
    #     pos.y0 - 0.03,         # slightly above or below, adjust as needed
    #     pos.width,             # same width as axs[-1]
    #     0.02                   # height of colorbar
    # ])
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs[-1], orientation='horizontal')
    cbar.set_label("Angle (radians)")
    cbar.set_ticks([ 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"3$\pi/2$", r"2\pi"])
    fig,axs = plt.subplots(1,2)
    # ---- Top: Colorwheel ----
    ax = axs[0]
    r = 4.5
    theta = np.linspace(0, 2*np.pi, 360)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    colors = cmap(norm(theta))
    ax.scatter(x, y, c=colors, s=20)
    ax.set_aspect('equal')
    # ax6.axis('off')
    ax.set_title("Colorwheel for Angles")
    
    ax = axs[1]
    ax.scatter(np.cos(phases), np.sin(phases), alpha = 0.1, color = "coral")
    ax.set_aspect('equal')
    

    return fig, axs, feature_matrix



def phase_binned_2d_histogram(phases, feature1, feature2, n_phase_bins=10, n_feat_bins=20):
    phases = (phases + 2 * np.pi) % (2 * np.pi)  # Ensure [0, 2π]
    phase_bins = np.linspace(0, 2 * np.pi, n_phase_bins + 1)
    bin_indices = np.digitize(phases, phase_bins) - 1
    bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
    

    fig, axs = plt.subplots(2, n_phase_bins // 2, figsize=(n_phase_bins * 1.2, 5), sharex=True, sharey=True)
    cmap = plt.cm.hsv
    norm = Normalize(vmin=0, vmax=2 * np.pi)
    colors = cmap(norm(bin_centers))
    
    
    
    for i in range(n_phase_bins):
        mask = bin_indices == i
        ax = axs.flat[i]
        if np.sum(mask) > 0:
            h = ax.hist2d(feature1[mask], feature2[mask],
                          bins=n_feat_bins, cmap='viridis')

        ax.set_title(f"{phase_bins[i]:.2f}-{phase_bins[i+1]:.2f} rad")
                # Add horizontal color bar for this phase bin
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.07, pos.x1 - pos.x0, 0.02])
        cbar_ax.imshow([[colors[i]]], aspect='auto')
        cbar_ax.set_xticks([])
        cbar_ax.set_yticks([])

    axs[0, 0].set_ylabel("Feature 2")
    axs[1, 0].set_ylabel("Feature 2")
    axs[1, n_phase_bins // 2 - 1].set_xlabel("Feature 1")
    
    # plt.tight_layout()
    cbar = fig.colorbar(h[3], ax=axs.ravel().tolist(), shrink=0.6, label='Counts')
    return fig

def phase_binned_2d_histogram_overlain(phases, feature1, feature2, feature1_label, feature2_label, masks, mask_labels, colors, fig = None, ax = None, n_phase_bins=10, n_feat_bins=20):
    phases = (phases + 2 * np.pi) % (2 * np.pi)  # Ensure [0, 2π]

    xmax = np.nanmax(feature1)
    xmin = np.nanmin(feature1)
    ymax = np.nanmax(feature2)
    ymin = np.nanmin(feature2)

    if fig is None: 
        fig, ax = plt.subplots()
    for i, (mask, label) in enumerate(zip(masks, mask_labels)):
        sns.kdeplot(x=feature1[mask==1], y=feature2[mask==1], levels=5, color=colors[i], linewidths=1, alpha=0.5, ax = ax)#, label = label)
        
        # ax.set_title(f"{label}")
        ax.set_xlabel(feature1_label)
        ax.set_ylabel(feature2_label)
        
        
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    # ax.legend()
    return fig


def plot_feature_phase_heat_maps(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, beh_labels,ax = None,  n_bins=10, window =10, subplot_fig_size = (6,6)):
        

    
    rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(beh_labels)
    fwds_after_rev_turn = mark_5frame_zero_windows(beh_labels, turn_of_rev_turn_bin, window=window)
    fwds_after_pure_rev = mark_5frame_zero_windows(beh_labels, pure_rev_bin, window=window)
    fwds_neither = np.logical_and(np.logical_and(beh_labels == 0, np.logical_not(fwds_after_pure_rev==1)), np.logical_not(fwds_after_rev_turn==1))
    
    cmaps = [
        plt.get_cmap('Greys') , 
        plt.get_cmap('Purples') , 
            plt.get_cmap('Blues') , 
            plt.get_cmap('Greens') , 
            plt.get_cmap( 'Oranges') , 
            plt.get_cmap('PuRd') , 
            
            
            
            
            ]
   
    
    masks = [
        fwds_after_pure_rev==1, 
        fwds_after_rev_turn==1, 
        fwds_neither==1, 

    ]
    
    mask_labels = [
        "fwds_after_pure_rev", "fwds_after_rev_turn", "fwds_neither"
    ]
        
    # # overlay mvoment over time  
    # fig, ax = plt.subplots(figsize = (5,5)) 
    # phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels,["blue", "orange", "purple"], fig = fig, ax = ax, n_phase_bins=10, n_feat_bins=20)
    
    #for each post rev subtype -- overlay 
    for m, (bin_array, bin_array_label) in enumerate(zip([turn_of_rev_turn_bin, pure_rev_bin], ["post rev turn", "post pure rev"])):
        # fig, ax = plt.subplots(1, window)  #for each time step overlay all time steps 
        fig, axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
        feature1_hist_fig, f1_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
        feature2_hist_fig, f2_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
        
        for win in range(0, window-1): #, figsize = (20,5)):
            fwds_after_reor_single_frame_t0 = mark_5frame_zero_windows_return_single_frame(beh_labels, bin_array,win,  window=window)==1
            fwds_after_reor_single_frame_t1 = mark_5frame_zero_windows_return_single_frame(beh_labels, bin_array, win+1, window=window)==1

            masks = [
                    fwds_after_reor_single_frame_t0 == 1, 
                    fwds_after_reor_single_frame_t1 == 1
            
            ]
            mask_labels = [
                    f"{win}\n{bin_array_label}", f"{win+1}\n{bin_array_label}"
            ]
            
            colors = [cmaps[m+2](np.linspace(0, 1, 10))[c] for c in [3,7]]
            
            phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
            # for i, mask in enumerate(masks):
            #     f1_hist_axs[win].hist(feature1_concat[mask], color = colors[i], density = True , label = f"t+{i}")
                
                
            #     f2_hist_axs[win].hist(feature2_concat[mask], color = colors[i], density = True,  label = f"t+{i}")
            # f1_hist_axs[win].set_title(feature1_label + "t vs t+1")
            # f2_hist_axs[win].set_title(feature2_label+ "t vs t+1")
            
            # f2_hist_axs[win].legend()
            # f1_hist_axs[win].legend()
            
            
            labels = ["t", "t+1"]
            f1_hist_axs[win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[win], title = f"{win} post RT" )
            f2_hist_axs[win] = plot_feature_hists_1d(feature2_label, feature2_concat, colors, masks, labels, f2_hist_axs[win], title = f"{win}post RT" )
            
            f2_hist_axs[-1].legend()
            f1_hist_axs[-1].legend()
                
    fig, axs = plt.subplots(1, window-1,figsize = subplot_fig_size )
    # feature1_hist_fig, f1_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    # feature2_hist_fig, f2_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    for win in range(0, window-1): #, figsize = (20,5)):  
        masks, mask_labels, colors = [], [], []
        for m, (bin_array, bin_array_label) in enumerate(zip([turn_of_rev_turn_bin, pure_rev_bin], ["post rev turn", "post pure rev"])):
        # fig, ax = plt.subplots(1, window)  #for each time step overlay all time steps 
        
        
            fwds_after_reor_single_frame_t0 = mark_5frame_zero_windows_return_single_frame(beh_labels, bin_array,win,  window=window)==1
            fwds_after_reor_single_frame_t1 = mark_5frame_zero_windows_return_single_frame(beh_labels, bin_array, win+1, window=window)==1

            masks = masks+[
                    fwds_after_reor_single_frame_t0 == 1, 
                    fwds_after_reor_single_frame_t1 == 1
            
            ]
            mask_labels = mask_labels+ [
                    f"{win}\n{bin_array_label}", f"{win+1}\n{bin_array_label}"
            ]
            
            colors = colors+[cmaps[m+2](np.linspace(0, 1, 10))[c] for c in [3,7]]
            
        phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
        # f1_hist_axs[win].hist(feature1_concat[fwds_after_reor_single_frame_t0==1], color = colors[1], density = True)
        # f1_hist_axs[win].set_title(feature1_label)
            
        # f2_hist_axs[win].hist(feature2_concat[fwds_after_reor_single_frame_t0==1], color = colors[1], density = True)
        # f2_hist_axs[win].set_title(feature2_ladbel)
        # labels = ["RT-t","PR-t", "RT-t+1" "PR-t+1"]
        # f1_hist_axs[win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[win], title = f"{win} post RT" )
        # f2_hist_axs[win] = plot_feature_hists_1d(feature2_label, feature2_concat, colors, masks, labels, f2_hist_axs[win], title = f"{win}post RT" )
        
        # f2_hist_axs[-1].legend()
        # f1_hist_axs[-1].legend()
        
    ##plot rev trun over time 
    fig, axs = plt.subplots(1, window-1,figsize = subplot_fig_size )
    feature1_hist_fig, f1_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    feature2_hist_fig, f2_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    for win in range(0, window-1):
        colors = []
        masks = []
        mask_labels = []
        fwds_after_rev_turn_t0 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win,  window=window)
        fwds_after_rev_turn_t1 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win+1,  window=window)
        
        masks.extend([fwds_after_rev_turn_t0==1, fwds_after_rev_turn_t1==1])
        mask_labels.extend([ f"{win}\n fwds post rev turn", f"{win+1}\n fwds post rev turn"])
        colors.extend([cmaps[0](np.linspace(0, 1, 10))[c] for c in [3,7]])
        axs[win].set_title(f"{win} \n post rev turn")
        phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
        for i, mask in enumerate(masks):
                f1_hist_axs[win].hist(feature1_concat[mask], color = colors[i], density = True , 
                                    #   label = f"t+{i}"
                                    label = mask_labels[i]

                                      )
                
                
                f2_hist_axs[win].hist(feature2_concat[mask], color = colors[i], density = True, 
                                    #   label = f"t+{i}"

                                        label = mask_labels[i]                                      
                                      )
        # f1_hist_axs[win].set_title(feature1_label + "t vs t+1")
        # f2_hist_axs[win].set_title(feature2_label+ "t vs t+1")
        labels = ["t", "t+1"]
        f1_hist_axs[win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[win], title = f"{win} post RT" )
        f2_hist_axs[win] = plot_feature_hists_1d(feature2_label, feature2_concat, colors, masks, labels, f2_hist_axs[win], title = f"{win}post RT" )
        
        f2_hist_axs[-1].legend()
        f1_hist_axs[-1].legend()
    
    # for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
    #     fig, axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    #     for win in range(0, window-1):
    #         colors = []
    #         masks = []
    #         mask_labels = []
                
        
    #         fwds_after_pure_rev_phase0_t0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win , phases,  phase_min, phase_max, window=window) ==1
    #         fwds_after_pure_rev_phase0_t1 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win+1 , phases,  phase_min, phase_max, window=window) ==1
    #         # print("j", j, "win", win, np.argwhere(fwds_after_pure_rev_phase0_t0 ).shape)
    #         masks.extend([fwds_after_pure_rev_phase0_t0==1, fwds_after_pure_rev_phase0_t1==1])
    #         mask_labels.extend([ f"{win}\n {j}*(2pi/5) fwds post pure rev", f"{win+1}\n  {j}*(2pi/5) fwds post pure rev"])
    #         colors.extend([cmaps[j+1](np.linspace(0, 1, 10))[c] for c in [3,7]])
    #         axs[win].set_title(f"{win}; {j}*(2pi/5) \npost pure rev")
            
    #         phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
    
    ###plot all post pure rev phases overlain
    fig, axs = plt.subplots(1, window,figsize = subplot_fig_size)
    feature1_hist_fig, f1_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    feature2_hist_fig, f2_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
        for win in range(0, window-1):
            colors = []
            masks = []
            mask_labels = []
                
        
            fwds_after_pure_rev_phase0_t0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win , phases,  phase_min, phase_max, window=window) ==1
            # print("j", j, "win", win, np.argwhere(fwds_after_pure_rev_phase0_t0 ).shape)
            masks.extend([fwds_after_pure_rev_phase0_t0==1])
            mask_labels.extend([ f"{win}\n {j}*(2pi/5) fwds post pure rev"])
            colors.extend([cmaps[j+1](np.linspace(0, 1, 10))[7]])
            axs[win].set_title(f"{win}")
            
            phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
            labels = ["t", "t+1"]
            f1_hist_axs[win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[win], title = f"{win} post PR \n {j}*(2pi/5)" )
            f2_hist_axs[win] = plot_feature_hists_1d(feature2_label, feature2_concat, colors, masks, labels, f2_hist_axs[win], title = f"{win}post PR\n {j}*(2pi/5)" )
            # for i, mask in enumerate(masks):
                
            #     f1_hist_axs[win].hist(feature1_concat[mask], color = colors[i], density = True , 
            #                           alpha = 0.3, 
            #                         #   label = f"t+{i}"
            #                         label = labels[i]

            #                           )
                
                
            #     f2_hist_axs[win].hist(feature2_concat[mask], color = colors[i], density = True, 
            #                         #   label = f"t+{i}"
            #                          alpha = 0.3, 

            #                             label = labels[i]                                      
            #                           )
            # f1_hist_axs[win].set_title(f"{win}\n {j}*(2pi/5)" )
            # f2_hist_axs[win].set_title(f"{win}\n {j}*(2pi/5)")
            
            # f1_hist_axs[win].set_xlabel(feature1_label )
            # f2_hist_axs[win].set_xlabel(feature2_label)
        f2_hist_axs[-1].legend()
        f1_hist_axs[-1].legend()
        plt.tight_layout()
        
            # for i, mask in enumerate(masks):
            #     f1_hist_axs[win].hist(feature1_concat[mask], color = colors[i], density = True , 
            #                         #   label = f"t+{i}"
            #                         label = mask_labels[i]

            #                           )
                
                
            #     f2_hist_axs[win].hist(feature2_concat[mask], color = colors[i], density = True, 
            #                         #   label = f"t+{i}"

            #                             label = mask_labels[i]                                      
            #                           )
            # f1_hist_axs[win].set_title(feature1_label + "t vs t+1")
            # f2_hist_axs[win].set_title(feature2_label+ "t vs t+1")
            
            # f2_hist_axs[win].legend()
            # f1_hist_axs[win].legend()
        
    
    fig2, axs2 = plt.subplots(figsize = (6,6) ) #make sure gieting pahse -ccroetct - plot
    axs2.hist(phases[mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win,  window=window)==1],  alpha =  0.3, color = cmaps[1](np.linspace(0, 1, 10))[7])
    axs2.set_xlim([0,2*np.pi])
    
    
    fig1, axs1 = plt.subplots(1, window,figsize = subplot_fig_size )

    
    
    for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
        fig, axs = plt.subplots(1, window,figsize = subplot_fig_size )
        feature1_hist_fig, f1_hist_axs = plt.subplots(1, window,figsize = subplot_fig_size)
        feature2_hist_fig, f2_hist_axs = plt.subplots(1, window,figsize = subplot_fig_size)
        for win in range(0, window):
            colors = []
            masks = []
            mask_labels = []
            fwds_after_rev_turn_t0 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win,  window=window)
            # fwds_after_rev_turn_t1 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win+1,  window=window)
            
            masks.extend([fwds_after_rev_turn_t0 ==1, 
                        #   fwds_after_rev_turn_t1
                          ])
            mask_labels.extend([ f"{win}\n fwds post rev turn", 
                                # f"{win+1}\n fwds post rev turn"
                                ])
            colors.extend([cmaps[0](np.linspace(0, 1, 10))[9]])# for c in [3,7]])
            
            fwds_after_pure_rev_phase0_t0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win , phases,  phase_min, phase_max, window=window).astype('bool')

            # print("j", j, "win", win, np.argwhere(fwds_after_pure_rev_phase0_t0 ).shape)
            masks.extend([fwds_after_pure_rev_phase0_t0
                        #   fwds_after_pure_rev_phase0_t1==1
                          ])
            mask_labels.extend([ f"{win}\n {j}*(2pi/5)",
                                # f"{win+1}\n  {j}*(2pi/5) fwds post pure rev"
                                ])
            colors.extend([cmaps[j+1](np.linspace(0, 1, 10))[7]])# for c in [3,7]])
            
            axs1[win].hist(phases[fwds_after_pure_rev_phase0_t0],  alpha =  0.3, color = cmaps[j+1](np.linspace(0, 1, 10))[7])
            axs[win].set_title(f"{win}\n {j}*(2pi/5)")
            
            phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
            labels = ["post rev-turn", "post pure rev"]
            f1_hist_axs[win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels,f1_hist_axs[win], title = f"{win}\n {j}*(2pi/5)" )
            f2_hist_axs[win] = plot_feature_hists_1d(feature2_label, feature2_concat, colors, masks,labels,  f2_hist_axs[win], title = f"{win}\n {j}*(2pi/5)" )
            # for i, mask in enumerate(masks):
                
            #     f1_hist_axs[win].hist(feature1_concat[mask], color = colors[i], density = True , 
            #                           alpha = 0.3, 
            #                         #   label = f"t+{i}"
            #                         label = labels[i]

            #                           )
                
                
            #     f2_hist_axs[win].hist(feature2_concat[mask], color = colors[i], density = True, 
            #                         #   label = f"t+{i}"
            #                          alpha = 0.3, 

            #                             label = labels[i]                                      
            #                           )
            # f1_hist_axs[win].set_title(f"{win}\n {j}*(2pi/5)" )
            # f2_hist_axs[win].set_title(f"{win}\n {j}*(2pi/5)")
            
            # f1_hist_axs[win].set_xlabel(feature1_label )
            # f2_hist_axs[win].set_xlabel(feature2_label)
        f2_hist_axs[-1].legend()
        f1_hist_axs[-1].legend()
        plt.tight_layout()
        
def plot_feature_hists_1d(feature1_label, feature1_concat, colors,masks,labels,  ax, title = "" ):
        # labels = ["post rev-turn", "post pure rev"]
        for i, mask in enumerate(masks):
            ax.hist(feature1_concat[mask], color = colors[i], density = True , 
                                    alpha = 0.3, 
                                #   label = f"t+{i}"
                                label = labels[i]

                                    )
            
            
            # f2_hist_axs[win].hist(feature2_concat[mask], color = colors[i], density = True, 
            #                     #   label = f"t+{i}"
            #                         alpha = 0.3, 

            #                         label = labels[i]                                      
            #                         )
        ax.set_title(title )
        
        ax.set_xlabel(feature1_label )
        
        
        return ax


def plot_feature_heat_maps_given_phase_1(phase_concat, features_list, feature_labels, z,   ax = None, n_bins=10, len_list = 1594,  window = 5, T=5, subplot_fig_size = (16,5)):
    
        
    beh_labels = np.concatenate([z_i[T:T+len_list] for z_i in z])
    # features_all_concat = [np.concatenate([f1[T:T+len_list] for f1 in feature1])for feature1 in features_list]
    feature1, feature2 = features_list
    feature1_label, feature2_label = feature_labels
    feature1_concat = np.concatenate([f1[T:T+len_list] for f1 in feature1])
    feature2_concat = np.concatenate([f2[T:T+len_list] for f2 in feature2])
    
    phases = phase_concat
    phases = (phases + 2 * np.pi) % (2 * np.pi)
    
    
    
    plot_feature_phase_heat_maps(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, beh_labels, ax = ax, n_bins = n_bins, window =window, subplot_fig_size = subplot_fig_size)
    


def plot_feature_heat_maps_given_phase_all(phases, features_all_concat,feature_labels, beh_labels,  absmaxs, ax = None, n_bins=10, window = 5):
        
    fig, ax, feature_matrix = phase_features_labels_heatmap(phases, features_all_concat, feature_labels,  absmaxs, n_bins=n_bins, agg=np.nanmean)
    ax[0].set_title("features vs phase")
    rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(beh_labels)
    fwds_after_rev_turn = mark_5frame_zero_windows(beh_labels, turn_of_rev_turn_bin, window=window)
    fwds_after_pure_rev = mark_5frame_zero_windows(beh_labels, pure_rev_bin, window=window)

    event_specific_phases = phases[fwds_after_pure_rev==1]
    event_specific_features = [f[fwds_after_pure_rev==1] for f in features_all_concat]
    # print(phases.shape)
    fig,ax, feature_matrix = phase_features_labels_heatmap(event_specific_phases, event_specific_features, feature_labels, absmaxs,  n_bins=n_bins, agg=np.nanmean)
    ax[0].set_title("post pure rev features vs phase")
    
    event_specific_phases = phases[fwds_after_rev_turn==1]
    event_specific_features = [f[fwds_after_rev_turn==1] for f in features_all_concat]
    fig,ax,feature_matrix = phase_features_labels_heatmap(event_specific_phases, event_specific_features, feature_labels,absmaxs, n_bins=n_bins, agg=np.nanmean)
    ax[0].set_title("post rev turn features vs phase")
    plt.subplots()
    plt.show()
    
    
def plot_feature_heat_maps_given_phase_post_pure_rev(phase_concat, features_list,feature_labels, z,  T = 5,  ax = None, n_bins = 10,  len_list = 1594, window = 5):
    
    beh_labels = np.concatenate([z_i[T:T+len_list] for z_i in z])
    features_all_concat = [np.concatenate([f1[T:T+len_list] for f1 in feature1])for feature1 in features_list]
    absmaxs = [np.nanmax(np.abs(feature1)) for feature1 in features_list]
    
    # feature1_concat = np.concatenate([f1[T:T+len_list] for f1 in feature1])
    # feature2_concat = np.concatenate([f2[T:T+len_list] for f2 in feature2])
    phases = phase_concat
    plot_feature_heat_maps_given_phase_all(phases, features_all_concat,feature_labels, beh_labels,  absmaxs, ax = ax, n_bins=n_bins, window = window)
# plot_feature_heat_maps_given_phase(worm_angles_concat_neural, acceleration[:,5:], velocity[:,5:],"acceleration", "velocity", np.array(z)[:,5:],   ax = None, n_bins=10, len_list = 1594)


def plot_feature_1d_hist_given_phase_1(phase_concat, features_list, feature_labels, z, n_bins=10, len_list = 1594,  window = 5, T=5, subplot_fig_size = (16,15)):
    
        
    beh_labels = np.concatenate([z_i[T:T+len_list] for z_i in z])

    features_concat = []
    for feature1 in features_list:
        features_concat.append( np.concatenate([f1[T:T+len_list] for f1 in feature1]))
        
    
    return plot_feature_phase_1d_hist(phase_concat, features_concat, feature_labels, beh_labels,  n_bins = n_bins, window =window, subplot_fig_size = subplot_fig_size)
    

def plot_feature_phase_1d_hist(phases, features_concat, feature_labels, beh_labels, n_bins=10, window =10, subplot_fig_size = (6,6)):
        

    
    rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(beh_labels)
    fwds_after_rev_turn = mark_5frame_zero_windows(beh_labels, turn_of_rev_turn_bin, window=window)
    fwds_after_pure_rev = mark_5frame_zero_windows(beh_labels, pure_rev_bin, window=window)
    fwds_neither = np.logical_and(np.logical_and(beh_labels == 0, np.logical_not(fwds_after_pure_rev==1)), np.logical_not(fwds_after_rev_turn==1))
    
    cmaps = [
        plt.get_cmap('Greys') , 
        plt.get_cmap('Purples') , 
            plt.get_cmap('Blues') , 
            plt.get_cmap('Greens') , 
            plt.get_cmap( 'Oranges') , 
            plt.get_cmap('PuRd') , 
            
    ]
   
    
    masks = [
        fwds_after_pure_rev==1, 
        fwds_after_rev_turn==1, 
        fwds_neither==1, 

    ]
    
    mask_labels = [
        "fwds_after_pure_rev", "fwds_after_rev_turn", "fwds_neither"
    ]
        
    # # overlay mvoment over time  
    # fig, ax = plt.subplots(figsize = (5,5)) 
    # phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels,["blue", "orange", "purple"], fig = fig, ax = ax, n_phase_bins=10, n_feat_bins=20)
    
    #for each post rev subtype -- overlay 
    n_features = len(features_concat)
    for m, (bin_array, bin_array_label) in enumerate(zip([turn_of_rev_turn_bin, pure_rev_bin], ["post rev turn", "post pure rev"])):
        # fig, ax = plt.subplots(1, window)  #for each time step overlay all time steps 
        # fig, axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
        feature1_hist_fig, f1_hist_axs = plt.subplots(n_features, window-1,figsize = subplot_fig_size)
        # feature2_hist_fig, f2_hist_axs = plt.subplots(n_features, window-1,figsize = subplot_fig_size)
        
        
        
        for win in range(0, window-1): #, figsize = (20,5)):
            fwds_after_reor_single_frame_t0 = mark_5frame_zero_windows_return_single_frame(beh_labels, bin_array,win,  window=window)==1
            fwds_after_reor_single_frame_t1 = mark_5frame_zero_windows_return_single_frame(beh_labels, bin_array, win+1, window=window)==1

            masks = [
                    fwds_after_reor_single_frame_t0 == 1, 
                    fwds_after_reor_single_frame_t1 == 1
            
            ]
            mask_labels = [
                    f"{win}\n{bin_array_label}", f"{win+1}\n{bin_array_label}"
            ]
            
            colors = [cmaps[m+2](np.linspace(0, 1, 10))[c] for c in [3,7]]
            
            labels = ["t", "t+1"]
            for f_i in range(n_features):
                feature1_label = feature_labels[f_i]
                feature1_concat = features_concat[f_i]
                f1_hist_axs[f_i, win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[f_i, win], title = f"{win} post RT" )
                f1_hist_axs[f_i, -1].legend()
                
    plt.tight_layout()
    ##plot rev trun over time 
    # fig, axs = plt.subplots(1, window-1,figsize = subplot_fig_size )
    feature1_hist_fig, f1_hist_axs = plt.subplots(n_features, window-1,figsize = subplot_fig_size)
    # feature2_hist_fig, f2_hist_axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    for win in range(0, window-1):
        colors = []
        masks = []
        mask_labels = []
        fwds_after_rev_turn_t0 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win,  window=window)
        fwds_after_rev_turn_t1 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win+1,  window=window)
        
        masks.extend([fwds_after_rev_turn_t0==1, fwds_after_rev_turn_t1==1])
        mask_labels.extend([ f"{win}\n fwds post rev turn", f"{win+1}\n fwds post rev turn"])
        colors.extend([cmaps[0](np.linspace(0, 1, 10))[c] for c in [3,7]])
        # axs[win].set_title(f"{win} \n post rev turn")
        # phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)

        # labels = ["t", "t+1"]
        # f1_hist_axs[win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[win], title = f"{win} post RT" )
        # f2_hist_axs[win] = plot_feature_hists_1d(feature2_label, feature2_concat, colors, masks, labels, f2_hist_axs[win], title = f"{win}post RT" )
        
        # f2_hist_axs[-1].legend()
        # f1_hist_axs[-1].legend()
        labels = ["t", "t+1"]
        for f_i in range(n_features):
            feature1_label = feature_labels[f_i]
            feature1_concat = features_concat[f_i]
            f1_hist_axs[f_i, win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[f_i,win], title = f"{win} post RT" )
            f1_hist_axs[f_i, -1].legend()
    plt.tight_layout()
    
    # for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
    #     fig, axs = plt.subplots(1, window-1,figsize = subplot_fig_size)
    #     for win in range(0, window-1):
    #         colors = []
    #         masks = []
    #         mask_labels = []
                
        
    #         fwds_after_pure_rev_phase0_t0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win , phases,  phase_min, phase_max, window=window) ==1
    #         fwds_after_pure_rev_phase0_t1 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win+1 , phases,  phase_min, phase_max, window=window) ==1
    #         # print("j", j, "win", win, np.argwhere(fwds_after_pure_rev_phase0_t0 ).shape)
    #         masks.extend([fwds_after_pure_rev_phase0_t0==1, fwds_after_pure_rev_phase0_t1==1])
    #         mask_labels.extend([ f"{win}\n {j}*(2pi/5) fwds post pure rev", f"{win+1}\n  {j}*(2pi/5) fwds post pure rev"])
    #         colors.extend([cmaps[j+1](np.linspace(0, 1, 10))[c] for c in [3,7]])
    #         axs[win].set_title(f"{win}; {j}*(2pi/5) \npost pure rev")
            
    #         phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
    
    ###plot all post pure rev phases overlain
    # fig, axs = plt.subplots(1, window,figsize = subplot_fig_size)
    feature1_hist_fig, f1_hist_axs = plt.subplots(n_features, window-1,figsize = subplot_fig_size)
    partitions =np.linspace(-np.pi,np.pi, 6)
    for j, (phase_min, phase_max) in enumerate(zip(partitions[0:5], partitions[1:6])):
        for win in range(0, window-1):
            colors = []
            masks = []
            mask_labels = []
                
        
            fwds_after_pure_rev_phase0_t0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win , phases,  phase_min, phase_max, window=window) ==1
            # print("j", j, "win", win, np.argwhere(fwds_after_pure_rev_phase0_t0 ).shape)
            masks.extend([fwds_after_pure_rev_phase0_t0==1])
            mask_labels.extend([ f"{win}\n {j}*(2pi/5) fwds post pure rev"])
            colors.extend([cmaps[j+1](np.linspace(0, 1, 10))[7]])
            # axs[win].set_title(f"{win}")
            
            # phase_binned_2d_histogram_overlain(phases, feature1_concat, feature2_concat, feature1_label, feature2_label, masks, mask_labels, colors,  fig = fig, ax = axs[win], n_phase_bins=10, n_feat_bins=20)
            labels = ["t", "t+1"]
        
            for f_i in range(n_features):
                feature1_label = feature_labels[f_i]
                feature1_concat = features_concat[f_i]
                f1_hist_axs[f_i, win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[f_i,win], title = f"{win} post PR \n {j}*(2pi/5)" )
                f1_hist_axs[f_i, -1].legend()


    plt.tight_layout()

    
    fig2, axs2 = plt.subplots(figsize = (6,6) ) #make sure gieting pahse -ccroetct - plot
    axs2.hist(phases[mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win,  window=window)==1],  alpha =  0.3, color = cmaps[1](np.linspace(0, 1, 10))[7])
    
    
    fig1, axs1 = plt.subplots(1, window,figsize = (subplot_fig_size[0],subplot_fig_size[1]/3  ))


            
    
    partitions =np.linspace(-np.pi,np.pi, 6)
    for j, (phase_min, phase_max) in enumerate(zip(partitions[0:5], partitions[1:6])):
        # fig, axs = plt.subplots(1, window,figsize = subplot_fig_size )
        feature1_hist_fig, f1_hist_axs = plt.subplots(n_features, window,figsize = subplot_fig_size)
        for win in range(0, window):
            colors = []
            masks = []
            mask_labels = []
            
            axs1[win].hist(phases[fwds_after_pure_rev_phase0_t0],  alpha =  0.3, color = cmaps[j+1](np.linspace(0, 1, 10))[7])
            axs1[win].set_title(f"{win}\n {j}*(2pi/5)")
            
            
            fwds_after_rev_turn_t0 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win,  window=window)
            # fwds_after_rev_turn_t1 = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win+1,  window=window)
            
            masks.extend([fwds_after_rev_turn_t0 ==1, 
                        #   fwds_after_rev_turn_t1
                          ])
            mask_labels.extend([ f"{win}\n fwds post rev turn", 
                                # f"{win+1}\n fwds post rev turn"
                                ])
            colors.extend([cmaps[0](np.linspace(0, 1, 10))[9]])# for c in [3,7]])
            
            fwds_after_pure_rev_phase0_t0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win , phases,  phase_min, phase_max, window=window).astype('bool')

            # print("j", j, "win", win, np.argwhere(fwds_after_pure_rev_phase0_t0 ).shape)
            masks.extend([fwds_after_pure_rev_phase0_t0
                        #   fwds_after_pure_rev_phase0_t1==1
                          ])
            mask_labels.extend([ f"{win}\n {j}*(2pi/5)",
                                # f"{win+1}\n  {j}*(2pi/5) fwds post pure rev"
                                ])
            colors.extend([cmaps[j+1](np.linspace(0, 1, 10))[7]])# for c in [3,7]])
            labels = ["post rev-turn", "post pure rev"]
            for f_i in range(n_features):
                feature1_label = feature_labels[f_i]
                feature1_concat = features_concat[f_i]
                f1_hist_axs[f_i, win] = plot_feature_hists_1d(feature1_label, feature1_concat, colors, masks, labels, f1_hist_axs[f_i, win], title = f"{win} post PR \n {j}*(2pi/5)" )
                f1_hist_axs[f_i, -1].legend()

    plt.tight_layout()
