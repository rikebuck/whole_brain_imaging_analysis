import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

def find_runs(x):
    """Find start indices, end indices, and values of runs in a 1D array."""
    n = len(x)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=x.dtype)

    change_idx = np.diff(x, prepend=x[0]-1).nonzero()[0]
    start_idx = change_idx
    end_idx = np.append(change_idx[1:], n)
    values = x[start_idx]
    return start_idx, end_idx, values


def mark_5frame_zero_windows(labels, mask, window=3):
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    out = np.zeros_like(labels, dtype=int)

    # Get run ends where mask == 1
    start_idx, end_idx, values = find_runs(mask)

    for s, e, v in zip(start_idx, end_idx, values):
        if v == 1:
            start_check = e
            end_check = e + window
            if end_check <= len(labels) and np.all(labels[start_check:end_check] == 0):
                out[start_check:end_check] = 1
    return out


def mark_5frame_zero_windows_return_single_frame(labels, mask, time_pt, window=3):
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    out = np.zeros_like(labels, dtype=int)

    # Get run ends where mask == 1
    start_idx, end_idx, values = find_runs(mask)

    for s, e, v in zip(start_idx, end_idx, values):
        if v == 1:
            start_check = e
            end_check = e + window
            if end_check <= len(labels) and np.all(labels[start_check:end_check] == 0):
                out[e+time_pt] = 1
    return out



def mark_5frame_zero_windows_with_start_phase(labels, mask, phases, phase_min, phase_max, window=3):
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    out = np.zeros_like(labels, dtype=int)

    # Get run ends where mask == 1
    start_idx, end_idx, values = find_runs(mask)

    for s, e, v in zip(start_idx, end_idx, values):
        if v == 1:
            start_check = e
            if phases[start_check]>phase_min and phases[start_check]<phase_max:
                end_check = e + window
                if end_check <= len(labels) and np.all(labels[start_check:end_check] == 0):
                    out[start_check:end_check] = 1
    return out


def mark_5frame_zero_windows_with_start_phase_return_single_frame(labels, mask, time_pt, phases, phase_min, phase_max, window=3):
    labels = np.asarray(labels)
    mask = np.asarray(mask)
    out = np.zeros_like(labels, dtype=int)

    # Get run ends where mask == 1
    start_idx, end_idx, values = find_runs(mask)

    for s, e, v in zip(start_idx, end_idx, values):
        if v == 1:
            start_check = e
            if phases[start_check]>phase_min and phases[start_check]<phase_max:
                end_check = e + window
                if end_check <= len(labels) and np.all(labels[start_check:end_check] == 0):
                    out[e+time_pt] = 1
    return out



def make_masks(labels):
    labels = np.asarray(labels)
    start_idx, end_idx, values = find_runs(labels)

    mask_1_to_2 = np.zeros_like(labels, dtype=int)
    mask_1_to_0 = np.zeros_like(labels, dtype=int)
    mask_2_after_1 = np.zeros_like(labels, dtype=int)
    mask_2_after_0 = np.zeros_like(labels, dtype=int)

    for i in range(len(values)):
        val = values[i]
        s, e = start_idx[i], end_idx[i]

        if val == 1:
            if i + 1 < len(values) and values[i + 1] == 2:
                mask_1_to_2[s:e] = 1
            elif i + 1 < len(values) and values[i + 1] == 0:
                mask_1_to_0[s:e] = 1

        elif val == 2:
            if i - 1 >= 0 and values[i - 1] == 1:
                mask_2_after_1[s:e] = 1
            elif i - 1 >= 0 and values[i - 1] == 0:
                mask_2_after_0[s:e] = 1

    return mask_1_to_2, mask_1_to_0, mask_2_after_1, mask_2_after_0


def plot_donut_beh_specific(pred_test_rad,test_angles,  binary_array, ax1, colors, cmap = None, alpha = 0.1 ):
    # Plot with Behavior Color Mapping
    pred_test_rad_to_plot  = pred_test_rad[binary_array]
    test_angles_to_plot  = test_angles[binary_array]
    if cmap is None: 
        ax1.scatter(
                    pred_test_rad_to_plot * np.cos(test_angles_to_plot),
                    pred_test_rad_to_plot * np.sin(test_angles_to_plot),
                    c=colors,
                    vmin=0, vmax=6, alpha=alpha, 
                    )
    else: 
        ax1.scatter(
            pred_test_rad_to_plot * np.cos(test_angles_to_plot),
            pred_test_rad_to_plot * np.sin(test_angles_to_plot),
            c=colors,
            vmin=0, vmax=6, alpha=alpha, cmap = cmap
            )

def plot_donut_behs_heldout(pred_test_angles,pred_test_rad,  
                        #  model, inputs, y, 
                         T,
                        #  stats,
                         new_worm_idx, 
                    
                         z,cmap,
                          end_time = 1599,
                         window = 5):
   
    # # beh_labels = z[new_worm_idx][T:end_time-5]
    # beh_labels = z[new_worm_idx][T+5:end_time]
    # # Trajectory Plots
    # fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    # ax1.scatter(pred_test_rad * np.cos(pred_test_angles), pred_test_rad * np.sin(pred_test_angles), c=pred_test_angles)
    # ax1.set_title(f"Predicted trajectory {new_worm_idx}")
    # ax1.legend()
    beh_labels = z[new_worm_idx][T+5:end_time]


    plot_donut_beh_figs(pred_test_angles,pred_test_rad,  beh_labels,
                        #  model, inputs, y, 
                        #  T,
                        #  stats,
     
                    
                        #  z,
                         cmap,
                   
                         window = window)
    
def plot_donut_behs_trained(worm_angles_concat,worm_radii_concat,  
                        #  model, inputs, y, 
                         T,
                        #  stats,
                         z, cmap,
                         len_list = 1594,
                        window = 5 ):

    


    beh_labels = np.concatenate([z_i[T:T+len_list] for z_i in z])
    plot_donut_beh_figs(worm_angles_concat,worm_radii_concat,  beh_labels,
                    #  model, inputs, y, 
                    #  T,
                    #  stats,

                
                    #  z,
                        cmap,
                
                        window = window)

def plot_donut_behs_heldout_shifted_verison(pred_test_angles,pred_test_rad,  
                        #  model, inputs, y, 
                         T,
                        #  stats,
                         new_worm_idx, 
                         z,cmap,
                          end_time = 1599,
                         window = 5,  ):
    beh_labels = z[new_worm_idx][T:end_time-5]
    plot_donut_beh_figs(pred_test_angles,pred_test_rad,  beh_labels,
                    #  model, inputs, y, 
                    #  T,
                    #  stats,

                
                    #  z,
                        cmap,
                
                        window = window)




def plot_donut_beh_figs(worm_angles_concat,worm_radii_concat,  beh_labels,
                        #  model, inputs, y, 
                        #  T,
                        #  stats,
      
                        #  z,
                         cmap,
                            n_bins = 10, 
                         window = 5, 
                         alpha = 0.1):
    worm_angles_concat = (worm_angles_concat+2*np.pi)%(2*np.pi)
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    print("bins", bins)
    
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(worm_radii_concat*np.cos(worm_angles_concat), worm_radii_concat*np.sin(worm_angles_concat), c=worm_angles_concat)
    ax1.set_title("Predicted trajectory")
    
    
    
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(worm_radii_concat*np.cos(worm_angles_concat), worm_radii_concat*np.sin(worm_angles_concat), c=beh_labels, cmap=cmap, vmin=0, vmax=6, alpha=0.1)
    for ax in [ax1, ax2]:
        ax.axis("off")
        ax.axis("equal")
    fig.tight_layout()


    # Plot with Behavior Color Mapping
    
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(
                worm_radii_concat * np.cos(worm_angles_concat),
                worm_radii_concat * np.sin(worm_angles_concat), 
                c=beh_labels,
                cmap=cmap,
                vmin=0,
                vmax=6, 
                alpha=0.2
                )

    ax1.legend()
    
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==0 , ax2, "coral" )
    # ax2.set_title("fwds all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==1 , ax2, "dodgerblue" )
    # ax2.set_title("rev all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==2 , ax2, "green" )
    # ax3.set_title("beh all")
    
    
    ax1.legend()
    rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(beh_labels)
    fwds_after_rev_turn = mark_5frame_zero_windows(beh_labels, turn_of_rev_turn_bin, window=window)
    fwds_after_pure_rev = mark_5frame_zero_windows(beh_labels, pure_rev_bin, window=window)
    palette = ["coral",     # forward
           "lightblue", # reverse
           "darkgreen", # turn
        #    "purple","red", "yellow", "black", "pink"
           ]    # pause
    
    cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
    fig,ax = plt.subplots(7,1, figsize = (10,16), sharex=True)
    ax[0].imshow(beh_labels[None, :], cmap = cmap, aspect='auto')
    plt.tight_layout()
    ax[1].imshow(pure_rev_bin[None, :], aspect='auto')
    ax[2].imshow(fwds_after_pure_rev[None,:], aspect='auto')
    ax[3].imshow(rev_of_rev_turn_bin[None, :], aspect='auto')
    ax[4].imshow(turn_of_rev_turn_bin[None, :], aspect='auto')
    ax[5].imshow(fwds_after_rev_turn[None,:], aspect='auto')
    ax[6].imshow(pure_turn_bin[None, :], aspect='auto')
    
    fig, axs = plt.subplots(1,3, figsize = (12,3))
    for ax in axs: 
        ax.set_aspect('equal')
    ax1,ax2,ax3 = axs
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==0 , ax1, "coral" )
    ax1.set_title("fwds all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==1 , ax2, "dodgerblue" )
    ax2.set_title("rev all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==2 , ax3, "green" )
    ax3.set_title("turn all")
    

    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize = (12,3))
     
    ax1.hist(worm_angles_concat[beh_labels==0], bins = bins)
    ax2.hist(worm_angles_concat[beh_labels==1], bins = bins)
    ax3.hist(worm_angles_concat[beh_labels==2], bins = bins)
    
        # Define colormap
    cmap = plt.cm.hsv  # or any cyclic colormap like 'twilight', 'hsv'
    norm = Normalize(vmin=-np.pi, vmax=np.pi)


    fig, (ax6, ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,6, figsize = (18,3))
    # ---- Top: Colorwheel ----
    r = 4.5
    theta = np.linspace(-np.pi, np.pi, 360)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    colors = cmap(norm(theta))
    ax6.scatter(x, y, c=colors, s=20)
    # ax6.set_aspect('equal')
    # ax6.axis('off')
    ax6.set_title("Colorwheel for Angles")

    
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==1 , ax1, "dodgerblue" )
    ax1.set_title("rev all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, pure_rev_bin.astype('bool'), ax2, "dodgerblue" )
    ax2.set_title("pure rev all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, rev_of_rev_turn_bin==1, ax3, "dodgerblue" )
    ax3.set_title("rev of rev turn all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  turn_of_rev_turn_bin==1, ax4, "cyan" )
    ax4.set_title("turn of rev turn all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, pure_turn_bin==1, ax5, "cyan" )
    ax5.set_title("pure turn all")
    
    r_max = np.max(worm_radii_concat)
    x_lim = (-r_max, r_max)
    y_lim = (-r_max, r_max)

    # Apply limits to all axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
    
    fig, (ax6, ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,6, figsize = (18,3))
    
    r = 4.5
    theta = np.linspace(0, 2*np.pi, 360)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    colors = cmap(norm(theta))
    ax6.scatter(x, y, c=colors, s=20)
    # ax6.set_aspect('equal')
    # ax6.axis('off')
    ax6.set_title("Colorwheel for Angles")
    # Color each bar in histogram by angle
    ax2.hist(worm_angles_concat[pure_rev_bin.astype('bool')], bins = bins)
    ax2.set_xticks([])

    pos = ax2.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.01, pos.width, 0.02])  # Adjust y0 and height as needed
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Angle (radians)")
    cbar.set_ticks([ 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi$/2", r"$2\pi$"])
            

        
    ax3.hist(worm_angles_concat[rev_of_rev_turn_bin==1], bins = bins)
    ax3.set_xlim((0, 2*np.pi))
    ax3.set_title("rev of rev turn all")
    
    ax4.hist(worm_angles_concat[turn_of_rev_turn_bin==1], bins = bins)
    ax4.set_xlim((0, 2*np.pi))
    ax4.set_title("turn of rev turn all")
    
    ax5.hist(worm_angles_concat[pure_turn_bin==1])
    ax5.set_title("pure turn all")

    
    r_max = np.max(worm_radii_concat)
    x_lim = (-r_max, r_max)
    y_lim = (-r_max, r_max)

    # # Apply limits to all axes
    # for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    #     ax.set_aspect('equal')  # ensure circular plots stay circular

    fig, ( ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize = (18,3))
    

    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, beh_labels==0, ax1, "coral" )
    ax1.set_title("fwd all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax2, "coral" )
    ax2.set_title("fwd post pure rev")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax3, "coral" )
    ax3.set_title("fwd post rev turn")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))
                                , ax4, "coral" )
    ax4.set_title("fwd neither")
    
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax5, "darkmagenta" )
    ax5.set_title("fwd post pure rev and rev turn overlain")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax5, "magenta" )

    
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize = (12,3))
    
    ax2.set_title("fwd post pure rev")
    ax2.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins)
    ax2.set_xlim((0, 2*np.pi))
    ax3.set_title("fwd post rev turn")
    ax3.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins)
    ax3.set_xlim((0, 2*np.pi))
    ax4.set_title("fwd neither")
    ax4.hist(worm_angles_concat[ np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))], bins = bins)
    ax4.set_xlim((0, 2*np.pi))

    ax5.hist(worm_angles_concat[fwds_after_pure_rev==1], density = True, alpha = 0.3, bins = bins)
    ax5.hist(worm_angles_concat[fwds_after_rev_turn==1], density = True, alpha = 0.3, bins = bins)
    ax5.hist(worm_angles_concat[ np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))], density = True, alpha = 0.3, bins = bins)
    ax5.set_xlim((0, 2*np.pi))
    ax5.set_title("fwd post pure rev and rev turn overlain")
    ax5.set_xticks([])

    pos = ax5.get_position()
    cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.01, pos.width, 0.02])  # Adjust y0 and height as needed
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Angle (radians)")
    cbar.set_ticks([ 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi$/2", r"$2\pi$"])
            

    # plot_donut_beh_specific(pred_test_rad,pred_test_angles, np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))
    #                         , ax5, "coral" )
    # ax4.set_title("fwd neither")
    
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize = (12,3))
    # print(np.argwhere(beh_labels==0))
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, beh_labels==0, ax1, "coral" )
    ax1.set_title("fwd all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, pure_rev_bin.astype('bool'), ax2, "dodgerblue" )
    ax2.set_title("pure rev all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax2, "coral" )
    ax2.set_title("fwd post pure rev")
    
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  turn_of_rev_turn_bin==1, ax3, "green" )
    ax3.set_title("turn of rev turn all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax3, "coral" )
    ax3.set_title("fwd post rev turn")
    
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  turn_of_rev_turn_bin==1, ax3, "green" )
    ax3.set_title("turn of rev turn all")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax3, "coral" )
    ax3.set_title("fwd post rev turn")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, rev_of_rev_turn_bin==1, ax3, "dodgerblue" )
    ax3.set_title("rev of rev turn all")
    

    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax5, "magenta" )
    ax5.set_title("fwd post rev turn")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax5, "darkmagenta" )
    ax5.set_title("fwd post pure rev")
    

    
    
    fig, axs = plt.subplots(4,5, figsize = (16,7))
    for win in range(1,6):
        fwds_after_rev_turn = mark_5frame_zero_windows(beh_labels, turn_of_rev_turn_bin, window=win)
        ax = axs[0,win-1]
        ax.set_title(f"{win};\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins)
        ax.set_xlim((0, 2*np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows(beh_labels, pure_rev_bin, window=win)
        ax = axs[2,win-1]
        ax.set_title(f"{win};\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,win-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins)
        ax.set_xlim((0, 2*np.pi))
        
    fig, axs = plt.subplots(4,5, figsize = (16,7))
    for i, win in enumerate(range(6,11)):
        fwds_after_rev_turn = mark_5frame_zero_windows(beh_labels, turn_of_rev_turn_bin, window=win)
        ax = axs[0,i-1]
        ax.set_title(f"{win};\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        
        
        ax = axs[1,i-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins)
        ax.set_xlim((0, 2*np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows(beh_labels, pure_rev_bin, window=win)
        ax = axs[2,i-1]
        ax.set_title(f"{win};\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,i-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins)
        ax.set_xlim((0,2*np.pi))
        
    # ######plot specific 
    # fig, axs = plt.subplots(4,5, figsize = (16,7))
    # for win in range(1,6):
    #     fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=win)
    #     ax = axs[0,win-1]
    #     ax.set_title(f"{win};\n fwd post rev turn")
    #     plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    #     ax.set_aspect('equal')  # ensure circular plots stay circular
        
    #     ax = axs[1,win-1]
    #     ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins)
    #     ax.set_xlim((0, 2*np.pi))
        
        
    #     fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin, win, window=win)
    #     ax = axs[2,win-1]
    #     ax.set_title(f"{win};\n fwd post pure rev")
    #     plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    #     ax.set_aspect('equal')  # ensure circular plots stay circular
        
    #     ax = axs[3,win-1]
    #     ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins)
    #     ax.set_xlim((0, 2*np.pi))
        
    # fig, axs = plt.subplots(4,5, figsize = (16,7))
    # for i, win in enumerate(range(6,11)):
    #     fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin,win, window=win)
    #     ax = axs[0,i-1]
    #     ax.set_title(f"{win};\n fwd post rev turn")
    #     plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    #     ax.set_aspect('equal')  # ensure circular plots stay circular
        
    #     ax = axs[1,i-1]
    #     ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins)
    #     ax.set_xlim((0, 2*np.pi))
    #     # print(f"Number of bins: {len(bins) - 1}")
        
        
    #     fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=win)
    #     ax = axs[2,i-1]
    #     ax.set_title(f"{win};\n fwd post pure rev")
    #     plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    #     ax.set_aspect('equal')  # ensure circular plots stay circular
        
    #     ax = axs[3,i-1]
    #     ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins)
    #     ax.set_xlim((0, 2*np.pi))
        
        
    fig, axs = plt.subplots(4,5, figsize = (16,7))
    
    window = 3
    for win in range(1,3):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win-1]
        ax.set_title(f"{win};\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin, win, window=window)
        ax = axs[2,win-1]
        ax.set_title(f"{win};\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,win-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
            
    
    window = 6
    for win in range(1,6):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win-1]
        ax.set_title(f"{win};\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin, win, window=window)
        ax = axs[2,win-1]
        ax.set_title(f"{win};\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,win-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
            
        
    window = 11
    for win in range(1,6):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win-1]
        ax.set_title(f"{win};\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin, win, window=window)
        ax = axs[2,win-1]
        ax.set_title(f"{win};\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,win-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
    ax.legend()
    axs[1,0].legend()
        
    fig, axs = plt.subplots(4,5, figsize = (16,7))
    for i, win in enumerate(range(6,11)):
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin,win, window=window)
        ax = axs[0,i-1]
        ax.set_title(f"{win};\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,i-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))

        
        
        fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
        ax = axs[2,i-1]
        ax.set_title(f"{win};\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,i-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
    ax.legend()
    
    
    cmap = plt.cm.hsv
    colors = cmap(np.linspace(0, 1, 6))
        
    
    window = 11
    fig, axs = plt.subplots(4,6, figsize = (16,7))
    for win in range(0,6):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win]
        ax.set_title(f"{win};\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
        for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
            # phase_min, phase_max = 
            fwds_after_pure_rev_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win, worm_angles_concat,  phase_min, phase_max, window=window)

            ax = axs[2,win]
            ax.set_title(f"{win};\n fwd post pure rev")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,win]
            ax.hist(worm_angles_concat[fwds_after_pure_rev_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
            ax.set_xlim((0, 2*np.pi))
    ax.legend()
    axs[1,0].legend()

    fig, axs = plt.subplots(4,5, figsize = (16,7))
    for i, win in enumerate(range(6,11)):
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin,win, window=window)
        ax = axs[0,i]
        ax.set_title(f"{win};\n fwd post rev turn")

        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax ,"coral")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,i]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        print(f"Number of bins: {len(bins) - 1}")
        
        
        for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
            fwds_after_pure_rev_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            
        # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
            ax = axs[2,i-1]
            ax.set_title(f"{win};\n fwd post pure rev")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,i-1]
            ax.hist(worm_angles_concat[fwds_after_pure_rev_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
            ax.set_xlim((0, 2*np.pi))
    ax.legend()
    
    
    # window = 11
    # fig, axs = plt.subplots(2+2*5,6, figsize = (20,32))
    # for i, win in enumerate(range(6)):
    #     fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
    #     ax = axs[0,win]
    #     ax.set_title(f"{win};\n fwd post rev turn")
    #     plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    #     ax.set_aspect('equal')  # ensure circular plots stay circular
        
    #     ax = axs[1,win]
    #     ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
    #     ax.set_xlim((0, 2*np.pi))
        
    #     for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
    #         # phase_min, phase_max = 
    #         fwds_after_pure_rev_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win, worm_angles_concat,  phase_min, phase_max, window=window)

    #         ax = axs[2+j*2,win]
    #         ax.set_title(f"{win};\n fwd post pure rev")
    #         plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev_phase0==1, ax, colors[j])
    #         ax.set_xlim(x_lim)
    #         ax.set_ylim(y_lim)
    #         ax.set_aspect('equal')  # ensure circular plots stay circular
            
    #         ax = axs[3+j*2,win]
    #         ax.hist(worm_angles_concat[fwds_after_pure_rev_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
    #         ax.set_xlim((0, 2*np.pi))
    # ax.legend()
    # axs[1,0].legend()

    # fig, axs = plt.subplots(2+2*5,6, figsize = (20,32))
    # for i, win in enumerate(range(6,11)):
    #     fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin,win, window=window)
    #     ax = axs[0,i]
    #     ax.set_title(f"{win};\n fwd post rev turn")

    #     plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax ,"coral")
    #     ax.set_xlim(x_lim)
    #     ax.set_ylim(y_lim)
    #     ax.set_aspect('equal')  # ensure circular plots stay circular
        
    #     ax = axs[1,i]
    #     ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
    #     ax.set_xlim((0, 2*np.pi))
    #     print(f"Number of bins: {len(bins) - 1}")
        
        
    #     for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
    #         fwds_after_pure_rev_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            
    #     # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
    #         # ax = axs[2,i-1]
    #         ax = axs[2+j*2,i]
    #         ax.set_title(f"{win};\n fwd post pure rev")
    #         plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev_phase0==1, ax, colors[j])
    #         ax.set_xlim(x_lim)
    #         ax.set_ylim(y_lim)
    #         ax.set_aspect('equal')  # ensure circular plots stay circular
            
    #         # ax = axs[3,i-1]
    #         ax = axs[3+j*2,i]
    #         ax.hist(worm_angles_concat[fwds_after_pure_rev_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
    #         ax.set_xlim((0, 2*np.pi))
    # ax.legend()
    
    ####Plot turning phase specific** 
    ###plot normal duraiton of a turn to get good widnow size
    # turn_bin = turn_of_rev_turn_bin
    # start_idx, end_idx, _ = find_runs(turn_of_rev_turn_bin) 
    # durations  = np.array(end_idx)-np.array(start_idx)
    # window = 5
    # fig, ax = plt.subplots()
    # ax.hist(durations)
    # ax.set_title("turn of rev turn durations")
    # ###plot 
    # fig, axs = plt.subplots(2+2*5,6, figsize = (20,32))
    # for i, win in enumerate(range(6,11)):
    #     for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
    #         turn_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, turn_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            
    #     # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
    #         # ax = axs[2,i-1]
    #         ax = axs[2+j*2,i]
    #         ax.set_title(f"{win};\n turn")
    #         plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_phase0==1, ax, colors[j])
    #         ax.set_xlim(x_lim)
    #         ax.set_ylim(y_lim)
    #         ax.set_aspect('equal')  # ensure circular plots stay circular
            
    #         # ax = axs[3,i-1]
    #         ax = axs[3+j*2,i]
    #         ax.hist(worm_angles_concat[turn_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
    #         ax.set_xlim((0, 2*np.pi))

    # turn_bin =  pure_turn_bin
    # start_idx, end_idx, _ = find_runs(turn_of_rev_turn_bin) 
    # durations  = np.array(end_idx)-np.array(start_idx)
    # fig, ax = plt.subplots()
    # ax.hist(durations)
    # ax.set_title("pure turn durations")
    # ###plot 
    # fig, axs = plt.subplots(2+2*5,6, figsize = (20,32))
    # for i, win in enumerate(range(6,11)):
    #     for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
    #         turn_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, turn_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            
    #     # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
    #         # ax = axs[2,i-1]
    #         ax = axs[2+j*2,i]
    #         ax.set_title(f"{win};\n turn")
    #         plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_phase0==1, ax, colors[j])
    #         ax.set_xlim(x_lim)
    #         ax.set_ylim(y_lim)
    #         ax.set_aspect('equal')  # ensure circular plots stay circular
            
    #         # ax = axs[3,i-1]
    #         ax = axs[3+j*2,i]
    #         ax.hist(worm_angles_concat[turn_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
    #         ax.set_xlim((0, 2*np.pi))


    cmap = plt.cm.hsv
    colors = cmap(np.linspace(0, 1, 6))
    
    turn_bin = beh_labels ==2#turn_of_rev_turn_bin
    start_idx, end_idx, _ = find_runs(turn_bin) 
    durations  = np.array(end_idx)-np.array(start_idx)
    start_phase =worm_angles_concat[np.array(start_idx)]
    fig, ax = plt.subplots()
    ax.scatter(start_phase, durations) 
    ax.set_xlabel("turn duration")
    ax.set_ylabel("turn start start_phase")
    
    # fig, ax = plt.subplots()
    # ax.scatter( worm_angles_concat[np.array(end_idx)], durations) 
    # ax.set_xlabel("turn duration")
    # ax.set_ylabel("turn  end _phase")
    
    window = 5
    fig, ax = plt.subplots()
    ax.hist(durations)
    ax.set_title("turn of rev turn durations")
        
    
    window = 30
    fig, axs = plt.subplots(4,6, figsize = (16,7))
    for win in range(0,6):######
        turn_bin_ = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_bin, win, window=window)
        ax = axs[0,win]
        ax.set_title(f"{win};\n turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win]
        ax.hist(worm_angles_concat[turn_bin_==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
        for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
            # phase_min, phase_max = 
            turn_bin_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, turn_bin, win, worm_angles_concat,  phase_min, phase_max, window=window)

            ax = axs[2,win]
            ax.set_title(f"{win};\n turn")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,win]
            ax.hist(worm_angles_concat[turn_bin_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
            ax.set_xlim((0, 2*np.pi))
    ax.legend()
    axs[1,0].legend()

    fig, axs = plt.subplots(4,5, figsize = (16,7))
    for i, win in enumerate(range(6,11)):
        turn_bin_ = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_bin,win, window=window)
        ax = axs[0,i]
        ax.set_title(f"{win};\n turn")

        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_==1, ax ,"coral")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,i]
        ax.hist(worm_angles_concat[turn_bin_==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        print(f"Number of bins: {len(bins) - 1}")
        
        
        for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
            turn_bin_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, turn_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            
        # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
            ax = axs[2,i-1]
            ax.set_title(f"{win};\n turn")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,i-1]
            ax.hist(worm_angles_concat[turn_bin_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
            ax.set_xlim((0, 2*np.pi))
    ax.legend()
        
    # turn_bin = turn_of_rev_turn_bin
    # start_idx, end_idx, _ = find_runs(pure_turn_bin) 
    # durations  = np.array(end_idx)-np.array(start_idx)
    # window = 5
    # fig, ax = plt.subplots()
    # ax.hist(durations)
    # ax.set_title("turn of rev turn durations")
        

    fig, axs = plt.subplots(4,6, figsize = (16,7))
    for i, win in enumerate(range(11,16)):######
        turn_bin_ = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_bin, win, window=window)
        ax = axs[0,i]
        ax.set_title(f"{win};\n turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,i]
        ax.hist(worm_angles_concat[turn_bin_==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
        for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
            # phase_min, phase_max = 
            turn_bin_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, turn_bin, win, worm_angles_concat,  phase_min, phase_max, window=window)

        # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
            ax = axs[2,i-1]
            ax.set_title(f"{win};\n turn")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,i-1]
            ax.hist(worm_angles_concat[turn_bin_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
            ax.set_xlim((0, 2*np.pi))
    ax.legend()
    axs[1,0].legend()

    fig, axs = plt.subplots(4,5, figsize = (16,7))
    # for i, win in enumerate(range(6,11)):
    for i, win in enumerate(range(16, 21)):######
        turn_bin_ = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_bin,win, window=window)
        ax = axs[0,i]
        ax.set_title(f"{win};\n turn")

        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_==1, ax ,"coral")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,i]
        ax.hist(worm_angles_concat[turn_bin_==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        print(f"Number of bins: {len(bins) - 1}")
        
        
        for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
            turn_bin_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, turn_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            
        # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
            ax = axs[2,i-1]
            ax.set_title(f"{win};\n turn")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,i-1]
            ax.hist(worm_angles_concat[turn_bin_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
            ax.set_xlim((0, 2*np.pi))
    ax.legend()
    
    fig, axs = plt.subplots(2,10, figsize = (28,7))
    # for i, win in enumerate(range(6,11)):
    for i, win in enumerate(range(0,400, int(400/10))):######
        turn_bin_ = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_bin,win, window=window)
        ax = axs[0,i]
        ax.set_title(f"{win};\n turn")

        
        
        for j, (phase_min, phase_max) in enumerate(zip(np.linspace(0,2*np.pi, 6)[0:5], np.linspace(0,2*np.pi, 6)[1:6])):
            # turn_bin_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, turn_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            turn_bin_phase0 = mark_5frame_zero_windows_with_start_phase(beh_labels, turn_bin, worm_angles_concat, phase_min, phase_max, window=3)
        
            ax = axs[0,i-1]
            ax.set_title(f"{win};\n turn")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, turn_bin_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[1,i-1]
            ax.hist(worm_angles_concat[turn_bin_phase0==1], bins = bins, label = f"full window = {window}", alpha =  0.3, color = colors[j])
            ax.set_xlim((0, 2*np.pi))
    
    

    

    ax.legend()
    

    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5, figsize = (12,3))
###########
##############
###############
################
    
def plot_donut_behs_1(worm_angles_concat,worm_radii_concat,  
                        #  model, inputs, y, 
                         T,
                        #  stats,
                         z, cmap,
                         len_list = 1594,
                        window = 5 ):

    


    beh_labels = np.concatenate([z_i[T:T+len_list] for z_i in z])
    all_figs = plot_donut_beh_figs_1(worm_angles_concat,worm_radii_concat,  beh_labels,
                    #  model, inputs, y, 
                    #  T,
                    #  stats,

                
                    #  z,
                        cmap,
                
                        window = window)
    return all_figs


def plot_donut_beh_figs_1(worm_angles_concat,worm_radii_concat,  beh_labels,

                         cmap,
                            n_bins = 10, 
                         window = 5):
    # worm_angles_concat = (worm_angles_concat+2*np.pi)%(2*np.pi)
    # bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    all_figs = []
    bins = np.linspace(-1* np.pi, np.pi, n_bins + 1)
    print("bins", bins)
    
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(4, 2))
    ax1.scatter(worm_radii_concat*np.cos(worm_angles_concat), worm_radii_concat*np.sin(worm_angles_concat), c=worm_angles_concat)
    ax1.set_title("Predicted trajectory")

    ax2.scatter(worm_radii_concat*np.cos(worm_angles_concat), worm_radii_concat*np.sin(worm_angles_concat), c=beh_labels, cmap=cmap, vmin=0, vmax=6, alpha=0.1)
    for ax in [ax1, ax2]:
        ax.axis("off")
        ax.axis("equal")
    fig.tight_layout()

    all_figs.append(fig)
    

    rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(beh_labels)
    fwds_after_rev_turn = mark_5frame_zero_windows(beh_labels, turn_of_rev_turn_bin, window=window)
    fwds_after_pure_rev = mark_5frame_zero_windows(beh_labels, pure_rev_bin, window=window)
    palette = ["coral",     # forward
           "lightblue", # reverse
           "darkgreen", # turn
        #    "purple","red", "yellow", "black", "pink"
           ]    # pause
    
    fig, axs = plt.subplots(2,4, figsize = (14,4))
    # for ax in axs: 
    #     ax.set_aspect('equal')
    # # ax1,ax2,ax3 = axs
    # bins = nplinspace()
    plot_donut_beh_specific(worm_radii_concat, worm_angles_concat,  beh_labels==0 , axs[0,0], "coral", alpha = 0.05 )
    axs[0,0].set_title("fwds all")
    axs[0,0].set_aspect('equal')
    axs[1,0].hist(worm_angles_concat[beh_labels==0], bins = bins, alpha = 0.3, color ="coral")
    
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==1 , axs[0,1], "dodgerblue" , alpha = 0.05 )
    axs[0,1].set_title("rev all")
    axs[0,1].set_aspect('equal')
    axs[1,1].hist(worm_angles_concat[beh_labels==1], bins = bins, alpha = 0.3, color ="dodgerblue")
     
     
    plot_donut_beh_specific(worm_radii_concat, worm_angles_concat,  beh_labels==2 , axs[0,2], "green" , alpha = 0.05)
    axs[0,2].set_title("turn all")
    axs[0,2].set_aspect('equal')
    axs[1,2].hist(worm_angles_concat[beh_labels==2], bins = bins, alpha = 0.3, color ="green")
    
    axs[0,3].set_title("overlain")
    step = 10
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step],  beh_labels[::step]==0 , axs[0,3], "coral" , alpha = 0.05)
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step],  beh_labels[::step]==1 , axs[0,3], "dodgerblue" , alpha = 0.05)
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step],  beh_labels[::step]==2 , axs[0,3], "green" , alpha = 0.05)
    axs[0,3].set_aspect('equal')
    axs[1,3].hist(worm_angles_concat[beh_labels==0], bins = bins, alpha = 0.3, color ="coral", density = True)
    axs[1,3].hist(worm_angles_concat[beh_labels==1], bins = bins, alpha = 0.3, color ="dodgerblue", density = True)
    axs[1,3].hist(worm_angles_concat[beh_labels==2], bins = bins, alpha = 0.3, color ="green", density = True)
        # Define colormap
    phase_cmap = plt.cm.hsv  # or any cyclic colormap like 'twilight', 'hsv'
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

        
    for ax in axs[1,:]:
        ax.set_xticks([])
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.01, pos.width, 0.02])  # Adjust y0 and height as needed
        sm = ScalarMappable(cmap=phase_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Angle (radians)")
        cbar.set_ticks([ -1*np.pi, -0.5*np.pi, 0, np.pi/2, np.pi])#, 3*np.pi/2, 2*np.pi])
        cbar.set_ticklabels([r"-$\pi$", r"-$\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    
    all_figs.append(fig)

    fig, axs = plt.subplots(2,6, figsize = (18,3))

    ax2,ax3, ax1, ax4,ax5, ax6 = axs[0,:]
    # plot_donut_beh_specific(worm_radii_concat,worm_angles_concat,  beh_labels==1 , ax1, "dodgerblue" )
    # ax1.set_title("rev all")
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], pure_rev_bin[::step].astype('bool'), ax2, "cornflowerblue" )
    ax2.set_title("pure rev all")
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], rev_of_rev_turn_bin[::step]==1, ax3, "darkcyan" )
    ax3.set_title("rev of rev turn all")
    ax1.set_title("all rev")
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], pure_rev_bin[::step].astype('bool'), ax1, "cornflowerblue" )
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], rev_of_rev_turn_bin[::step].astype('bool'), ax1, "darkcyan" )
    
    
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step],  turn_of_rev_turn_bin[::step]==1, ax4, "darkkhaki" )
    ax4.set_title("turn of rev turn all")
    
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], pure_turn_bin[::step]==1, ax5, "darkgreen" )
    ax5.set_title("pure turn all")
    
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], turn_of_rev_turn_bin[::step].astype('bool'), ax6, "darkkhaki" )
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], pure_turn_bin[::step].astype('bool'), ax6, "darkgreen" )
    ax6.set_title("all turn")

    
    r_max = np.max(worm_radii_concat)
    x_lim = (-r_max, r_max)
    y_lim = (-r_max, r_max)

    # Apply limits to all axes
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
    ax2,ax3, ax1,ax4,ax5, ax6 = axs[1,:]
    ax2.hist(worm_angles_concat[pure_rev_bin.astype('bool')], bins = bins, alpha =0.3, color = "cornflowerblue" )
    

    ax3.hist(worm_angles_concat[rev_of_rev_turn_bin==1], bins = bins, alpha =0.3, color = "darkcyan" )

    ax1.hist(worm_angles_concat[pure_rev_bin.astype('bool')], bins = bins, alpha =0.3, color = "cornflowerblue" , density = True)
    ax1.hist(worm_angles_concat[rev_of_rev_turn_bin==1], bins = bins, alpha =0.3, color = "darkcyan" , density = True)
    
    ax4.hist(worm_angles_concat[turn_of_rev_turn_bin==1], bins = bins, alpha =0.3, color = "darkkhaki" )
    
    ax5.hist(worm_angles_concat[pure_turn_bin==1], bins = bins, alpha =0.3, color = "darkgreen" )


    ax6.hist(worm_angles_concat[turn_of_rev_turn_bin==1], bins = bins, alpha =0.3, color = "darkkhaki" , density = True)
    ax6.hist(worm_angles_concat[pure_turn_bin==1], bins = bins, alpha =0.3, color = "darkgreen" , density = True)
    
    for ax in axs[1,:]:
        ax.set_xticks([])
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.01, pos.width, 0.02])  # Adjust y0 and height as needed
        sm = ScalarMappable(cmap=phase_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Angle (radians)")
        cbar.set_ticks([ -1*np.pi, -0.5*np.pi, 0, np.pi/2, np.pi])#, 3*np.pi/2, 2*np.pi])
        cbar.set_ticklabels([r"-$\pi$", r"-$\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        
        # cbar.set_ticks([ 0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        # cbar.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi$/2", r"$2\pi$"])
    all_figs.append(fig)

    fig, axs= plt.subplots(2,4, figsize = (12,3))
    ax2,ax3,  ax4 , ax5 = axs[0,:]


    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax2, "darkorange" )
    ax2.set_title("fwd post pure rev")
    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax3, "darkmagenta" )
    ax3.set_title("fwd post rev turn")
    
    
    ax5.set_title("all overlain")
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], fwds_after_pure_rev[::step]==1, ax5, "darkorange" )
    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], fwds_after_rev_turn[::step]==1, ax5, "darkmagenta" )

    plot_donut_beh_specific(worm_radii_concat[::step],worm_angles_concat[::step], np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))[::step]
                                    , ax5, "hotpink" )

    plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))
                                , ax4, "hotpink" )
    ax4.set_title("fwd neither")
    for ax in axs[0,:]:
        ax.set_aspect("equal")
    
    ax2,ax3,  ax4, ax5 = axs[1,:]

    
    
    ax2.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, alpha = 0.3, color =  "darkorange" )

    ax3.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, alpha = 0.3, color =  "darkmagenta" )
    
    ax5.hist(worm_angles_concat[fwds_after_pure_rev==1], density = True, alpha = 0.3, bins = bins, color =  "darkorange" )
    ax5.hist(worm_angles_concat[fwds_after_rev_turn==1], density = True, alpha = 0.3, bins = bins, color =  "darkmagenta" )
    ax5.hist(worm_angles_concat[ np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))], density = True, alpha = 0.3, bins = bins, color = "hotpink")

  
    ax4.hist(worm_angles_concat[ np.logical_and( beh_labels==0, np.logical_and(fwds_after_pure_rev==0, fwds_after_rev_turn==0))], alpha = 0.3, bins = bins, color = "hotpink")

    for ax in axs[1,:]:
        ax.set_xticks([])
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.01, pos.width, 0.02])  # Adjust y0 and height as needed
        sm = ScalarMappable(cmap=phase_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label("Angle (radians)")
        cbar.set_ticks([ -1*np.pi, -0.5*np.pi, 0, np.pi/2, np.pi])#, 3*np.pi/2, 2*np.pi])
        cbar.set_ticklabels([r"-$\pi$", r"-$\pi/2$", "0", r"$\pi/2$", r"$\pi$"])    
        
    all_figs.append(fig)
    
    dt = 0.6
    fig, axs = plt.subplots(4,6, figsize = (19,7))
    
    window = 3
    for win in range(1,3):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win-1]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        # ax.set_xlim((0, 2*np.pi))
        ax.set_xlim((-np.pi, np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin, win, window=window)
        ax = axs[2,win-1]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,win-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        # ax.set_xlim((0, 2*np.pi))
        ax.set_xlim((-np.pi, np.pi))
        
            
    
    window = 6
    for win in range(1,6):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win-1]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        # ax.set_xlim((0, 2*np.pi))
        ax.set_xlim((-np.pi, np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin, win, window=window)
        ax = axs[2,win-1]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,win-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        ax.set_xlim((0, 2*np.pi))
        
            
        
    window = 11
    
    for win in range(1,7):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win-1]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win-1]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        # ax.set_xlim((0, 2*np.pi))
        ax.set_xlim((-np.pi, np.pi))
        
        
        fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin, win, window=window)
        ax = axs[2,win-1]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post pure rev")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[3,win-1]
        ax.hist(worm_angles_concat[fwds_after_pure_rev==1], bins = bins, label = f"full window = {window}", alpha =  0.3)
        # ax.set_xlim((0, 2*np.pi))
        ax.set_xlim((-np.pi, np.pi))
    ax.legend()
    axs[1,0].legend()
    
    cmap = plt.cm.hsv
    colors = cmap(np.linspace(0, 1, 6))
        
    all_figs.append(fig)
    
    window = 11
    fig, axs = plt.subplots(4,6, figsize = (16,7))
    for win in range(0,6):######
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin, win, window=window)
        ax = axs[0,win]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post rev turn")
        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax, "coral" )
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,win]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, #label = f"{np.round(phase_min,2)}",
                alpha =  0.3)
        # ax.set_xlim((0, 2*np.pi))
        ax.set_xlim((-np.pi, np.pi))
        
        # partitions = np.linspace(0,2*np.pi, 6)
        partitions = np.linspace(-np.pi,np.pi, 6)
        for j, (phase_min, phase_max) in enumerate(zip(partitions[0:5], partitions[1:6])):
            # phase_min, phase_max = 
            fwds_after_pure_rev_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win, worm_angles_concat,  phase_min, phase_max, window=window)

            ax = axs[2,win]
            ax.set_title(f"{np.round(win*dt)}s;\n fwd post pure rev")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,win]
            ax.hist(worm_angles_concat[fwds_after_pure_rev_phase0==1], bins = bins, label = f"{np.round(phase_min,2)}", alpha =  0.3, color = colors[j])
            # ax.set_xlim((0, 2*np.pi))
            ax.set_xlim((-np.pi, np.pi))
    # ax.legend()
    # axs[1,0].legend()
    
    all_figs.append(fig)
    

    fig, axs = plt.subplots(4,5, figsize = (16,7))
    for i, win in enumerate(range(6,11)):
        fwds_after_rev_turn = mark_5frame_zero_windows_return_single_frame(beh_labels, turn_of_rev_turn_bin,win, window=window)
        ax = axs[0,i]
        ax.set_title(f"{np.round(win*dt)}s;\n fwd post rev turn")

        plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_rev_turn==1, ax ,"coral")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect('equal')  # ensure circular plots stay circular
        
        ax = axs[1,i]
        ax.hist(worm_angles_concat[fwds_after_rev_turn==1], bins = bins, #label = f"{np.round(phase_min,2)}", 
                alpha =  0.3)
        # ax.set_xlim((0, 2*np.pi))
        ax.set_xlim((-np.pi, np.pi))
        print(f"Number of bins: {len(bins) - 1}")
        
        
        for j, (phase_min, phase_max) in enumerate(zip(partitions[0:5], partitions[1:6])):
            fwds_after_pure_rev_phase0 = mark_5frame_zero_windows_with_start_phase_return_single_frame(beh_labels, pure_rev_bin, win,worm_angles_concat,  phase_min, phase_max, window=window)
            
        # fwds_after_pure_rev = mark_5frame_zero_windows_return_single_frame(beh_labels, pure_rev_bin,win,  window=window)
            ax = axs[2,i-1]
            ax.set_title(f"{np.round(win*dt)}s;\n fwd post pure rev")
            plot_donut_beh_specific(worm_radii_concat,worm_angles_concat, fwds_after_pure_rev_phase0==1, ax, colors[j])
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_aspect('equal')  # ensure circular plots stay circular
            
            ax = axs[3,i-1]
            ax.hist(worm_angles_concat[fwds_after_pure_rev_phase0==1], bins = bins, #label = f"{np.round(phase_min,2)}", 
                    alpha =  0.3, color = colors[j])
            # ax.set_xlim((0, 2*np.pi))
            ax.set_xlim((-np.pi, np.pi))
    print(all_figs)
    return all_figs
