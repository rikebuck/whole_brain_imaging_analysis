from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
def plot_states_and_neurons(neurons, neuron_to_color, z_w, q_z_w ,traces, labels, date, fig = None, axs = None, palette = None, cmap = None):
    
    if cmap is None:

        #color palette for plotting, colors as in make_behavior_ethogram
        palette = ["coral",     # forward
                "lightblue", # reverse
                "darkgreen", # turn
                "purple","red", "yellow", "black", "pink"]    # pause
        cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))

    if axs is None: 
        fig, axs = plt.subplots(2, 1, figsize=(18, 6))
    for i, (ax, state) in enumerate(zip(axs, [z_w, q_z_w])):
        # Twin axis for imshow
        ax_img = ax.twinx()
        ax_img.imshow(state[None,:], aspect="auto", cmap=cmap, alpha=0.3, 
                    vmin=0, vmax=len(palette)
                    , extent=[0, len(state), 0, 1]
                    )
        ax_img.set_yticks([])
        ax_img.set_ylim(0, 1)  # Keep the background thin and constant
  
        # Hide twin axis spines
        for spine in ax_img.spines.values():
            spine.set_visible(False)

        ax.set_yticks([])
        if i == 0:
            ax.set_title(f"{date}; Beh")
        else:
            ax.set_title("Inferred by rSLDS")
            ax.set_xticks([])

    # Plot traces

    for neuron in neurons:
        if f"F - {neuron}" not in labels:
            continue
        neuron_i = np.argwhere(labels == f"F - {neuron}").flatten()[0]
        for ax in axs:
            ax.plot(np.arange(traces[:, neuron_i].shape[0]), traces[:, neuron_i], c =neuron_to_color[neuron], label = neuron)
    axs[0].legend(loc="upper left")
    return fig, axs 

def plot_states_and_neurons_state_list(neurons, neuron_to_color, date_state_list, date_state_label, state_cmaps, vmaxs, traces, labels, date, fig = None, axs = None):
    

        # #color palette for plotting, colors as in make_behavior_ethogram

        # state_cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
    n_states = len(date_state_list)
    if axs is None: 
        fig, axs = plt.subplots(n_states, 1, figsize=(18, 6))
    for i, (ax, state, state_lbl, state_cmap, vmax) in enumerate(zip(axs, date_state_list, date_state_label, state_cmaps, vmaxs)):
        # Twin axis for imshow
        ax_img = ax.twinx()
        ax_img.imshow(state[None,:], aspect="auto", cmap=state_cmap, alpha=0.3, 
                    vmin=0, vmax=vmax
                    , extent=[0, len(state), 0, 1]
                    )
        ax_img.set_yticks([])
        ax_img.set_ylim(0, 1)  # Keep the background thin and constant
  
        # Hide twin axis spines
        for spine in ax_img.spines.values():
            spine.set_visible(False)

        ax.set_yticks([])
        if i == 0:
            ax.set_title(f"{date}; {state_lbl}")
        else:
            ax.set_title(f"{state_lbl}")
            ax.set_xticks([])

    # Plot traces

    for neuron in neurons:
        if f"F - {neuron}" not in labels:
            continue
        neuron_i = np.argwhere(labels == f"F - {neuron}").flatten()[0]
        for ax in axs:
            ax.plot(np.arange(traces[:, neuron_i].shape[0]), traces[:, neuron_i], c =neuron_to_color[neuron], label = neuron)
    axs[0].legend(loc="upper left")
    return fig, axs 