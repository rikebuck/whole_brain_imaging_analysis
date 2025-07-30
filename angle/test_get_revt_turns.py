
import numpy as np
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def mark_5frame_zero_windows(labels, mask, window=5):
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


from matplotlib.colors import LinearSegmentedColormap

#color palette for plotting, colors as in make_behavior_ethogram
palette = ["coral",     # forward
           "lightblue", # reverse
           "darkgreen", # turn
           ]    # pause
cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette))
# Example usage
labels = np.array([0, 1, 1, 1, 2, 0,0,0, 1, 1, 1, 2, 0,0,0,0,0,0,0, 1, 1, 0, 1, 1, 0,0,0,0,0,0,2, 0,0,0,0,0,0, 2, 1, 2])
m1_to_2, m1_to_0, m2_after_1, m2_after_0 = make_masks(labels)

fig,(ax,ax2) = plt.subplots(2,1)
ax.imshow(labels[None, :], cmap = LinearSegmentedColormap.from_list("behavior", palette, N=len(palette)))
ax2.imshow(m1_to_2[None, :])

fig,(ax,ax2) = plt.subplots(2,1)
ax.imshow(labels[None, :], cmap = cmap)
ax2.imshow(m1_to_0[None, :])

fig,(ax,ax2) = plt.subplots(2,1)
ax.imshow(labels[None, :], cmap = cmap)
ax2.imshow(m2_after_1[None, :])


fig,(ax,ax2) = plt.subplots(2,1)
ax.imshow(labels[None, :], cmap = cmap)
ax2.imshow(m2_after_0[None, :])



print()


follow_5zeros_after_2after1 = mark_5frame_zero_windows(labels, m2_after_1, window=5)
follow_5zeros_after_1to0 = mark_5frame_zero_windows(labels, m1_to_0, window=5)

fig,(ax,ax2) = plt.subplots(2,1)
ax.imshow(labels[None, :], cmap = cmap)
ax2.imshow(follow_5zeros_after_2after1[None, :])

fig,(ax,ax2) = plt.subplots(2,1)
ax.imshow(labels[None, :], cmap = cmap)
ax2.imshow(follow_5zeros_after_1to0[None, :])


plt.show()