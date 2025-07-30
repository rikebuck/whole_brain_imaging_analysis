import numpy as np
import torch
import matplotlib.pyplot as plt
from angle.preprocess import create_sequences

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


def plot_prediction(model, inputs, y,
                     T, y_train_mean, y_train_std, z,cmap='hsv', pca_x=None, normalize=True):

    worm_angles, worm_radii = [], []
    worm_angles_real, worm_radii_real = [], []

    for worm_idx in range(len(inputs[0])):
        X = np.array(inputs)[:,worm_idx,:].T
        y_true = y[worm_idx]

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_true, dtype=torch.float32)

        if T > 0:
            X_seq, y_seq = create_sequences(X_tensor, y_tensor, T)
        else:
            X_seq = X_tensor
            y_seq = y_tensor

        if normalize:
            # Normalize like training
            X_seq = (X_seq - X_seq.mean(0)) / (X_seq.std(0) + 1e-8)

        model.eval()
        with torch.no_grad(): # what does this mean? 
            y_pred = model(X_seq).squeeze()
        y_pred = y_pred * y_train_std + y_train_mean

        end_time = min(len(y_true) - T, len(y_pred))
        pred_angles = np.arctan2(y_pred[:end_time, 1], y_pred[:end_time, 0])#Y_sin, y_cos
        true_angles = np.arctan2(y_true[T:T+end_time, 1], y_true[T:T+end_time, 0])

        pred_radii = y_pred[:end_time, 2]
        true_radii = y_true[T:T+end_time, 2]

        worm_angles.append(pred_angles)
        worm_radii.append(pred_radii)
        worm_angles_real.append(true_angles)
        worm_radii_real.append(true_radii)

    worm_angles_list = worm_angles
    worm_angles = np.concatenate(worm_angles)
    worm_radii = np.concatenate(worm_radii)
    worm_angles_real = np.concatenate(worm_angles_real)
    worm_radii_real = np.concatenate(worm_radii_real)

    # Phase and radius over time
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 4))
    ax1.plot(worm_angles_real, label="True", alpha=0.7)
    ax1.plot(worm_angles, label="Predicted", linestyle="dashed")
    ax1.set_title("Predicted vs True Phase")
    ax1.set_ylabel("Phase")
    ax1.set_xlabel("Time")
    ax1.legend()

    ax2.plot(worm_radii_real, label="True")
    ax2.plot(worm_radii, label="Predicted", linestyle="dashed")
    ax2.set_title("Predicted vs True Radius")
    ax2.set_ylabel("Radius")
    ax2.set_xlabel("Time")
    ax2.legend()
    fig.tight_layout()

    # Trajectory in polar coordinates
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(worm_radii*np.cos(worm_angles), worm_radii*np.sin(worm_angles), c=worm_angles)
    ax1.set_title("Predicted trajectory")

    ax2.scatter(worm_radii_real*np.cos(worm_angles_real), worm_radii_real*np.sin(worm_angles_real), c=worm_angles_real)
    ax2.set_title("True trajectory")
    for ax in [ax1, ax2]:
        ax.axis("equal")
        ax.axis("off")
    fig.tight_layout()


    # Behavior color overlay
    behs = np.concatenate([z_i[T:T+len(ang)] for z_i, ang in zip(z, worm_angles_list)])
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(worm_radii*np.cos(worm_angles), worm_radii*np.sin(worm_angles), c=behs, cmap=cmap, vmin=0, vmax=6, alpha=0.1)
    ax2.scatter(worm_radii_real*np.cos(worm_angles_real), worm_radii_real*np.sin(worm_angles_real), c=behs, cmap=cmap, vmin=0, vmax=6, alpha=0.1)
    for ax in [ax1, ax2]:
        ax.axis("off")
        ax.axis("equal")
    fig.tight_layout()

    #sctter true vs pred angels 
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(worm_angles, worm_angles_real)
    ax.axis("off")
    ax.axis("equal")
    ax.set_xlabel("inferred")
    ax.set_xlabel("true")
    fig.tight_layout()

    # Optional 3D PCA plot
    if pca_x is not None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(pca_x[16000:18000, 0], pca_x[16000:18000, 1], pca_x[16000:18000, 2], alpha=0.3)
    return worm_angles, worm_radii


def plot_held_out_worm(model, inputs, y, T, stats, new_worm_idx, z,cmap, normalize=True):
    X_new_worm = np.array(inputs)[:,new_worm_idx,:].T
    y_new_worm = y[new_worm_idx]

    X_new_tensor = torch.tensor(X_new_worm, dtype=torch.float32)
    y_new_tensor = torch.tensor(y_new_worm, dtype=torch.float32)

    if T > 0:
        X_new_seq, y_new_seq = create_sequences(X_new_tensor, y_new_tensor, T)
    else:
        X_new_seq = X_new_tensor
        y_new_seq = y_new_tensor

    if normalize:
        # Normalize the inputs
        X_train_mean, X_train_std = X_new_seq.mean(axis=0), X_new_seq.std(axis=0)
        X_new_seq = (X_new_seq - X_train_mean) / (X_train_std + 1e-8)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_new_pred = model(X_new_seq).squeeze()

    # Denormalize the predicted outputs
    y_new_pred = y_new_pred * stats['y_std'] + stats['y_mean']  # Denormalize to original scale

    end_time = 1599
    test_angles = np.arctan2(y_new_worm[T:end_time, 1], y_new_worm[T:end_time, 0])
    pred_test_angles = np.arctan2(y_new_pred[:end_time, 1], y_new_pred[:end_time, 0])

    pred_test_rad = y_new_pred[:end_time, 2]
    test_rad = y_new_worm[T:end_time + T, 2]


    # Plot Phase and Radius
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(20, 4))
    ax1.plot((test_angles), label="True", alpha=0.7)
    ax1.plot((pred_test_angles), label="Predicted", linestyle="dashed")
    ax1.set_title(f"Predicted vs True for Worm {new_worm_idx}")
    ax1.set_ylabel("Phase")
    ax1.set_xlabel("Time")
    ax1.legend()

    ax2.plot(test_rad, label="True")
    ax2.plot(pred_test_rad, label="Predicted", linestyle="dashed")
    ax2.set_title(f"Predicted vs True for Worm {new_worm_idx}")
    ax2.set_ylabel("Radius")
    ax2.set_xlabel("Time")
    ax2.legend()
    fig.tight_layout()

    # Trajectory Plots
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(pred_test_rad * np.cos(pred_test_angles), pred_test_rad * np.sin(pred_test_angles), c=pred_test_angles)
    ax1.set_title(f"Predicted trajectory {new_worm_idx}")
    ax1.legend()

    ax2.scatter(test_rad * np.cos(test_angles), test_rad * np.sin(test_angles), c=pred_test_angles)
    ax2.set_title(f"Actual trajectory {new_worm_idx}")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()

    # Plot with Behavior Color Mapping
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    ax1.scatter(pred_test_rad * np.cos(pred_test_angles), pred_test_rad * np.sin(pred_test_angles), c=z[new_worm_idx][T:end_time], cmap=cmap, vmin=0, vmax=6, alpha=0.2)
    ax1.set_title(f"Predicted trajectory {new_worm_idx}")
    ax1.legend()

    ax2.scatter(test_rad * np.cos(test_angles), test_rad * np.sin(test_angles), c=z[new_worm_idx][T:end_time], cmap=cmap, vmin=0, vmax=6)
    ax2.set_title(f"Actual trajectory {new_worm_idx}")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(pred_test_angles, test_angles)
    ax.axis("off")
    ax.axis("equal")
    ax.set_xlabel("inferred")
    ax.set_xlabel("true")

    return pred_test_angles, pred_test_rad


def plot_donut_beh_specific(pred_test_rad,test_angles,  binary_array, ax1, colors ):
    # Plot with Behavior Color Mapping
    pred_test_rad_to_plot  = pred_test_rad[binary_array]
    test_angles_to_plot  = test_angles[binary_array]
  
    ax1.scatter(
                pred_test_rad_to_plot * np.cos(test_angles_to_plot),
                pred_test_rad_to_plot * np.sin(test_angles_to_plot),
                c=colors,
                vmin=0, vmax=6, alpha=0.2
                )


# rev_of_rev_turn_bin, pure_rev_bin, turn_of_rev_turn_bin, pure_turn_bin = make_masks(z)
# fwds_after_rev_turn = mark_5frame_zero_windows(z, turn_of_rev_turn_bin, window=5)
# fwds_after_pure_rev = mark_5frame_zero_windows(z, pure_rev_bin, window=5)


# fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)

# plot_donut_beh_specific(pred_test_rad,test_angles,  z==1 , ax1, "dodgerblue" )
# plot_donut_beh_specific(pred_test_rad,test_angles, pure_rev_bin, ax2, "dodgerblue" )
# plot_donut_beh_specific(pred_test_rad,test_angles, rev_of_rev_turn_bin, ax3, "dodgerblue" )
# plot_donut_beh_specific(pred_test_rad,test_angles, z,  turn_of_rev_turn_bin, ax4, "cyan" )
# plot_donut_beh_specific(pred_test_rad,test_angles, z,  pure_turn_bin, ax5, "cyan" )


# fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(1,5)
# plot_donut_beh_specific(pred_test_rad,test_angles, z==0, ax2, "dodgerblue" )
# plot_donut_beh_specific(pred_test_rad,test_angles, fwds_after_rev_turn, ax2, "dodgerblue" )
# plot_donut_beh_specific(pred_test_rad,test_angles, fwds_after_rev_turn, ax3, "dodgerblue" )