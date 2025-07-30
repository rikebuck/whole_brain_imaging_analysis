import numpy as np
import torch

def create_sequences(X, y, T):
    X_seq, y_seq = [], []
    for i in range(len(X) - T):
        X_seq.append(X[i:i+T])
        y_seq.append(y[i+T])
    return torch.stack(X_seq), torch.stack(y_seq)

    
def prepare_data(inputs, targets, T=5, holdout_idxs=[20, 21]):
    all_worms = np.arange(len(inputs[0]))
    to_train = ~np.isin(all_worms, holdout_idxs)
    held_in_worms = all_worms[to_train]

    X_all_worms, y_all_worms = [], []
    for worm_idx in held_in_worms:

        X_single_worm = np.stack(np.array(inputs)[:, worm_idx,:], axis=1)
        y_single_worm = targets[worm_idx]

        if T != 0:
            X_seq, y_seq = create_sequences(
                torch.tensor(X_single_worm, dtype=torch.float32),
                torch.tensor(y_single_worm, dtype=torch.float32),
                T
            )
            X_all_worms.append(X_seq)
            y_all_worms.append(y_seq)
        else:
            X_all_worms.append(torch.tensor(X_single_worm, dtype=torch.float32))
            y_all_worms.append(torch.tensor(y_single_worm, dtype=torch.float32))

    X_all = torch.cat(X_all_worms, dim=0)
    y_all = torch.cat(y_all_worms, dim=0)

    return X_all, y_all

def split_and_normalize(X_all, y_all, train_ratio=0.8, normalize=True):
    indices = np.arange(len(X_all))
    np.random.shuffle(indices)
    X_all, y_all = X_all[indices], y_all[indices]

    split_idx = int(train_ratio * len(X_all))
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]

    # Normalize using training stats
    X_mean, X_std = X_train.mean(dim=0), X_train.std(dim=0)
    y_mean, y_std = y_train.mean(), y_train.std()
    # y_mean, y_std = 0,1
    if normalize:
        X_train = (X_train - X_mean) / (X_std + 1e-8)
        X_test = (X_test - X_mean) / (X_std + 1e-8)

    y_train = (y_train - y_mean) / (y_std + 1e-8)
    y_test = (y_test - y_mean) / (y_std + 1e-8)

    stats = {
        'X_mean': X_mean, 'X_std': X_std,
        'y_mean': y_mean, 'y_std': y_std
    }

    return X_train, X_test, y_train, y_test, stats
