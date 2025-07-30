import torch
import torch.nn as nn
import torch.optim as optim
from angle.models.lstm_model import LSTM
from angle.models.ff_model import FeedForward

def train_model(X_train, y_train, X_test, y_test,
                input_dim, output_dim,
                model_type='lstm',
                hidden_dim=64,
                num_layers=3,
                dropout=0.5,
                learning_rate=0.001,
                num_epochs=500):

    if model_type == 'lstm':
        model = LSTM(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
    elif model_type == 'ff':
        model = FeedForward(input_dim, hidden_dim, output_dim, num_layers=num_layers, dropout=dropout)
        # Flatten input
        X_train = X_train.view(X_train.shape[0], -1)
        X_test = X_test.view(X_test.shape[0], -1)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test).item()

        train_losses.append(train_loss.item())
        test_losses.append(test_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss:.4f}")

    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")

    return model, train_losses, test_losses
