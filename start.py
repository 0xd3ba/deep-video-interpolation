# start.py -- Module containing the code for training/testing

import torch
import torch.nn as nn
import torch.optim as optim


def train(model, data_loader, n_epochs, chkpt_epochs, chkpt_dir, device):
    """ Starts the training of the model """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()

    # Start the training
    for e in range(n_epochs):
        epoch_loss = 0

        for prev_frame, next_frame, target_frame in data_loader:
            prev_frame = prev_frame.to(device)
            next_frame = next_frame.to(device)
            target_frame = target_frame.to(device)

            # Predict the middle frame using the model
            pred_frame = model(prev_frame, next_frame)

            # Compute the MSE loss between the two frames
            loss = mse_loss(pred_frame, target_frame)

            # Compute the gradients and back-propagate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add the loss to check the training progress
            epoch_loss += loss.cpu().item()

        # TODO: Save the model
        if e % chkpt_epochs == 0:
            pass

        print(f'Epoch[{e}]: {epoch_loss}')


def test(model, data_loader, output_dir, gpu_available):
    """ Starts the testing of the model """
    raise NotImplementedError
