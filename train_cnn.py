import os
import argparse
import pickle as pk
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from CNN import *
os.environ['KMP_DUPLICATE_LIB_OK']='True'

use_gpu = False
num_timesteps_input = 5
num_timesteps_output = 1

epochs = 50
batch_size = 100

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--enable-cuda', action='store_true',
                    help='Enable CUDA')
args = parser.parse_args()
args.device = torch.device('cuda')


def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])

    epoch_training_losses = []
    for i in range(0, training_input.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.type(torch.FloatTensor).to(device=args.device)
        y_batch = y_batch.type(torch.FloatTensor).to(device=args.device)
        out = net(X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        epoch_training_losses.append(loss.detach().cpu().numpy())
        print("\r", " batch:%d train_loss: %.10f" % (int(i/batch_size), loss.item()), end='', flush=True)
    return sum(epoch_training_losses)/len(epoch_training_losses)


if __name__ == '__main__':
    torch.manual_seed(7)

    X = load_data()
    split_line1 = int(X.shape[0] * 0.8)
    split_line2 = int(X.shape[0] * 0.9)

    train_original_data = X[:split_line1]
    val_original_data = X[split_line1:split_line2]
    test_original_data = X[split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)

    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)

    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)


    net = CNN(training_input.shape[2],num_timesteps_input,num_timesteps_output).cuda()

    #reshape
    training_input = training_input.reshape(training_input.shape[0], -1, training_input.shape[3], training_input.shape[4])
    val_input = val_input.reshape(val_input.shape[0], -1, val_input.shape[3], val_input.shape[4])
    test_input = test_input.reshape(test_input.shape[0], -1, test_input.shape[3], test_input.shape[4])

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()
    training_loss = []
    validation_loss=100
    for epoch in range(epochs):
        loss = train_epoch(training_input, training_target,
                           batch_size=batch_size)
        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.type(torch.FloatTensor).to(device=args.device)
            val_target = val_target.type(torch.FloatTensor).to(device=args.device)
            out = net(val_input)
            rmse_loss = torch.sqrt( loss_criterion(out, val_target)).to(device="cpu").detach().numpy().item()
            out =out.detach().cpu().numpy()
            val_target_ =val_target.detach().cpu().numpy()
            mape_loss = np.fabs((val_target_-out)/np.clip(val_target_,0.1,1)).mean()
            out = None
        print("\n******************************** ")
        print("\r", " epoch:%d MAPE: %.10f RMSE: %.10f " % (epoch, mape_loss, rmse_loss))
        print("******************************** ")
        if validation_loss>rmse_loss:
            validation_loss=rmse_loss
            torch.save(net, './model/best_cnn_model.pkl')

