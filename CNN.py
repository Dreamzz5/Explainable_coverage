import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):

    def __init__(self,num_features, num_timesteps_input,
                 num_timesteps_output):
        super(CNN, self).__init__()
        self.num_features=num_features
        self.num_timesteps_input=num_timesteps_input
        self.CNN1 = nn.Conv2d(10, 32, kernel_size=3, stride=1, padding=1, bias= True)
        self.CNN2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.CNN3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.CNN4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.CNN5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.CNN6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.CNN7 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1, bias=True)
    def forward(self, X):
        X = self.CNN1(X)
        X = self.CNN2(X)
        X = self.CNN3(X)
        X = self.CNN4(X)
        X = self.CNN5(X)
        X = self.CNN6(X)
        X = F.relu(self.CNN7(X))
        return X
def load_data():
    data = np.load("data/Chengdu_train_30day.npy")
    data = data/np.max(data)
    return data

def generate_dataset(X, num_timesteps_input, num_timesteps_output):

    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[0] - (
                num_timesteps_input + num_timesteps_output) + 1)]
    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(X[i: i + num_timesteps_input])
        target.append(X[i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target)).squeeze()