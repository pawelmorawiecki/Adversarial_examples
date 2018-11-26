# Fully-connected model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


def train(net, device, train_loader):    
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # run the main training loop
    for epoch in range(10):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28*28)
            optimizer.zero_grad()
            net_out = net(data)
            loss = F.nll_loss(net_out, target)
            loss.backward()
            optimizer.step()

            
            
def test(model, device, test_loader):
    
    correct = 0
    
    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        data = data.view(-1, 28*28)

        # Forward pass the data through the model
        output = model(data)
        prediction = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        if prediction.item() == target.item():
            correct += 1
    
    acc = correct/float(len(test_loader))
    print(f'Accuracy: {acc}')
    
    
class Fully_connected(nn.Module):
    def __init__(self):
        super(Fully_connected, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)