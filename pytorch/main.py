import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# can be used to use subsets (e.g. a single image) of the MNIST dataset
from torch.utils.data import Subset

# define hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# download MNIST (train & test) dataset
train_dataset = torchvision.datasets.MNIST(root='./mnist-data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./mnist-data',
                                          train=False,
                                          transform=transforms.ToTensor())

# load data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# fully connected neural network
class NeuralNet(nn.Module):

    def __init__(self, nn_input_size, nn_hidden_size, nn_num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(nn_input_size, nn_hidden_size)
        self.fc2 = nn.Linear(nn_hidden_size, nn_num_classes)

    # sequence: fully-connected -> ReLU -> fully-connected -> softmax
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)

        return out


model = NeuralNet(input_size, hidden_size, num_classes)

# loss function: cross entropy
loss_func = nn.CrossEntropyLoss()

# optimizer: stochastic gradient descent (NOTE: implementing momentum is not necessary for the project)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# training phase
model.train()

# loop over epochs
for epoch in range(num_epochs):
    # loop over training dataset
    for i, (images, labels) in enumerate(train_loader):
        # transform image array of shape (batch_size, 1, 28, 28) to (batch_size, 784)
        images = images.reshape(-1, 28 * 28)

        # exec forward pass
        outputs = model(images)

        # compute loss
        loss = loss_func(outputs, labels)

        # exec backward pass to optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log loss over time
        if i % 100 == 0:
            print('Epoch {}, Step {}, Loss: {:.6f}'.format(epoch, i, loss.item()))

# testing phase
model.eval()

# no grad computation required since there is no backward pass
with torch.no_grad():
    # counter variables to compute accuracy
    correct = 0
    total = 0

    # loop over testing dataset
    for images, labels in test_loader:
        # transform image array of shape (batch_size, 1, 28, 28) to (batch_size, 784)
        images = images.reshape(-1, 28 * 28)

        # exec forward pass
        outputs = model(images)

        # evaluate prediction: compute fraction of correct predictions
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy: {} %'.format(100. * correct / total))
