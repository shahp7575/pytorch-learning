# =================================================================== #
# =========================== IMPORTS ============================== #
# =================================================================== #
import torch
# all neural network modules
import torch.nn as nn
# all the optimizers (Adam, SGD etc.)
import torch.optim as optim
# all the functions that don't have any parameters. (activation functions : tanh, ReLu)
import torch.nn.functional as F
# easier dataset management using DataLoader
from torch.utils.data import DataLoader
# standard datasets
import torchvision.datasets as datasets
# transformations that we can perform on our dataset
import torchvision.transforms as transforms

# ========================================================================== #
# ================= CREATE SIMPLE FULLY CONNECTED NETWORK ================== #
# ========================================================================== #

class NN(nn.Module):

    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # 1st linear layer. 50 hidden nodes
        self.fc1 = nn.Linear(input_size, 50)
        # 2nd layer. 50 to number of classes
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ========================================================================== #
# =============================== SET DEVICE =============================== #
# ========================================================================== #

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# ========================================================================== #
# ============================ HYPERPARAMETERS ============================= #
# ========================================================================== #

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# ========================================================================== #
# =============================== LOAD DATA ================================ #
# ========================================================================== #

# training set
train_dataset = datasets.MNIST(root='dataset/', 
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# test set
test_dataset = datasets.MNIST(root='dataset/', 
                               train=False, 
                               transform=transforms.ToTensor(),
                               download=True)

test_loader = DataLoader(dataset=test_dataset,
                          batch_size=batch_size,
                          shuffle=True)

# ========================================================================== #
# ========================== INITIALIZE NETWORK ============================ #
# ========================================================================== #

model = NN(input_size=input_size,
           num_classes=num_classes).to(device)

# ========================================================================== #
# ============================ LOSS & OPTIMIZER ============================ #
# ========================================================================== #

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate)

# ========================================================================== #
# ============================= TRAIN NETWORK ============================== #
# ========================================================================== #

# 1 epoch means the network has seen all the images in the dataset
for epoch in range(num_epochs):
    # go through each batch. enumerate to see which batch it is
    for batch_idx, (data, targets) in enumerate(train_loader):
        
        # get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # unroll the data to make into one long vector
        # since the MNIST image is 1x28x28. we want to unroll it to make it into a single vector
        data = data.reshape(data.shape[0], -1)
        
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        # we want to set all gradients to zero for each batch
        # so that it doesn't store the backprop calculations for the previous batch
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        # update the weights depending on the gradients computed
        optimizer.step()

# ========================================================================== #
# ======== CHECK ACCURACY ON TRAINING AND SEE HOW GOOD THE MODEL IS ======== #
# ========================================================================== #

# Check accuracy on training set and see how good the model is

def check_accuracy(loader, model):

    num_correct = 0
    num_samples = 0
    model.eval()

    # when we want to check the accuracy we don't have to actually compute the gradients
    # so we do torch.no_grad() to let PyTorch know we don't need gradients for this step
    with torch.no_grad():
        # to print if it is running on train or test data
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")

        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1) # index of the max value
            
            # predictions that are equal to the correct label
            num_correct += (predictions == y).sum()
            # size of the first dimension. (64)
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/(num_samples)*100: .2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
