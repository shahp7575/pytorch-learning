# =================================================================== #
# =========================== IMPORTS ============================== #
# =================================================================== #
import torch
import torchvision
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
# =============================== SET DEVICE =============================== #
# ========================================================================== #

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# ========================================================================== #
# ============================ HYPERPARAMETERS ============================= #
# ========================================================================== #

# For MNIST it is Nx1x28x28.
# So we can view this as we have 28 time sequences and each sequence has 28 features.

input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256 # nodes in the hidden layer
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# ========================================================================== #
# ============================== CREATE RNN ================================ #
# ========================================================================== #

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True) # N (batch_size) x time_seq x features
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # forward prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1) # keep the batch and then concatenate everything
        out = self.fc(out)

        return out

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

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

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
        data = data.to(device=device).squeeze(1) # 
        targets = targets.to(device=device)
        
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
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

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
