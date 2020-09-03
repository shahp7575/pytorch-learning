# imports
import torch
from torch import nn
import numpy as np

# define sentences
text = ['hey how are you', 'good i am fine', 'have a nice day']

# join all sents together and extract the unique characters from the combined sentences
chars = set(''.join(text))

# creating a dictionary that maps integers to chars
int2char = dict(enumerate(chars))

# dict mapping chars to ints
char2int = {char: idx for idx, char in int2char.items()}

# padding input sentences to ensure all sentences are of standard length
# RNNs typically manage variably sized inputs, but we usually feed training data in batches
# to speed up the training process. In order to use batches to train we must ensure that
# each sequence within the input data is of equal size.

# Finding length of the longest string in our data
maxlen = len(max(text, key=len))

# Padding
for i in range(len(text)):
    while len(text[i]) < maxlen:
        text[i] += ' '



#########
# For predicting the next character in the sequence
# We divide each sentence into
# Input Data : The last input character should be excluded as it doesn't need to be fed into the model
# Target : One time-step ahead of the input data as this will be the 'correct answer'
#########

# lists that will hold our input and target sequences
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])

    # Remove first character for target sequence
    target_seq.append(text[i][1:])

    print(f"Input Sequence : {input_seq[i]}, Target Sequence : {target_seq[i]}")

# convert out input and target sequences to sequence of integers
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

#########
# Before converting them into one-hot vectors we define 3 key variables
# 1) dict_size : Number of unique characters we have in our text
# 2) seq_len : Length of sequences we are feeding into the model; maxlength - 1 as we removed last character input
# 3) batch_size : Number of sentences we are going to feed into our model
#########

dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):

    # Create a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # One-hot encoding
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)

# numpy array to PyTorch tensor
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)

# set device
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

########
# BUILD RNN
########
class Model(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim, n_layers):

        super(Model, self).__init__()

        # Defining params
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Defining layers
        ## RNN
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        ## FCC
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        batch_size = x.size(0)

        # init hidden state for first input 
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # reshaping the outputs so it can be fit into fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):

        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)

        return hidden

# Instantiate model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
model = model.to(device)

# Hyperparams
n_epochs = 100
lr = 0.01

# Define loss, optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

##### 
# TRAINING
#####
input_seq = input_seq.to(device)

for epoch in range(1, n_epochs + 1):
    
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    output, hidden = model(input_seq)
    output = output.to(device)
    target_seq = target_seq.to(device)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Backprops and calculates gradients
    optimizer.step() # Updates the weights accordingly

    if epoch % 10 == 0:
        print(f"Epoch : {epoch}/{n_epochs}")
        print(f"Loss : {loss:.4f}")
            

######
# TESTING
######
def predict(model, character):
    # one-hot encoding input
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    character = character.to(device)

    out, hidden = model(character)

    prob = nn.functional.softmax(out[-1], dim=0).data

    # highest prob
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

print("GENERATED SENTENCE -> ", sample(model, 15, 'hey'))