# imports
import torch
import torch.nn as nn

######
# input_dim : size of the input at each time step
# hidden_dim : size of the hidden state and cell state
# num_layers : number of LSTM layers stacked on top of each other
######

input_dim = 5
hidden_dim = 10
n_layers = 1

lstm_layer = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)

#### Random Testing
batch_size = 1
seq_len = 3

inp = torch.randn(batch_size, seq_len, input_dim)
hidden_state = torch.randn(n_layers, batch_size, hidden_dim)
cell_state = torch.randn(n_layers, batch_size, hidden_dim)
hidden = (hidden_state, cell_state)

# feeding the input and hidden states
out, hidden = lstm_layer(inp, hidden)
print(out.shape)
print(hidden)