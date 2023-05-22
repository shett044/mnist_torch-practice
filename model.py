import torch.nn as nn
class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size) :
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, X):
        act1 = self.relu(self.lin1(X))
        return self.lin2(act1)
