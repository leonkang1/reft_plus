import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.input_layer = nn.Linear(512, 4096)
        self.hidden_layers = nn.ModuleList([nn.Linear(4096, 4096) for _ in range(64)])
        self.output_layer = nn.Linear(4096, 512)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
