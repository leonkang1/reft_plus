import torch
import torch.nn as nn
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __len__(self):
        return 128

    def __getitem__(self, index):
        return torch.randn(512), torch.randn(1024)  # Simulated input and output

class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, device_list):
        super(ParallelLinear, self).__init__()
        self.devices = device_list
        num_devices = len(device_list)
        
        # Split the weights and bias for as many devices as specified
        self.split_weights = nn.ParameterList([
            nn.Parameter(torch.randn(out_features // num_devices, in_features).to(device))
            for device in self.devices
        ])
        self.split_biases = nn.ParameterList([
            nn.Parameter(torch.randn(out_features // num_devices).to(device))
            for device in self.devices
        ])

    def forward(self, x):
        outputs = []
        for i, device in enumerate(self.devices):
            xi = x.to(device)
            output = xi @ self.split_weights[i].t() + self.split_biases[i]
            outputs.append(output)
        
        # Concatenate results across devices and return
        y = torch.cat(outputs, dim=1)
        return y

class EnhancedModel(nn.Module):
    def __init__(self, device_list):
        super(EnhancedModel, self).__init__()
        
        self.device_list = device_list
        self.input_layer = nn.Linear(512, 1024).to(device_list[0])
        
        # Only create ParallelLinear without to(device)
        self.parallel_layers = nn.ModuleList([
            ParallelLinear(1024, 1024, device_list) for _ in range(4)
        ])
        
        self.output_layer = nn.Linear(1024, 1024).to(device_list[0])

    def forward(self, x):
        x = self.input_layer(x)
        
        for i, layer in enumerate(self.parallel_layers):
            device = self.device_list[i % len(self.device_list)]
            layer.to(device)
            x = layer(x.to(device))

        x = self.output_layer(x.to(self.device_list[0]))
        return x