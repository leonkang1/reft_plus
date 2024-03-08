import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
from deepspeed.utils import safe_get_local_fp32_param, safe_get_full_fp32_param, safe_get_full_optimizer_state, \
safe_get_local_optimizer_state, safe_get_local_grad, safe_get_full_grad


# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)

# Initialize the model and the optimizer
model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# DeepSpeed configuration
config = {
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.001
        }
    },
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    "offload_optimizer": {
        "device": "cpu",
        "pin_memory": True,
    },
    "train_micro_batch_size_per_gpu": 8  # Add this line
}

deepspeed.init_distributed()

# Initialize DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(model=model, 
                                              optimizer=optimizer, 
                                              config=config)


# Create a random dataset and a DataLoader
inputs = torch.randn(100, 10).to(model.local_rank)
targets = torch.randint(0, 2, (100, 1)).float().to(model.local_rank)
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=8)

# Training loop
for epoch in range(2):  # 10 epochs
    for batch in dataloader:
        inputs, targets = batch
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs, targets)

        model.backward(loss)  # backward pass
        model.step()  # update weights
        
for n, lp in model.named_parameters():
    hp = safe_get_full_fp32_param(lp)
    hp_grad = safe_get_full_grad(lp)
    exp_avg = safe_get_full_optimizer_state(lp, "exp_avg")
    exp_avg_sq = safe_get_full_optimizer_state(lp, "exp_avg_sq")
    if model.local_rank == 0:
        local_hp = safe_get_local_fp32_param(lp)
        local_hp_grad = safe_get_local_grad(lp)
        local_exp_avg = safe_get_local_optimizer_state(lp, "exp_avg")
        local_exp_avg_sq = safe_get_local_optimizer_state(lp, "exp_avg_sq")
        print("name", n)
        print("lp", lp)
        print("hp", hp)
        print("exp_avg", exp_avg)
        print("exp_avg_sq", exp_avg_sq)
        
        print("local_hp", local_hp)
        print("local_exp_avg", local_exp_avg)
        print("local_exp_avg_sq", local_exp_avg_sq)
        
    