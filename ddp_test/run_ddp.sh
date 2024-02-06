# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 raim5_system.py
torchrun --nnodes=1 --nproc-per-node=4 raim5_system.py
