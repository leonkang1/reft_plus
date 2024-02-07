# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 raim5_system.py
# torchrun --nnodes=1 --nproc-per-node=4 raim5_system.py


# Run snapshotting test
options="\
    --data_size=512\
    --use_snapshot \
    --use_copy \
    "
CUDA_VISIBLE_DEVICES=3,4,6,7 torchrun --nproc_per_node=4 snapshot_test.py ${options}

# Run get parity and ckpt
# torchrun --nproc_per_node=4 get_parity_n_ckpt.py