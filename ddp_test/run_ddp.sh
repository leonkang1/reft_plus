# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nnodes=1 --nproc-per-node=4 raim5_system.py
# torchrun --nnodes=1 --nproc-per-node=4 raim5_system.py


# Run snapshotting test
batch_size=1024
data_size=$((100 * ${batch_size}))

# no snapshot
options=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --num_epochs 1 \
"
# snapshot with copy_ async_snapshot
options=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --num_epochs 1 \
    --use_copy \
    --async_snapshot \
"
# snapshot with copy_ using non_blocking
options=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --num_epochs 1 \
    --use_copy \
    --non_blocking_copy \
"

CUDA_VISIBLE_DEVICES=3,4,5,6 torchrun --nproc_per_node=4 snapshot_test.py ${options}

# Run get parity and ckpt
# torchrun --nproc_per_node=4 get_parity_n_ckpt.py