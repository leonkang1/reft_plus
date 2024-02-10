# Run snapshotting test
batch_size=1024
data_size=$((100 * ${batch_size}))
num_epochs=1

# no snapshot
# options=" \
#     --batch_size ${batch_size} \
#     --data_size ${data_size} \
#     --num_epochs ${num_epochs} \
# "
# snapshot with copy_ async_snapshot
options=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --num_epochs ${num_epochs} \
    --use_copy \
    --async_snapshot \
"
# snapshot with copy_ using non_blocking
# options=" \
#     --batch_size ${batch_size} \
#     --data_size ${data_size} \
#     --num_epochs ${num_epochs} \
#     --use_copy \
#     --non_blocking_copy \
# "

CUDA_VISIBLE_DEVICES=0 python snapshot_test_single_gpu.py ${options}
