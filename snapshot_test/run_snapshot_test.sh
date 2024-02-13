# Run snapshotting test
batch_size=1024
data_size=$((20 * ${batch_size}))
num_epochs=1
seed=42

# no snapshot use_timer
# options=" \
#     --batch_size ${batch_size} \
#     --data_size ${data_size} \
#     --epochs ${num_epochs} \
#     --seed ${seed} \
#     --use_timer \
# "
# use_snapshot ues_copy_ async_snapshot use_timer
options=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --seed ${seed} \
    --use_timer \
    --use_snapshot \
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
#     --seed ${seed} \
# "

CUDA_VISIBLE_DEVICES=2 python snapshot_test_single_gpu.py ${options}
