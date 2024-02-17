# Run snapshotting test
batch_size=4096
data_size=$((100 * ${batch_size}))
num_epochs=1
seed=42

times=1

# no snapshot use_timer
options_0=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --seed ${seed} \
    --use_timer \
    --option_num 0 \
"

run_options_0() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=1 python snapshot_test_single_gpu.py ${options_0}
    done
}

# use_snapshot use_copy_ async_snapshot use_timer
options_1=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --seed ${seed} \
    --use_timer \
    --use_snapshot \
    --use_copy \
    --async_snapshot \
    --option_num 1 \
"

run_options_1() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=3 python snapshot_test_single_gpu.py ${options_1}
    done
}

# snapshot with copy_ using non_blocking
options_2=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --use_timer \
    --use_snapshot \
    --use_copy \
    --non_blocking_copy \
    --seed ${seed} \
    --option_num 2 \
"

run_options_2() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=4 python snapshot_test_single_gpu.py ${options_2}
    done
}

# use_snapshot no async_snapshot blocking
options_3=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --use_timer \
    --use_snapshot \
    --use_copy \
    --seed ${seed} \
    --option_num 3 \
"

run_options_3() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=5 python snapshot_test_single_gpu.py ${options_3}
    done
}

run_options_0 &
run_options_1 &
run_options_2 &
run_options_3 &

wait

echo "All scripts have completed."

# use synthetic input

# no snapshot use_timer
# options=" \
#     --batch_size ${batch_size} \
#     --data_size ${data_size} \
#     --epochs ${num_epochs} \
#     --seed ${seed} \
#     --use_timer \
#     --use_synthetic_input \
# "

# use_snapshot use_copy_ async_snapshot use_timer
# options=" \
#     --batch_size ${batch_size} \
#     --data_size ${data_size} \
#     --epochs ${num_epochs} \
#     --seed ${seed} \
#     --use_timer \
#     --use_snapshot \
#     --use_copy \
#     --async_snapshot \
# "

# snapshot with copy_ using non_blocking
# options=" \
#     --batch_size ${batch_size} \
#     --data_size ${data_size} \
#     --epochs ${num_epochs} \
#     --use_timer \
#     --use_snapshot \
#     --use_copy \
#     --non_blocking_copy \
#     --seed ${seed} \
# "

# use_snapshot no async_snapshot blocking
# options=" \
#     --batch_size ${batch_size} \
#     --data_size ${data_size} \
#     --epochs ${num_epochs} \
#     --use_timer \
#     --use_snapshot \
#     --use_copy \
#     --seed ${seed} \
#     --use_synthetic_input
# "


