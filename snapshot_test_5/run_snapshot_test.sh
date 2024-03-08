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
        CUDA_VISIBLE_DEVICES=0 python snapshot_test_single_gpu.py ${options_0}
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
        CUDA_VISIBLE_DEVICES=1 python snapshot_test_single_gpu.py ${options_1}
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
        CUDA_VISIBLE_DEVICES=3 python snapshot_test_single_gpu.py ${options_2}
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


# no snapshot enable_profiling
options_4=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --seed ${seed} \
    --enable_profiling \
    --option_num 4 \
"

run_options_4() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=1 python snapshot_test_single_gpu.py ${options_4}
    done
}

# use_snapshot use_copy_ async_snapshot enable_profiling
options_5=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --seed ${seed} \
    --enable_profiling \
    --use_snapshot \
    --use_copy \
    --async_snapshot \
    --option_num 5 \
"

run_options_5() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=2 python snapshot_test_single_gpu.py ${options_5}
    done
}

# snapshot with copy_ using non_blocking
options_6=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --enable_profiling \
    --use_snapshot \
    --use_copy \
    --non_blocking_copy \
    --seed ${seed} \
    --option_num 6 \
"

run_options_6() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=3 python snapshot_test_single_gpu.py ${options_6}
    done
}

run_options_4 &
run_options_5 &


wait

echo "All scripts have completed."

