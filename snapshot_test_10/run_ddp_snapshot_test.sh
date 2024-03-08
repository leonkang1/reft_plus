batch_size=2048
data_size=$((400 * ${batch_size}))
num_epochs=1
seed=42

times=1

options_0=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --seed ${seed} \
    --enable_profiling \
    --option_num 0 \
"

run_options_0() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --nproc_per_node=4 snapshot_test.py ${options_0}
    done
}

options_1=" \
    --batch_size ${batch_size} \
    --data_size ${data_size} \
    --epochs ${num_epochs} \
    --seed ${seed} \
    --enable_profiling \
    --use_snapshot \
    --use_copy \
    --async_snapshot \
    --option_num 1 \
    --non_blocking_copy \
    --use_pin_memory
"

run_options_1() {
    for i in $(seq 1 ${times}); do
        CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --nproc_per_node=4 snapshot_test.py ${options_1}
    done
}

run_options_1