#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"
cd "${PROJECT_DIR}"

dataset="cifar10"
exp_path_base="experiment_${dataset}_acc_17"
config_file="config_files_cifar"
fitness_metric="best_accuracy"
data_path="${PROJECT_DIR}/${dataset}_data"
log_level="INFO"
network_config="default"
dataset_sample_size=10000

configs=("config15.txt")
exps=("exp1")
cuda_devices=("0,1")

# Loop over the length of the configs array
for ((j=0; j<${#configs[@]}; j++)); do
    config="${configs[$j]}"
    exp="${exps[$j]}"
    cuda_device="${cuda_devices[$j]}"

    echo "Running evolution experiment with $config"

    for ((i=1; i<=3; i++)); do # Change the range to the number of repeats
        exp_path="${exp_path_base}/${exp}_repeat_$i"

        CUDA_VISIBLE_DEVICES="$cuda_device" uv run python "${SRC_DIR}/run_evolution.py" \
            --experiment_path "$exp_path" \
            --config_file "${config_dir}/${config}" \
            --data_path "$data_path" \
            --dataset "$dataset" \
            --limit_data_value "$dataset_sample_size" \
            --fitness_metric "$fitness_metric" \
            --early_stopping \
            --network_config "$network_config" \
            --en_pop_crossover \
            --log_level "$log_level"
    done
done
