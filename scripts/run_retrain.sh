#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"
cd "${PROJECT_DIR}"

# Define variables for retrain experiment
dataset="cifar10"
network_config="default"

exp_num="17"
exp="exp1"

repeat="1"
echo "Starting $exp_num F13 repeat $repeat"
exp_path="${PROJECT_DIR}/experiment_${dataset}_acc_${exp_num}/${exp}_repeat_${repeat}"

# Retrain model
CUDA_VISIBLE_DEVICES=0 uv run python "${SRC_DIR}/retrain_model.py" \
    --experiment_path "$exp_path" \
    --data_path "${PROJECT_DIR}/${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F13 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --network_config "$network_config" \
    --optimizer "AdamW"


repeat="2"
echo "Starting $exp_num F13 repeat $repeat"
exp_path="experiment_${dataset}_acc_${exp_num}/${exp}_repeat_${repeat}"

# Retrain model
CUDA_VISIBLE_DEVICES=0 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F13 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --network_config "$network_config" \
    --optimizer "AdamW"

repeat="3"
echo "Starting $exp_num F13 repeat $repeat"
exp_path="experiment_${dataset}_acc_${exp_num}/${exp}_repeat_${repeat}"

# Retrain model
CUDA_VISIBLE_DEVICES=0 python retrain_model.py \
    --experiment_path "$exp_path" \
    --data_path "${dataset}_data" \
    --dataset "$dataset" \
    --retrain_folder retrain \
    --config_code F13 \
    --log_level INFO \
    --max_epochs 300 \
    --batch_size 256 \
    --eval_batch_size 256 \
    --device cuda:0 \
    --num_repetitions 3 \
    --lr_scheduler "multistep" \
    --data_augmentation \
    --network_config "$network_config" \
    --optimizer "AdamW"    

# Check if the previous command was successful
if [ $? -ne 0 ]; then
    echo "Error: Retrain model script failed."
    exit 1
fi

echo "Retrain model script completed successfully."
