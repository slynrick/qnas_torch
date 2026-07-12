#!/bin/bash
# Runs the full Q-NAS pipeline in order: evolve -> retrain -> infographic.
# Each step only runs if the previous one succeeded.
#
# Usage:
#   scripts/run_pipeline.sh -e experiment_cifar10_progressive/exp1 \
#       -c configs/config_files_cifar/config_progressive.yml -d cifar10
#
# Required:
#   -e  experiment_path (also used for retrain and infographic)
#   -c  config_file for evolution (path relative to project root)
#   -d  dataset (default: cifar10)
#
# Optional (evolution):
#   -n  network_config (default: default)
#   -m  fitness_metric override for run_evolution.py (omit to use config file's value)
#   -C  continue_path (resume a previous run instead of starting fresh)
#   -X  pass to enable population crossover (en_pop_crossover flag, no value)
#
# Optional (retrain):
#   -g  config_code, used for output filenames (default: run1)
#   -E  retrain max_epochs (default: 300)
#   -R  retrain num_repetitions (default: 1)
#   -L  retrain lr_scheduler: cosine|reduce_on_plateau|exponential|multistep|None (default: None)
#   -A  pass to enable retrain data_augmentation (flag, no value)
#   -T  pass to enable retrain early stopping (flag, no value)
#   -P  retrain early_stopping_patience (default: 10, only used if -T is passed)
#   -M  pass to enable AMP mixed precision during retrain (default: FP32, flag, no value)
#
# Optional (skip steps, e.g. to only regenerate the infographic):
#   -S  comma-separated steps to skip: evolve,retrain,infographic (default: none)

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/path_config.sh"
cd "${PROJECT_DIR}"

dataset="cifar10"
network_config="default"
fitness_metric=""
continue_path=""
config_code="run1"
retrain_max_epochs=300
retrain_num_repetitions=1
lr_scheduler="None"
data_augmentation_flag=""
early_stopping_flag=""
early_stopping_patience=10
en_pop_crossover_flag=""
mixed_precision_flag=""
skip_steps=""

while getopts "e:c:d:n:m:C:g:E:R:L:ATP:S:XM" opt; do
    case "$opt" in
        e) experiment_path="$OPTARG" ;;
        c) config_file="$OPTARG" ;;
        d) dataset="$OPTARG" ;;
        n) network_config="$OPTARG" ;;
        m) fitness_metric="$OPTARG" ;;
        C) continue_path="$OPTARG" ;;
        g) config_code="$OPTARG" ;;
        E) retrain_max_epochs="$OPTARG" ;;
        R) retrain_num_repetitions="$OPTARG" ;;
        L) lr_scheduler="$OPTARG" ;;
        A) data_augmentation_flag="--data_augmentation" ;;
        T) early_stopping_flag="--early_stopping_enabled" ;;
        P) early_stopping_patience="$OPTARG" ;;
        S) skip_steps="$OPTARG" ;;
        X) en_pop_crossover_flag="--en_pop_crossover" ;;
        M) mixed_precision_flag="--mixed_precision" ;;
        *) echo "Unknown option"; exit 1 ;;
    esac
done

if [[ -z "$experiment_path" ]]; then
    echo "Error: -e experiment_path is required." >&2
    exit 1
fi

should_run() {
    # should_run <step_name> - true unless step_name is in the comma-separated skip_steps
    [[ ",${skip_steps}," != *",$1,"* ]]
}

data_path="${PROJECT_DIR}/${dataset}_data"

if should_run "evolve"; then
    if [[ -z "$config_file" ]]; then
        echo "Error: -c config_file is required to run the evolve step (or pass -S evolve to skip it)." >&2
        exit 1
    fi

    echo "=== [1/3] Evolve: ${experiment_path} ==="
    evolve_args=(
        --experiment_path "$experiment_path"
        --config_file "$config_file"
        --data_path "$data_path"
        --dataset "$dataset"
        --network_config "$network_config"
        --log_level INFO
    )
    [[ -n "$fitness_metric" ]] && evolve_args+=(--fitness_metric "$fitness_metric")
    [[ -n "$continue_path" ]] && evolve_args+=(--continue_path "$continue_path")
    [[ -n "$en_pop_crossover_flag" ]] && evolve_args+=("$en_pop_crossover_flag")

    uv run python "${SRC_DIR}/run_evolution.py" "${evolve_args[@]}"
else
    echo "=== [1/3] Evolve: skipped ==="
fi

if should_run "retrain"; then
    echo "=== [2/3] Retrain: ${experiment_path} (config_code=${config_code}) ==="
    uv run python "${SRC_DIR}/retrain_model.py" \
        --experiment_path "$experiment_path" \
        --data_path "$data_path" \
        --dataset "$dataset" \
        --network_config "$network_config" \
        --config_code "$config_code" \
        --log_level INFO \
        --device cuda:0 \
        --num_repetitions "$retrain_num_repetitions" \
        --max_epochs "$retrain_max_epochs" \
        --lr_scheduler "$lr_scheduler" \
        --early_stopping_patience "$early_stopping_patience" \
        $data_augmentation_flag \
        $early_stopping_flag \
        $mixed_precision_flag
else
    echo "=== [2/3] Retrain: skipped ==="
fi

if should_run "infographic"; then
    echo "=== [3/3] Infographic: ${experiment_path} ==="
    uv run python "${SRC_DIR}/generate_infographic.py" --experiment_path "$experiment_path"
else
    echo "=== [3/3] Infographic: skipped ==="
fi

echo "Pipeline finished for ${experiment_path}."
