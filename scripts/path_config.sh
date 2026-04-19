#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${PROJECT_DIR}/src"

# Centralized config directories.
CONFIGS_DIR="${PROJECT_DIR}/configs"
CONFIG_FILES_CIFAR_DIR="${CONFIGS_DIR}/config_files_cifar"
CONFIG_FILES_ATLETA_DIR="${CONFIGS_DIR}/config_files_atleta"
CONFIG_FILES_MED_DIR="${CONFIGS_DIR}/config_files_med"
CONFIG_FILES_MEDMNIST_DIR="${CONFIGS_DIR}/config_files_medmnist"
