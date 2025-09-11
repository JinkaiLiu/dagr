#!/usr/bin/env bash
set -euo pipefail

# GPU selection (optional)
# export CUDA_VISIBLE_DEVICES=0

# 根据需要打开w&b
export WANDB_MODE=disabled

# Paths
PYTHON=python
TEST_SCRIPT=scripts/test_dsec_snn.py 

# Output
OUTPUT_DIR=/media/data/hucao/jinkai/dagr/logs_snn
EXP_NAME=event_only_4_timeslices

# SNN backbone config 
SNN_YAML=src/dagr/cfg/snn_yolov8.yaml
SNN_SCALE=s
SNN_TEMPORAL_BINS=4

# Hyperparameters
BATCH_SIZE=64

# Dataset name
DATASET=DSEC_Det

# Dataset root Directory
DATASET_DIR=/media/data/hucao/zhenwu/hucao/DSEC

# Checkpoint path
CHECKPOINT="${OUTPUT_DIR}/DSEC_Det/detection/event_only_4_timeslices/last_model.pth" 

# Create log file with timestamp
LOG_FILE="${OUTPUT_DIR}/${EXP_NAME}_test_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$OUTPUT_DIR"

echo "Testing log will be saved to: $LOG_FILE"
echo "Starting testing..."

$PYTHON "$TEST_SCRIPT" \
  --config config/dagr-s-dsec.yaml \
  --dataset "$DATASET" \
  --output_directory "$OUTPUT_DIR" \
  --exp_name "${EXP_NAME}_test" \
  --batch_size "$BATCH_SIZE" \
  --use_snn_backbone \
  --snn_yaml_path "$SNN_YAML" \
  --snn_scale "$SNN_SCALE" \
  --snn_temporal_bins "$SNN_TEMPORAL_BINS" \
  --dataset_directory "$DATASET_DIR" \
  --checkpoint "$CHECKPOINT" \
  2>&1 | tee "$LOG_FILE"