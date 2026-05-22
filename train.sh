#!/bin/bash
# =============================================================
# MEMTO Training Script for CUSTOM Dataset
# =============================================================
# Usage:
#   chmod +x train.sh
#   ./train.sh
#
# Prerequisites:
#   - Place your data in ./data/test_complex/
#     (train.csv with 'y' column for labels)
#   - Adjust hyperparameters below if needed
# =============================================================

# ---- Hyperparameters (modify as needed) ----
DATASET="CUSTOM"
DATA_PATH="./data/"
WIN_SIZE=100
INPUT_C=33          # number of features/channels (f0-f33, excluding y)
OUTPUT_C=33         # should match INPUT_C
BATCH_SIZE=8
NUM_EPOCHS=10
N_MEMORY=128
D_MODEL=512
LR=0.0001
TEMP_PARAM=0.05
LAMBDA=0.01
TEMPERATURE=0.1
ANOMALY_RATIO=1.0   # threshold percentile
NUM_WORKERS=4
DEVICE="cuda:0"
MODEL_SAVE_PATH="checkpoints"
# --------------------------------------------

echo "=============================================="
echo " MEMTO Training Pipeline - ${DATASET}"
echo "=============================================="
echo ""

# ---- Phase 1: First Training (random memory init) ----
echo ">> Phase 1: First training (random memory initialization)"
python main.py \
    --mode train \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --win_size ${WIN_SIZE} \
    --input_c ${INPUT_C} \
    --output_c ${OUTPUT_C} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --n_memory ${N_MEMORY} \
    --d_model ${D_MODEL} \
    --lr ${LR} \
    --temp_param ${TEMP_PARAM} \
    --lambd ${LAMBDA} \
    --temperature ${TEMPERATURE} \
    --anomaly_ratio ${ANOMALY_RATIO} \
    --num_workers ${NUM_WORKERS} \
    --device ${DEVICE} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --phase_type first_train

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 1 training failed!"
    exit 1
fi
echo ">> Phase 1 completed."
echo ""

# ---- Phase 2: Memory Initialization (K-means) ----
echo ">> Phase 2: Memory initialization (K-means clustering)"
python main.py \
    --mode memory_initial \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --win_size ${WIN_SIZE} \
    --input_c ${INPUT_C} \
    --output_c ${OUTPUT_C} \
    --batch_size ${BATCH_SIZE} \
    --n_memory ${N_MEMORY} \
    --d_model ${D_MODEL} \
    --device ${DEVICE} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --phase_type second_train

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 2 memory initialization failed!"
    exit 1
fi
echo ">> Phase 2 completed."
echo ""

# ---- Phase 3: Second Training (K-means initialized memory) ----
echo ">> Phase 3: Second training (K-means initialized memory)"
python main.py \
    --mode train \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --win_size ${WIN_SIZE} \
    --input_c ${INPUT_C} \
    --output_c ${OUTPUT_C} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --n_memory ${N_MEMORY} \
    --d_model ${D_MODEL} \
    --lr ${LR} \
    --temp_param ${TEMP_PARAM} \
    --lambd ${LAMBDA} \
    --temperature ${TEMPERATURE} \
    --anomaly_ratio ${ANOMALY_RATIO} \
    --num_workers ${NUM_WORKERS} \
    --device ${DEVICE} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --phase_type second_train

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 3 training failed!"
    exit 1
fi
echo ">> Phase 3 completed."
echo ""

# ---- Phase 4: Test ----
echo ">> Phase 4: Testing"
python main.py \
    --mode test \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --win_size ${WIN_SIZE} \
    --input_c ${INPUT_C} \
    --output_c ${OUTPUT_C} \
    --batch_size ${BATCH_SIZE} \
    --n_memory ${N_MEMORY} \
    --d_model ${D_MODEL} \
    --anomaly_ratio ${ANOMALY_RATIO} \
    --device ${DEVICE} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --phase_type test

if [ $? -ne 0 ]; then
    echo "ERROR: Testing failed!"
    exit 1
fi
echo ">> Testing completed."
echo ""

echo "=============================================="
echo " All phases completed successfully!"
echo "=============================================="
