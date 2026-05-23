#!/bin/bash
# =============================================================
# Experiment: No IQR clipping + lambd=0
# Purpose: Test whether IQR preprocessing & entropy regularization
#          are the main bottleneck for AUC-PR on CUSTOM dataset.
# =============================================================
# Usage:
#   chmod +x run_noiqr_experiment.sh
#   ./run_noiqr_experiment.sh
# =============================================================

# ---- Experiment config ----
DATASET="CUSTOM"
DATA_PATH="./data/"
WIN_SIZE=100
INPUT_C=33
OUTPUT_C=33
BATCH_SIZE=8
NUM_EPOCHS=100
N_MEMORY=128
D_MODEL=512
LR=0.0001
TEMP_PARAM=0.05
LAMBDA=0            # <--- entropy loss disabled
TEMPERATURE=0.1
ANOMALY_RATIO=1.0
NUM_WORKERS=4
DEVICE="cuda:0"
MODEL_SAVE_PATH="checkpoints/exp_noiqr_lambd0"   # isolated checkpoints
USE_IQR=0           # <--- IQR clipping disabled
# --------------------------------------------

mkdir -p ${MODEL_SAVE_PATH}

echo "=============================================="
echo " Experiment: No IQR + lambd=0"
echo "=============================================="
echo " Hyperparameters:"
echo "   use_iqr=${USE_IQR}   lambd=${LAMBDA}"
echo "   win_size=${WIN_SIZE}  d_model=${D_MODEL}"
echo "   n_memory=${N_MEMORY}  lr=${LR}"
echo "   batch_size=${BATCH_SIZE}  epochs=${NUM_EPOCHS}"
echo "=============================================="
echo ""

# ---- Phase 1: First Training (random memory init) ----
echo ">> [1/4] Phase 1: First training (random memory initialization)"
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
    --phase_type first_train \
    --use_iqr ${USE_IQR}

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 1 failed!"
    exit 1
fi
echo ">> [1/4] Phase 1 completed."
echo ""

# ---- Phase 2: Memory Initialization (K-means) ----
echo ">> [2/4] Phase 2: Memory initialization (K-means clustering)"
python main.py \
    --mode memory_initial \
    --dataset ${DATASET} \
    --data_path ${DATA_PATH} \
    --win_size ${WIN_SIZE} \
    --input_c ${INPUT_C} \
    --output_c ${OUTPUT_C} \
    --batch_size ${BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --n_memory ${N_MEMORY} \
    --d_model ${D_MODEL} \
    --device ${DEVICE} \
    --model_save_path ${MODEL_SAVE_PATH} \
    --phase_type second_train \
    --use_iqr ${USE_IQR}

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 2 failed!"
    exit 1
fi
echo ">> [2/4] Phase 2 completed."
echo ""

# ---- Phase 3: Second Training (K-means initialized memory) ----
echo ">> [3/4] Phase 3: Second training (K-means initialized memory)"
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
    --phase_type second_train \
    --use_iqr ${USE_IQR}

if [ $? -ne 0 ]; then
    echo "ERROR: Phase 3 failed!"
    exit 1
fi
echo ">> [3/4] Phase 3 completed."
echo ""

# ---- Phase 4: Test ----
echo ">> [4/4] Phase 4: Testing"
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
    --phase_type test \
    --use_iqr ${USE_IQR}

if [ $? -ne 0 ]; then
    echo "ERROR: Test failed!"
    exit 1
fi

echo ""
echo "=============================================="
echo " Experiment complete!"
echo " Config: no IQR + lambd=0"
echo " Checkpoints: ${MODEL_SAVE_PATH}/"
echo "=============================================="
echo ""
echo "Compare the AUC-PR above with the baseline"
echo "(IQR enabled + lambd=0.01 from train.sh)."
echo "If AUC-PR rose significantly -> preprocessing"
echo "and training objective were the bottleneck."
echo "=============================================="
