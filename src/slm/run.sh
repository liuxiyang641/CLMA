#!/usr/bin/env bash

source ~/.bashrc
conda activate CLMA
wandb offline

DATASET_NAME="tacrev"
#DATASET_NAME="tacred"
# DATASET_NAME="retacred"
# DATASET_NAME="semeval"

EXP_NAME="test_slm"

LRE_SETTING="k-shot"
K_SHOT_LRE="8"
K_SHOT_ID="1"

CUDA_VISIBLE_DEVICES=0 python slm_finetune.py \
--exp_name ${EXP_NAME} \
--dataset ${DATASET_NAME} \
--lre_setting ${LRE_SETTING} \
--k_shot_lre ${K_SHOT_LRE} \
--k_shot_id ${K_SHOT_ID} \
--percent_lre ${PERCENT_LRE} \
--percent_id ${PERCENT_ID} \
--mode run