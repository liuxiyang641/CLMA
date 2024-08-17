#!/usr/bin/env bash

source ~/.bashrc
conda activate CLMA
wandb disabled

DATASET_NAME="tacrev"
#DATASET_NAME="tacred"
# DATASET_NAME="retacred"
# DATASET_NAME="semeval"

EXP_NAME="test"

EXP_METHOD="unlabeled_rule_iter"

LRE_SETTING="k-shot"
K_SHOT_LRE="8"
K_SHOT_ID="1"

CUDA_VISIBLE_DEVICES=0 python main.py \
--exp_name ${EXP_NAME} \
--api_key Your_API_Key \
--method ${EXP_METHOD} \
--dataset ${DATASET_NAME} \
--lre_setting ${LRE_SETTING} \
--k_shot_lre ${K_SHOT_LRE} \
--k_shot_id ${K_SHOT_ID} \
--percent_lre ${PERCENT_LRE} \
--percent_id ${PERCENT_ID} \
--mode run \
--split train \
--in_context_size 5