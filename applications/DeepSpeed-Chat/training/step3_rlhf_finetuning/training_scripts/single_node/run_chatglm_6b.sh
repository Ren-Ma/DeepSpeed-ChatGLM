#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# DeepSpeed Team
ROOT="/data/renma/unigpt/"
# ACTOR_MODEL_PATH=/data/caihua/huggingfaceModels/sft_model_lora_2048_aug0602
ACTOR_MODEL_PATH=$ROOT/inference_models/rlhf0601_unichat_round1/actor
# CRITIC_MODEL_PATH=$ROOT/inference_models/reward0525
CRITIC_MODEL_PATH=$ROOT/inference_models/rlhf0601_unichat_round1/critic
OUTPUT=$ROOT/inference_models/rlhf0602_unichat_round1
dataPath=$ROOT/unichat_num212.xlsx
dataOutputPath=$ROOT/GPT-4-LLM/data/ppo_data/tokenized_data

mkdir -p $OUTPUT
Actor_Lr=9.65e-6
Critic_Lr=5e-6

# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export NCCL_DEBUG=INFO
#    --offload_reference_model \

nohup deepspeed --master_port $MASTER_PORT main.py \
   --data_path  $dataPath\
   --data_output_path $dataOutputPath\
   --data_split 0,0,10 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 1 \
   --per_device_mini_train_batch_size 1 \
   --generation_batch_numbers 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 512 \
   --actor_learning_rate ${Actor_Lr} \
   --critic_learning_rate ${Critic_Lr} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 4 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --disable_actor_dropout \
   --num_warmup_steps 100 \
   --deepspeed --seed 1234 \
   --enable_hybrid_engine \
   --actor_zero_stage 2 \
   --critic_zero_stage 2 \
   --output_dir $OUTPUT \
   --model_class chatglm \
    > $OUTPUT/training.log &
