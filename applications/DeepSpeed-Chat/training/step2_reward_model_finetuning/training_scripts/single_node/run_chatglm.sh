#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# ROOT="/data/renma/unigpt/DeepSpeedExamples/applications/DeepSpeed-Chat"
ROOT="/data/renma/unigpt/"
# modelPath=$ROOT/inference_models/chatglm-6b
# OUTPUT=$ROOT/inference_models/chatglm-6b-pt-reward0509-test
modelPath=$ROOT/inference_models/chatglm-6b
OUTPUT=$ROOT/inference_models/reward0525
dataPath=/data/renma/unigpt/GPT-4-LLM/data/reward_data/unidt_FAQ_reward0525.csv
dataOutputPath=/data/renma/unigpt/GPT-4-LLM/data/reward_data/tokenized_data
mkdir -p $OUTPUT
# export CUDA_VISIBLE_DEVICES=3,4,5,6
   # --load_from_checkpoint True \
nohup deepspeed main.py \
   --data_path  $dataPath \
   --data_split 0,8,2 \
   --model_name_or_path $modelPath \
   --num_padding_at_beginning 0 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 1536 \
   --learning_rate 5e-5 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --disable_dropout \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --output_dir $OUTPUT \
   --data_output_path  $dataOutputPath \
   --model_class chatglm \
   > $OUTPUT/training.log &