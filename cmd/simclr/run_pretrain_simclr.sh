#!/bin/bash

set -e

root="."
dataset="Maysee/tiny-imagenet"
model='resnet18'

learning_rate=${1:-1.2}
per_device_train_batch_size=${2:-256}
num_train_epochs=${3:-200}

train_name="simclr-${model}-pretrain-lr${learning_rate}-bs${per_device_train_batch_size}-ep${num_train_epochs}"
output_dir="${root}/checkpoints/${train_name}"

echo output_dir: $output_dir
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=1 python run_contrastive.py \
    --dataset_name $dataset \
    --model_type $model \
    --contrastive_framework simclr \
    --add_projection \
    --temperature 0.5 \
    --output_dir $output_dir \
    --remove_unused_columns False \
    --do_train \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_train_batch_size \
    --dataloader_num_workers 12 \
    --optim sgd \
    --learning_rate $learning_rate \
    --weight_decay 1e-6 \
    --lr_scheduler_type cosine \
    --logging_strategy steps \
    --logging_steps 1000 \
    --save_strategy steps \
    --save_steps 20000 \
    --save_total_limit 2
