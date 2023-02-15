#!/bin/bash

set -e

root="."
dataset="Maysee/tiny-imagenet"
model='resnet18'

learning_rate=${1:-1e-1}
per_device_train_batch_size=${2:-64}
num_train_epochs=${3:-100}

train_name="${model}-${learning_rate}-bsz${per_device_train_batch_size}-randaug-mixup-moreepoch"
output_dir="${root}/checkpoints/${train_name}"

echo output_dir: $output_dir
mkdir -p $output_dir


CUDA_VISIBLE_DEVICES=1 python run_train.py \
    --dataset_name $dataset \
    --model_type $model \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_train_batch_size \
    --dataloader_num_workers 12 \
    --optim sgd \
    --learning_rate $learning_rate \
    --lr_scheduler_type cosine \
    --weight_decay 1e-4 \
    --logging_strategy steps \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --save_total_limit 1 \
    --rand_aug \
    --mixup
