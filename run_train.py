import os
import sys
import numpy as np
import evaluate

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    set_seed,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from arguments import DataArguments, ModelArguments
from data import DefaultCollator, TrainDataset, EvalDataset, MixupCollator
from trainer import SimpleTrainer as Trainer
from modeling import setup_backbone_model, ImageClassificationModelWithMixup


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    model_args: ModelArguments
    data_args: DataArguments
    train_args: TrainingArguments
    
     # Set seed before initializing model.
    set_seed(train_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        task="image-classification",
    )

    labels = dataset["train"].features["labels"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
        
    if train_args.do_train:
        if data_args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=train_args.seed).select(range(data_args.max_train_samples))
        train_dataset = TrainDataset(data_args, dataset["train"])

    if train_args.do_eval:
        if data_args.max_eval_samples is not None:
            dataset["valid"] = dataset["valid"].shuffle(seed=train_args.seed).select(range(data_args.max_eval_samples))
        eval_dataset = EvalDataset(data_args, dataset["valid"])
        
    collator = DefaultCollator()
    if data_args.mixup:
        collator = MixupCollator()

    # Define our compute_metrics function.
    metric = evaluate.load("accuracy")
    
    def compute_metrics(p):
        """Computes accuracy on a batch of predictions"""
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
    # Set up model
    model = ImageClassificationModelWithMixup.build(
        model_args,
        num_labels=len(id2label),
        finetuning_task="ImageClassification",
        id2label=id2label,
        label2id=label2id,
    )
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and train_args.resume_from_checkpoint is None:
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initalize our trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=eval_dataset if train_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=collator,
    )

    # Training
    if train_args.do_train:
        checkpoint = None
        if train_args.resume_from_checkpoint is not None:
            checkpoint = train_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if train_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
