import os
import sys

from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    set_seed,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from arguments import DataArguments, ModelArguments
from data import SimclrPreTrainDataset, MoCoPreTrainDataset, MoCo2PreTrainDataset, ContrastivePreTrainCollator
from trainer import ContrastivePreTrainer as Trainer
from modeling import SimclrPreTrainModel, MoCoPreTrainModel, ContrastivePreTrainModel


CONTRASTIVE_FRAMEWORK = {
    "simclr": [SimclrPreTrainDataset, SimclrPreTrainModel],
    "moco": [MoCoPreTrainDataset, MoCoPreTrainModel],
    "moco-v2": [MoCo2PreTrainDataset, MoCoPreTrainModel]
}


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()
    
    model_args: ModelArguments
    data_args: DataArguments
    train_args: TrainingArguments
    
     # Set seed before initializing model.
    set_seed(train_args.seed)

    # Initialize our dataset and prepare it for the 'image-classification' task.
    dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        
    if train_args.do_train:
        if data_args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=train_args.seed).select(range(data_args.max_train_samples))
        train_dataset = CONTRASTIVE_FRAMEWORK[model_args.contrastive_framework][0](data_args, dataset["train"])
    
    # Set up model
    model: ContrastivePreTrainModel = CONTRASTIVE_FRAMEWORK[model_args.contrastive_framework][1].build(model_args)
    
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
        data_collator=ContrastivePreTrainCollator(),
    )

    # Training
    if train_args.do_train:
        checkpoint = None
        if train_args.resume_from_checkpoint is not None:
            checkpoint = train_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.save_state()


if __name__ == "__main__":
    main()
