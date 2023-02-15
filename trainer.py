import os

from typing import Optional, Dict, List
from torch.utils.data import DataLoader
from transformers.trainer import Trainer

from data import DefaultCollator

import logging
logger = logging.getLogger(__name__)


class SimpleTrainer(Trainer):
    
    def evaluate(self, eval_dataset = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        changed = False
        if hasattr(self.model, "config") and self.model.config.problem_type == "multi_label_classification":
            changed = True
            self.model.config.problem_type = "single_label_classification"
        result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        if hasattr(self.model, "config") and changed:
            self.model.config.problem_type = "multi_label_classification"
        return result
    
    def get_eval_dataloader(self, eval_dataset):
        eval_dataset = self.eval_dataset
        data_collator = DefaultCollator()
        data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class ContrastivePreTrainer(Trainer):
        
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        self.model.save(output_dir)

