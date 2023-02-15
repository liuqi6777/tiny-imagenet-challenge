from dataclasses import dataclass, field
from typing import Optional

from transformers.training_args import TrainingArguments


@dataclass
class DataArguments:

    dataset_name: Optional[str] = field(default='Maysee/tiny-imagenet')
    dataset_config_name: Optional[str] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    
    input_size: int = 224
    
    rand_aug: bool = False
    mixup: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)

    contrastive_framework: Optional[str] = field(default=None)
    add_projection: Optional[bool] = field(default=False)
    projection_size: int = field(default=128)
    temperature: float = field(default=1.)
