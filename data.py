import torch

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing
from timm.data.auto_augment import rand_augment_transform, AugMixAugment

from arguments import DataArguments


class TrainDataset(Dataset):
    """Basic dataset for resnet training or finetuning."""
    
    def __init__(self, data_args: DataArguments, dataset: HFDataset):
        super().__init__()
        self.dataset = dataset
        self.data_args = data_args
        self.init_transform()
        self.dataset.set_transform(self._data_transform)
        
    def init_transform(self, transform=None):
        if transform:
            self.transform = transform
            return
        if not self.data_args.rand_aug:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.data_args.input_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.data_args.input_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                rand_augment_transform('rand-m7-mstd0.5', hparams={'translate_const': 117, 'img_mean': (124, 116, 104)}),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.dataset)

    def _data_transform(self, example_batch):
        example_batch["pixel_values"] = [
            self.transform(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        return example_batch

    def __getitem__(self, item):
        return self.dataset[item]


class EvalDataset(TrainDataset):
    
    def init_transform(self):
        self.transform = transforms.Compose([
            transforms.Resize(self.data_args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class DefaultCollator:

    def __call__(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
        
        
class MixupCollator:
    
    def __init__(self):
        self.mixup = Mixup(mixup_alpha=0.2, cutmix_alpha=1., num_classes=200)
    
    def __call__(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        pixel_values, labels = self.mixup(pixel_values, labels)
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
        

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class SimclrPreTrainDataset(TrainDataset):
    
    def init_transform(self, transform=None):
        # the same as SimCLR https://arxiv.org/abs/2002.05709
        self.transform = TwoCropsTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=self.data_args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=7)], p=0.5),
            transforms.ToTensor()
        ]))
        
    def _data_transform(self, example_batch):
        transformed_image = [
            self.transform(pil_img.convert("RGB")) for pil_img in example_batch["image"]
        ]
        example_batch["image_q"] = [pair[0] for pair in transformed_image]
        example_batch["image_k"] = [pair[1] for pair in transformed_image]
        return example_batch


class MoCoPreTrainDataset(SimclrPreTrainDataset):
    
    def init_transform(self, transform=None):
        # the same as InstDisc https://arxiv.org/abs/1805.01978
        self.transform = TwoCropsTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=self.data_args.input_size, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        

class MoCo2PreTrainDataset(SimclrPreTrainDataset):
    
    def init_transform(self, transform=None):
        # similar to SimCLR https://arxiv.org/abs/2002.05709
        self.transform = TwoCropsTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=self.data_args.input_size, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=7)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))


class ContrastivePreTrainCollator:
    def __call__(self, examples):
        image_q = torch.stack([example["image_q"] for example in examples])
        image_k = torch.stack([example["image_k"] for example in examples])
        return {
            "image_q": {"pixel_values": image_q},
            "image_k": {"pixel_values": image_k}
        }
