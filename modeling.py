import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Type, Dict
from transformers import AutoConfig, AutoModel
from transformers.models.resnet.modeling_resnet import ResNetConfig
from transformers.models.vit.modeling_vit import ViTConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from timm.loss.cross_entropy import SoftTargetCrossEntropy

from arguments import ModelArguments


BACKBONE_CONFIG = {
    "resnet18": {"depths": [2, 2, 2, 2], "hidden_sizes": [64, 128, 256, 512], "layer_type": "basic"},
    "resnet34": {"depths": [3, 4, 6, 3], "hidden_sizes": [64, 128, 256, 512], "layer_type": "basic"},
    "resnet50": {"depths": [3, 4, 6, 3], "hidden_sizes": [256, 512, 1024, 2048], "layer_type": "bottleneck"},
    "vit-small": {"hidden_size": 384, "num_attention_heads": 6, "num_hidden_layers": 12},
    "vit-tiny": {"hidden_size": 192, "num_attention_heads": 3, "num_hidden_layers": 12},
    "vit-atto": {"hidden_size": 32, "num_attention_heads": 2, "num_hidden_layers": 2},
}


class ImageClassificationModelWithMixup(nn.Module):
    def __init__(self, encoder: PreTrainedModel, classifier: nn.Linear, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.num_labels = num_labels

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        num_labels: int,
        **hf_args
    ):
        encoder = setup_backbone_model(model_args, **hf_args)
        if hasattr(encoder.config, "hidden_size"):
            hidden_size = encoder.config.hidden_size
        elif hasattr(encoder.config, "hidden_sizes"):
             hidden_size = encoder.config.hidden_sizes[-1]
        classifier = nn.Linear(hidden_size, num_labels)
        
        if model_args.model_name_or_path:
            with open(os.path.join(model_args.model_name_or_path, "classifier.pt", "rb")) as f:
                classifier.load_state_dict(torch.load(f))
        
        model = cls(
            encoder=encoder,
            classifier=classifier,
            num_labels=num_labels
        )
        
        return model
    
    def forward(
        self, 
        pixel_values: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None
    ) -> ImageClassifierOutputWithNoAttention:

        outputs: BaseModelOutputWithPooling = self.encoder(pixel_values, return_dict=True)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if labels.dtype == torch.long or labels.dtype == torch.int:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = SoftTargetCrossEntropy()
                loss = loss_fct(logits, labels)
        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits)
    
    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, 'classifier.pt'))


def setup_backbone_model(model_args: ModelArguments, **kwargs) -> PreTrainedModel:
    if model_args.model_name_or_path and os.path.isdir(model_args.model_name_or_path):
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **kwargs)
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config)
    elif model_args.model_type:
        if model_args.model_type.startswith("resnet"):
            config = ResNetConfig(
                **BACKBONE_CONFIG[model_args.model_type],
                **kwargs,
            )
        elif model_args.model_type.startswith("vit"):
            config = ViTConfig(
                **BACKBONE_CONFIG[model_args.model_type],
                **kwargs,
            )
        model = AutoModel.from_config(config)
    else:
        raise ValueError
    return model


class ContrastivePreTrainModel(nn.Module):
    def __init__(self, encoder: PreTrainedModel, fc: nn.Module, temperature: float = 1.):
        super().__init__()
        self.encoder = encoder
        self.fc = fc
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
        
    def forward(self, image_q: Dict[str, torch.Tensor], image_k: Dict[str, torch.Tensor]):
        raise NotImplementedError
    
    def encode_image(self, images):
        h = self.encoder(**images).pooler_output.squeeze()
        h = self.fc(h)
        return F.normalize(h, dim=-1)

    @classmethod
    def build(
        cls,
        model_args: ModelArguments,
        **hf_args
    ):
        encoder = setup_backbone_model(model_args, **hf_args)
        fc = nn.Linear(encoder.config.hidden_sizes[-1], model_args.projection_size)

        if model_args.add_projection:
            fc = nn.Sequential(
                nn.Linear(encoder.config.hidden_sizes[-1], encoder.config.hidden_sizes[-1]),
                nn.ReLU(),
                nn.Linear(encoder.config.hidden_sizes[-1], model_args.projection_size)
            )
        
        model = cls(
            encoder=encoder,
            fc=fc,
            temperature=model_args.temperature
        )
        
        return model
    
    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        torch.save(self.fc.state_dict(), os.path.join(output_dir, 'fc.pt'))
    
    
class SimclrPreTrainModel(ContrastivePreTrainModel):
    
    def forward(self, image_q: Dict[str, torch.Tensor], image_k: Dict[str, torch.Tensor]):
        q = self.encode_image(image_q)
        k = self.encode_image(image_k)
        
        bsz = q.size(0)
        labels = torch.cat([torch.arange(bsz, device=q.device) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        reps = torch.cat([q, k], dim=0)
        scores = torch.matmul(reps, reps.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool, device=q.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        scores = scores[~mask].view(scores.shape[0], -1)
        l_pos = scores[labels.bool()].view(labels.shape[0], -1)
        l_neg = scores[~labels.bool()].view(scores.shape[0], -1)
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)
        
        # compute loss
        loss = self.cross_entropy(logits, labels)

        return {"loss": loss}


class MoCoPreTrainModel(ContrastivePreTrainModel):
    
    K = 65536
    m = 0.999
    
    def __init__(self, encoder: PreTrainedModel, fc: nn.Module, temperature: float = 0.2):
        super().__init__(encoder, fc, temperature)
        # create the key encoders
        self.encoder_k = copy.deepcopy(encoder)
        self.fc_k = copy.deepcopy(fc)
        
        for module_q, module_k in zip([self.encoder, self.fc], [self.encoder_k, self.fc_k]):
            for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        # create the queue
        if isinstance(fc, nn.Linear):
            projection_dim = fc.weight.data.size(0)
        else:
            projection_dim = fc[-1].weight.data.size(0)
        self.register_buffer("queue", torch.randn(projection_dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for module_q, module_k in zip([self.encoder, self.fc], [self.encoder_k, self.fc_k]):
            for param_q, param_k in zip(module_q.parameters(), module_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr
    
    def encode_image_k(self, images):
        h = self.encoder_k(**images).pooler_output.squeeze()
        h = self.fc_k(h)
        return F.normalize(h, dim=-1)
    
    def forward(self, image_q: Dict[str, torch.Tensor], image_k: Dict[str, torch.Tensor]):
        # compute query features
        q = self.encode_image(image_q)  # queries: NxC

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encode_image_k(image_k)  # keys: NxC

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q.device)

        self._dequeue_and_enqueue(k)
        
        loss = self.cross_entropy(logits, labels)
        return {"loss": loss}
