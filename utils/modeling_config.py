import json
import os
import torch.nn as nn
import torch
from .modeling_classifier import STSHead, PIHead, TripletHead
from .constants import TaskType
from typing import Dict, List, Optional

class BaseConfig:
    def __init__(self, input_dim, output_dim, dropout, layer, meta=None, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.layer = layer
        self.meta = meta or {}
        
    def to_dict(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'layer': self.layer,
            'meta': self.meta,
        }
        
    def save_pretrained(self, save_path: str, classifier_name: str):
        config_dict = self.to_dict()
        config_dict['classifier_name'] = classifier_name
        if isinstance(self, ContrastiveClassifierConfig):
             dirname = f"contrastive_logit_layer:{self.layer}_dim:{self.output_dim}"
        elif isinstance(self, MLP2LayerConfig):
             dirname = f"mlp2_layer:{self.layer}_dim:{self.output_dim}"
        else:
             dirname = f"linear_layer:{self.layer}_dim:{self.output_dim}"
             
        save_path = os.path.join(save_path, dirname, f"{classifier_name}.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=4)

class LinearLayerConfig(BaseConfig):
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            input_dim=config_dict.get('input_dim'), 
            output_dim=config_dict['output_dim'],
            dropout=config_dict['dropout'],
            layer=config_dict.get('layer'),
            meta=config_dict.get('meta', {}),
        )

class MLP2LayerConfig(BaseConfig):
    def __init__(self, input_dim, intermediate_dim, bottleneck_dim, output_dim, dropout, layer, meta=None):
        super().__init__(input_dim, output_dim, dropout, layer, meta)
        self.intermediate_dim = intermediate_dim
        self.bottleneck_dim = bottleneck_dim

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'intermediate_dim': self.intermediate_dim,
            'bottleneck_dim': self.bottleneck_dim
        })
        return d

    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            input_dim=config_dict.get('input_dim'),
            intermediate_dim=config_dict['intermediate_dim'],
            bottleneck_dim=config_dict['bottleneck_dim'],
            output_dim=config_dict['output_dim'],
            dropout=config_dict['dropout'],
            layer=config_dict.get('layer'),
            meta=config_dict.get('meta', {}),
        )

class ContrastiveClassifierConfig(BaseConfig):
    def __init__(self, input_dim, intermediate_dim, output_dim, dropout, layer, meta=None):
        super().__init__(input_dim, output_dim, dropout, layer, meta)
        self.intermediate_dim = intermediate_dim

    def to_dict(self):
        d = super().to_dict()
        d['intermediate_dim'] = self.intermediate_dim
        return d
    
    @classmethod
    def from_dict(cls, config_dict):
        return cls(
            input_dim=config_dict.get('input_dim'),
            intermediate_dim=config_dict['intermediate_dim'],
            output_dim=config_dict['output_dim'],
            dropout=config_dict['dropout'],
            layer=config_dict.get('layer'),
            meta=config_dict.get('meta', {}),
        )

def build_classifiers(classifier_configs: dict, model_config) -> (nn.ModuleDict, dict):
    modules = nn.ModuleDict()
    clf_configs = {}

    for name, params in classifier_configs.items():
        params = params.copy()
        params.setdefault("name", name)
        ctype = params.get("type") 
        objective = params.get("objective", "regression")
        
        task_type = None
        try:
            task_type = TaskType.from_str(objective)
        except ValueError:
            pass 

        if ctype == 'linear':
            cfg = LinearLayerConfig(
                input_dim=model_config.hidden_size,
                output_dim=params["output_dim"],
                dropout=params.get("dropout", 0.1),
                layer=params.get("layer", model_config.num_hidden_layers - 1),
                meta={
                    "name": name,
                    "type": "linear",
                    "objective": objective,
                    "distance": params.get("distance", "cosine"),
                },
            )
            
            if task_type == TaskType.STS:
                module = STSHead(cfg)
            elif task_type == TaskType.PI:
                module = PIHead(cfg)
            else:
                module = STSHead(cfg)

        elif ctype == 'contrastive_logit':
            cfg = ContrastiveClassifierConfig(
                input_dim=model_config.hidden_size,
                intermediate_dim=params["intermediate_dim"],
                output_dim=params["output_dim"],
                dropout=params.get("dropout", 0.1),
                layer=params.get("layer", model_config.num_hidden_layers - 1),
                meta={
                    "name": name,
                    "type": "contrastive_logit",
                    "objective": objective,
                    "distance": params.get("distance", "cosine"),
                },
            )
            module = TripletHead(cfg)

        else:
            raise ValueError(f"Unknown classifier type: {ctype}")

        modules[name] = module
        clf_configs[name] = cfg

    return modules, clf_configs

def load_classifiers(classifier_configs: Dict, model_config: Dict, save_dir: List) -> nn.ModuleDict:
    modules, _ = build_classifiers(classifier_configs, model_config)

    for name, classifier_path in zip(list(modules.keys()), save_dir):
        weight_path = classifier_path
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Classifier weight file for {name} not found at {weight_path}")
        state = {} if not os.path.getsize(weight_path) else torch.load(weight_path, map_location="cpu")
        modules[name].load_state_dict(state)

    return modules
