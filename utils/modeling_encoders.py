import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F
import logging
import os

from transformers import PreTrainedModel, AutoModel
from .modeling_config import build_classifiers, load_classifiers

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def concat_features(*features):
    return torch.cat(features, dim=0) if features[0] is not None else None


class Pooler(nn.Module):
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last', 'last', 'max'], \
            'unrecognized pooling type %s' % self.pooler_type

    def forward(self, attention_mask, outputs, target_layer=-1):
        last_hidden = outputs.last_hidden_state
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == 'avg':
            return ((hidden_states[target_layer] * attention_mask.unsqueeze(-1)).sum(1) / 
                    attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == 'max':
            mask = attention_mask.unsqueeze(-1).bool()
            masked_hidden = hidden_states[target_layer].masked_fill(
                ~mask, torch.finfo(hidden_states[target_layer].dtype).min
            )
            pooled_result, _ = masked_hidden.max(dim=1)
            return pooled_result
        elif self.pooler_type == 'avg_first_last':
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / \
                            attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'avg_top2':
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / \
                            attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == 'last':
            lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(attention_mask.size(0), device=attention_mask.device)
            return hidden_states[target_layer][batch_idx, lengths]
        else:
            raise NotImplementedError


class BiEncoderForClassification(PreTrainedModel):
    def __init__(self, model_config, classifier_configs):
        super().__init__(model_config)
        self.config = model_config
        self.model_path = (
            getattr(model_config, "model_name_or_path", None) or 
            getattr(model_config, "name_or_path", None)
        )
        self.classifier_configs = classifier_configs
        
        self._init_backbone(model_config)
        self.pooler = Pooler(model_config.pooler_type)
        
        if model_config.pooler_type in {'avg_first_last', 'avg_top2'} or classifier_configs:
            self.output_hidden_states = True
        else:
            self.output_hidden_states = False

        self.embedding_classifiers = nn.ModuleDict()
        self.clf_configs = {}

        if self.classifier_configs:
            self.embedding_classifiers, self.clf_configs = build_classifiers(
                self.classifier_configs, model_config
            )
            self.classifier_save_directory = getattr(model_config, 'classifier_save_directory', None)
            
        self.post_init()
        self._move_classifiers_to_device()

    def _init_backbone(self, model_config):
        common_args = {
            'from_tf': bool(self.model_path and '.ckpt' in self.model_path),
            'config': model_config,
            'cache_dir': getattr(model_config, "cache_dir", None),
            'revision': getattr(model_config, "model_revision", None),
            'use_auth_token': True if getattr(model_config, "use_auth_token", None) else None,
            'attn_implementation': getattr(model_config, "attn_implementation", "eager"),
            'torch_dtype': getattr(model_config, "torch_dtype", torch.float16),
        }
        
        if getattr(model_config, "device_map", None):
            common_args['device_map'] = model_config.device_map
            
        self.backbone = AutoModel.from_pretrained(self.model_path, **common_args).base_model

    def _move_classifiers_to_device(self):
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        print(f"dtype: {dtype}, device: {device}")
        for classifier in self.embedding_classifiers.values():
            classifier.to(device=device, dtype=dtype)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        input_ids_2=None,
        attention_mask_2=None,
        token_type_ids_2=None,
        position_ids_2=None,
        head_mask_2=None,
        inputs_embeds_2=None,
        input_ids_3=None,
        attention_mask_3=None,
        token_type_ids_3=None,
        position_ids_3=None,
        head_mask_3=None,
        inputs_embeds_3=None,            
        labels=None, 
        **kwargs,
    ):
        if input_ids_3 is None:
            return self._forward_pairwise(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
                input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, head_mask_2, inputs_embeds_2
            )
        else:
            return self._forward_triplet(
                input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
                input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, head_mask_2, inputs_embeds_2,
                input_ids_3, attention_mask_3, token_type_ids_3, position_ids_3, head_mask_3, inputs_embeds_3,
                labels, **kwargs
            )

    def _forward_pairwise(
        self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
        input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, head_mask_2, inputs_embeds_2
    ):
        if input_ids is None:
            raise ValueError("input_ids is None")
        
        bsz = input_ids.shape[0]
        input_ids = concat_features(input_ids, input_ids_2)
        attention_mask = concat_features(attention_mask, attention_mask_2)
        token_type_ids = concat_features(token_type_ids, token_type_ids_2)
        position_ids = concat_features(position_ids, position_ids_2)
        head_mask = concat_features(head_mask, head_mask_2)
        inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2)
        
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
        )
        
        outputs_dict = {}
        if self.classifier_configs:
            for name, classifier in self.embedding_classifiers.items():
                target_layer = int(self.classifier_configs[name]["layer"])
                pooled_features = self.pooler(attention_mask, outputs, target_layer)
                features1, features2 = torch.split(pooled_features, bsz, dim=0)
                
                output1 = classifier(features1)
                output2 = classifier(features2)
                
                similarity = self._compute_similarity(
                    output1, output2, 
                    self.classifier_configs[name]["distance"],
                    self.classifier_configs[name]["objective"]
                )
                outputs_dict[name] = similarity

        features = self.pooler(attention_mask, outputs)
        features_1, features_2 = torch.split(features, bsz, dim=0)
        logits = cosine_similarity(features_1, features_2, dim=1).to(features_1.dtype)
        outputs_dict["overall_similarity"] = logits
        
        return outputs_dict

    def _forward_triplet(
        self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds,
        input_ids_2, attention_mask_2, token_type_ids_2, position_ids_2, head_mask_2, inputs_embeds_2,
        input_ids_3, attention_mask_3, token_type_ids_3, position_ids_3, head_mask_3, inputs_embeds_3,
        labels, **kwargs
    ):
        if input_ids is None or input_ids_2 is None or input_ids_3 is None:
            raise ValueError("input_ids, input_ids_2, input_ids_3 must not be None")
        
        bsz1, bsz2, bsz3 = input_ids.shape[0], input_ids_2.shape[0], input_ids_3.shape[0]
        
        input_ids = concat_features(input_ids, input_ids_2, input_ids_3)
        attention_mask = concat_features(attention_mask, attention_mask_2, attention_mask_3)
        token_type_ids = concat_features(token_type_ids, token_type_ids_2, token_type_ids_3)
        position_ids = concat_features(position_ids, position_ids_2, position_ids_3)
        head_mask = concat_features(head_mask, head_mask_2, head_mask_3)
        inputs_embeds = concat_features(inputs_embeds, inputs_embeds_2, inputs_embeds_3)
        
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
        )
        
        outputs_dict = {}
        if self.classifier_configs:
            for name, classifier in self.embedding_classifiers.items():
                if self.classifier_configs[name]["type"] != "contrastive_logit":
                    continue
                
                target_layer = int(self.classifier_configs[name]["layer"])
                pooled_features = self.pooler(attention_mask, outputs, target_layer)
                features1, features2, features3 = torch.split(pooled_features, [bsz1, bsz2, bsz3], dim=0)
                
                output1, prob1 = classifier(features1)
                output2, prob2 = classifier(features2)
                output3, prob3 = classifier(features3)

                distance_metric = self.classifier_configs[name]["distance"]
                pos_similarity = self._compute_distance(output1, output2, distance_metric)
                neg_similarity = self._compute_distance(output1, output3, distance_metric)

                outputs_dict[f"{name}_pos_similarity"] = pos_similarity
                outputs_dict[f"{name}_neg_similarity"] = neg_similarity
                outputs_dict[f"{name}_anchor_prob"] = prob1
                outputs_dict[f"{name}_positive_prob"] = prob2
                outputs_dict[f"{name}_negative_prob"] = prob3
                
        features = self.pooler(attention_mask, outputs)
        features_1, features_2, features_3 = torch.split(features, [bsz1, bsz2, bsz3], dim=0)
        logits1 = cosine_similarity(features_1, features_2, dim=1).to(features_1.dtype)
        logits2 = cosine_similarity(features_1, features_3, dim=1).to(features_1.dtype)
        outputs_dict["overall_pos_similarity"] = logits1
        outputs_dict["overall_neg_similarity"] = logits2
        
        return outputs_dict

    def _compute_similarity(self, output1, output2, distance, objective):
        similarity = self._compute_distance(output1, output2, distance)
        if objective == "binary_classification":
            similarity = torch.sigmoid(similarity).to(output1.dtype)
        return similarity

    def _compute_distance(self, output1, output2, distance):
        if distance == "cosine":
            return cosine_similarity(output1, output2, dim=1).to(output1.dtype)
        elif distance == "euclidean":
            return torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        elif distance == "dot_product":
            return torch.sum(output1 * output2, dim=1)
        raise ValueError(f"Unknown distance metric: {distance}")

    def encode(
        self, input_ids=None, attention_mask=None, token_type_ids=None,
        position_ids=None, head_mask=None, inputs_embeds=None,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.output_hidden_states,
        )
        
        outputs_dict = {}
        if self.classifier_configs:
            for name, classifier in self.embedding_classifiers.items():
                target_layer = int(self.classifier_configs[name]["layer"])
                pooled_features = self.pooler(attention_mask, outputs, target_layer)
                embedding = classifier.encode(pooled_features)
                outputs_dict[name] = embedding

        features = self.pooler(attention_mask, outputs)
        outputs_dict["original"] = features
        return outputs_dict

    def save_pretrained(self, model_save_directory, **kwargs):
        os.makedirs(model_save_directory, exist_ok=True)
        classifier_save_directory = self.classifier_save_directory or model_save_directory
        os.makedirs(classifier_save_directory, exist_ok=True)
        
        if not self.config.freeze_encoder:
            super().save_pretrained(model_save_directory, **kwargs)
        self.config.save_pretrained(model_save_directory)
        
        for name, module in self.embedding_classifiers.items():
            param_str = f"{self.classifier_configs[name]['type']}_layer:{self.classifier_configs[name]['layer']}_dim:{self.classifier_configs[name]['output_dim']}"
            save_path = os.path.join(classifier_save_directory, param_str, f"{name}_classifier.bin")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(module.state_dict(), save_path)
            self.clf_configs[name].save_pretrained(classifier_save_directory, name)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, model_config, 
                       classifier_save_directory=None, classifier_configs=None, **kwargs):
        model = cls(model_config, classifier_configs)
        
        for name, param in model.backbone.named_parameters():
            if torch.isnan(param).any():
                print(f"{name} contains NaN values")
        
        if classifier_save_directory and classifier_configs:
            loaded_heads = load_classifiers(classifier_configs, model_config, classifier_save_directory)
            model.embedding_classifiers = loaded_heads

        return model
