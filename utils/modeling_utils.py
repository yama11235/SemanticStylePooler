from .modeling_encoders import BiEncoderForClassification
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForBiEncoder:
    """
    Data collator that pads sentence pairs (and optional triplets) to a common length.

    Args:
        tokenizer: Instance of PreTrainedTokenizerBase for padding.
        padding: Padding strategy (e.g., 'max_length', 'longest', True/False).
        pad_to_multiple_of: If set, pad sequence lengths to a multiple of this value.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Optional[bool | str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    dtype: torch.dtype = torch.float32
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Detect if a third sentence field is present
        has_s3 = any("input_ids_3" in f for f in features)

        # Split features per sentence
        sentence1_feats, sentence2_feats, sentence3_feats = [], [], [] if has_s3 else None
        label_keys = set()
        for f in features:
            s1, s2 = {}, {}
            s3 = {} if has_s3 else None
            for key, value in f.items():
                if key.endswith("_3") and has_s3:
                    s3[key[:-2]] = value
                elif key.endswith("_2"):
                    s2[key[:-2]] = value
                elif key in ("input_ids", "attention_mask", "token_type_ids"):
                    s1[key] = value
                else:
                    label_keys.add(key)
            sentence1_feats.append(s1)
            sentence2_feats.append(s2)
            if has_s3:
                sentence3_feats.append(s3)

        # Compute maximum length across all sentences
        max_len_s1 = max(len(s["input_ids"]) for s in sentence1_feats)
        max_len_s2 = max(len(s["input_ids"]) for s in sentence2_feats)
        if has_s3:
            max_len_s3 = max(len(s["input_ids"]) for s in sentence3_feats)
            combined_len = max(max_len_s1, max_len_s2, max_len_s3)
        else:
            combined_len = max(max_len_s1, max_len_s2)

        # Pad each group to the same combined length
        batch_s1 = self.tokenizer.pad(
            sentence1_feats,
            padding=self.padding,
            max_length=combined_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch_s2 = self.tokenizer.pad(
            sentence2_feats,
            padding=self.padding,
            max_length=combined_len,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt"
        )
        batch = {k: v for k, v in batch_s1.items()}
        for k, v in batch_s2.items():
            batch[f"{k}_2"] = v

        if has_s3:
            batch_s3 = self.tokenizer.pad(
                sentence3_feats,
                padding=self.padding,
                max_length=combined_len,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt"
            )
            for k, v in batch_s3.items():
                batch[f"{k}_3"] = v

        # Collate labels into a single tensor dict
        labels = {}
        for lk in label_keys:
            vals = [float('nan') if f.get(lk) is None else f[lk] for f in features]
            labels[lk] = torch.tensor(vals, dtype=self.dtype)
        batch['labels'] = labels

        return batch

    
def get_model(model_args):
    if model_args.encoding_type == 'bi_encoder':
        return BiEncoderForClassification
    raise ValueError(f'Invalid model type: {model_args.encoding_type}')
