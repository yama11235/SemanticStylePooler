import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from .constants import TaskType
from .loss import LossFactory, TripletLoss

class CustomTrainer(Trainer):
    def __init__(self, *args, classifier_configs=None, dtype=torch.float16, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_configs = classifier_configs
        self.loss_fns = {}
        self.dtype = dtype
        
        if classifier_configs:
            for name, config in classifier_configs.items():
                objective = config.get("objective", "regression")
                try:
                    task_type = TaskType.from_str(objective)
                    self.loss_fns[name] = LossFactory.get_loss_fn(task_type, config)
                except ValueError:
                    print(f"Warning: Unknown objective {objective} for classifier {name}")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        labels = inputs.get("labels", {})
        
        if not isinstance(labels, dict):
            pass

        if outputs and isinstance(outputs, dict):
             for v in outputs.values():
                 if isinstance(v, torch.Tensor):
                     device = v.device
                     break
             else:
                 device = inputs['input_ids'].device
        else:
             device = inputs['input_ids'].device

        total_loss = torch.tensor(0.0, device=device, requires_grad=True, dtype=torch.float32)

        for name, loss_fn in self.loss_fns.items():
            if name in labels:
                target = labels[name]
                
                if isinstance(loss_fn, TripletLoss):
                     loss = loss_fn(outputs, target, name)
                     total_loss = total_loss + loss
                
                elif name in outputs:
                    output = outputs[name]
                    mask = ~torch.isnan(target)
                    if mask.sum() > 0:
                        loss = loss_fn(output[mask], target[mask])
                        total_loss = total_loss + loss.to(torch.float32)

        return (total_loss, outputs) if return_outputs else total_loss

    def evaluation_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            labels = inputs.get("labels", {})

        if prediction_loss_only:
            return (loss, None, None)
            
        return (loss, outputs, labels)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        return self.evaluation_step(model, inputs, prediction_loss_only, ignore_keys)
