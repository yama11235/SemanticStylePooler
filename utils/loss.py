import torch
import torch.nn.functional as F
from .constants import TaskType

class LossFactory:
    @staticmethod
    def get_loss_fn(task_type: TaskType, config=None):
        if task_type == TaskType.STS:
            return STSLoss()
        elif task_type == TaskType.PI:
            return PILoss()
        elif task_type == TaskType.TRIPLET:
            return TripletLoss(margin=config.get("margin", 0.2), alpha=config.get("alpha", 1.0))
        raise ValueError(f"Unknown task type: {task_type}")

class STSLoss:
    def __call__(self, output, target, **kwargs):
        return F.mse_loss(output.to(torch.float32), target.to(torch.float32))

class PILoss:
    def __call__(self, output, target, **kwargs):
        return F.binary_cross_entropy(output.to(torch.float32), target.to(torch.float32))

class TripletLoss:
    def __init__(self, margin=0.2, alpha=1.0):
        self.margin = margin
        self.alpha = alpha

    def __call__(self, outputs, target, name, **kwargs):
        device = target.device
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        anchor_key = f"{name}_anchor_prob"
        if anchor_key in outputs:
            logits = outputs[anchor_key]
            total_loss += F.cross_entropy(logits.to(torch.float32), target.to(torch.long))

        pos = outputs.get(f"{name}_pos_similarity")
        neg = outputs.get(f"{name}_neg_similarity")
        
        if pos is not None and neg is not None:
             triplet_loss = F.relu(neg - pos + self.margin).mean()
             total_loss += self.alpha * triplet_loss.to(torch.float32)
             
        return total_loss
