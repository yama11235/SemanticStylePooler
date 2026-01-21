import torch
from torch import nn
from .constants import TaskType

class BaseHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def encode(self, embedding):
        raise NotImplementedError

class STSHead(BaseHead):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.output_dim),
        )
        
    def forward(self, embedding):
        return self.linear(embedding)
    
    def encode(self, embedding):
        return self.forward(embedding)

class PIHead(BaseHead):
    def __init__(self, config):
        super().__init__(config)
        self.linear = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.output_dim),
        )
        
    def forward(self, embedding):
        return self.linear(embedding)
    
    def encode(self, embedding):
        return self.forward(embedding)

class TripletHead(BaseHead):
    def __init__(self, config):
        super().__init__(config)
        self.embedder = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.input_dim, config.intermediate_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.intermediate_dim, config.output_dim),
        )

    def forward(self, x):
        embedding = self.embedder(x)
        logits = self.classifier(embedding)
        if logits.size(-1) == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=-1)
        return embedding, probs
    
    def encode(self, x):
        return self.embedder(x)

# Backward compatibility
LinearLayer = STSHead
ContrastiveClassifier = TripletHead
