from enum import Enum

class TaskType(Enum):
    STS = "STS"
    PI = "PI"
    TRIPLET = "Triplet"

    @classmethod
    def from_str(cls, label):
        if label == "regression":
            return cls.STS
        elif label == "binary_classification":
            return cls.PI
        elif label == "contrastive_logit":
            return cls.TRIPLET
        elif label in [t.value for t in cls]:
            return cls(label)
        raise ValueError(f"Unknown task type: {label}")

class DistanceMetric(Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    
    @classmethod
    def from_str(cls, label):
        return cls(label)
