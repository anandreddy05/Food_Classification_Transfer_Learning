from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch import nn

def create_base_model(use_pretrained=True):
    """
    Loads EfficientNet-B0 and removes the classifier head.

    Returns:
        base_model: feature extractor
        num_features: number of features output by base_model
    """
    weights = EfficientNet_B0_Weights.DEFAULT if use_pretrained else None
    base_model = efficientnet_b0(weights=weights)
    
    # Get number of features from original classifier
    num_features = base_model.classifier[1].in_features
    
    # Remove entire classifier (dropout + linear)
    base_model.classifier = nn.Identity() # type: ignore
    
    return base_model, num_features