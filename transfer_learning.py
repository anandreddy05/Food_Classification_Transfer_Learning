from base_model import create_base_model
from torch import nn
import torch

def create_tranfer_learning_model(device:torch.device,num_classes: int = 34, use_pretrained=True):
    """
    Creates a transfer learning model using EfficientNet-B0 as the base.

    The function loads a pretrained EfficientNet-B0 (if `use_pretrained=True`), 
    removes its classifier head, freezes all base layers, and attaches a new 
    classification head suitable for the target dataset.

    Args:
        num_classes (int, optional): Number of classes for the classification 
            head. Defaults to 34.
        use_pretrained (bool, optional): Whether to load pretrained weights 
            for EfficientNet-B0. Defaults to True.
        device: cpu or cuda

    Returns:
        torch.nn.Sequential: A sequential model combining the frozen base 
        EfficientNet-B0 and a custom classification head.
    """
    base_model, num_features = create_base_model()
    
    for param in base_model.parameters():
        # Freeze all the parameters of the base_model
        param.requires_grad = False 
    
    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features=num_features, out_features=num_classes)  # type: ignore
    )
    
    model = nn.Sequential(base_model, classifier)
    return model.to(device=device)