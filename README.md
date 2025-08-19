# PyTorch Transfer Learning with EfficientNet-B0

A PyTorch implementation for image classification using transfer learning with EfficientNet-B0 as the backbone model.

## Overview

This project implements a transfer learning pipeline that:
- Uses a pre-trained EfficientNet-B0 model as a feature extractor
- Freezes the base model parameters to preserve learned features
- Adds a custom classification head for your specific dataset
- Supports training on datasets with up to 34 classes (configurable)

## Project Structure

```
├── base_model.py           # EfficientNet-B0 base model setup
├── data_preprocessing.py   # Dataset splitting script
├── data_setup.py          # DataLoader creation utilities
├── train.py               # Main training script
├── train_test_fun.py      # Training and testing functions
├── transfer_learning.py   # Transfer learning model creation
├── utils.py               # Utility functions and transforms
└── README.md              # This file
```

## Requirements

```bash
pip install torch torchvision matplotlib scikit-learn tqdm
```

## Dataset Setup

### 1. Organize Your Data

Place your images in the following structure:
```
data/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

### 2. Split Dataset

Run the preprocessing script to automatically split your data into train/test sets (80/20 split):

```bash
python data_preprocessing.py
```

This creates:
```
data_split/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Training

### Quick Start

```bash
python train.py
```

### Configuration

You can modify training parameters in `train.py`:

```python
# Training hyperparameters
LR = 0.001              # Learning rate
EPOCHS = 10             # Number of training epochs
batch_size = 32         # Batch size

# Model configuration
num_classes = 34        # Number of classes in your dataset
```

### Custom Configuration

To use different settings:

```python
from transfer_learning import create_tranfer_learning_model
from data_setup import create_dataloader

# Create model for different number of classes
model = create_tranfer_learning_model(
    device=device,
    num_classes=10,  # Change this to match your dataset
    use_pretrained=True
)

# Create dataloaders with custom batch size
train_dataloader, test_dataloader, class_names = create_dataloader(
    train_path="./data_split/train",
    test_path="./data_split/test",
    train_transform=train_transform,
    test_transform=test_transform,
    batch_size=64,  # Custom batch size
    pin_memory=True
)
```

## Model Architecture

The transfer learning model consists of:

1. **Base Model**: Pre-trained EfficientNet-B0 (frozen)
   - Feature extraction only
   - All parameters frozen to preserve learned representations

2. **Custom Classifier Head**:
   ```python
   classifier = nn.Sequential(
       nn.Dropout(0.3),
       nn.Linear(in_features=1280, out_features=num_classes)
   )
   ```

## Data Augmentation

### Training Transforms
- Resize to 224×224 pixels
- Random horizontal flip
- Normalization with ImageNet statistics

### Test Transforms
- Resize to 224×224 pixels
- Normalization with ImageNet statistics (no augmentation)

## Output

After training, the script will:

1. **Print Results**: Training and validation metrics for each epoch
2. **Save Plots**: Training curves saved to `plots/training_curves.png`
3. **Display Timing**: Total training time

## Example Output

```
Epoch 1/10
Train Loss: 2.834 | Train Acc: 0.234
Test Loss: 2.456 | Test Acc: 0.312

Epoch 2/10
Train Loss: 1.923 | Train Acc: 0.456
Test Loss: 1.678 | Test Acc: 0.523
...

Results:
{'train_loss': [2.834, 1.923, ...], 'train_acc': [0.234, 0.456, ...], 
 'test_loss': [2.456, 1.678, ...], 'test_acc': [0.312, 0.523, ...]}

Time taken: 245.123 seconds
```

## Key Features

- **Transfer Learning**: Leverages pre-trained EfficientNet-B0 for better performance
- **Automatic GPU Detection**: Uses CUDA if available, falls back to CPU
- **Progress Tracking**: Real-time training progress with tqdm
- **Visualization**: Automatic plotting of loss and accuracy curves
- **Flexible Configuration**: Easy to modify for different datasets and parameters

## Customization

### Different Base Models

To use a different base model, modify `base_model.py`:

```python
from torchvision.models import resnet50, ResNet50_Weights

def create_base_model(use_pretrained=True):
    weights = ResNet50_Weights.DEFAULT if use_pretrained else None
    base_model = resnet50(weights=weights)
    num_features = base_model.fc.in_features
    base_model.fc = nn.Identity()
    return base_model, num_features
```

### Different Optimizers

Modify the optimizer in `train.py`:

```python
# Use SGD instead of Adam
OPTIMIZER = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

# Use different learning rates for different parts
OPTIMIZER = optim.Adam([
    {'params': model[0].parameters(), 'lr': 0.0001},  # Base model
    {'params': model[1].parameters(), 'lr': 0.001}    # Classifier
])
```

## License

MIT License
