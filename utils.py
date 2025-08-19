from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Dict,List

def train_transform(size=224):
    """Transformations for training data (with augmentation)."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),       
        transforms.RandomHorizontalFlip(),    
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],     
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def test_transform(size=224):
    """Transformations for test/validation data (no augmentation)."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform

def plot_loss_curves(results:Dict[str,List[float]]):
  """ Plots training curves of a results dictionary """
  loss = results["train_loss"]
  test_loss = results["test_loss"]

  train_acc = results["train_acc"]
  test_acc = results["test_acc"]

  epochs = range(len(results["train_loss"]))

  plt.figure(figsize=(15,4))
  plt.subplot(1,2,1)
  plt.plot(epochs,loss,label="train_loss")
  plt.plot(epochs,test_loss,label="test_loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  plt.legend()

  plt.subplot(1,2,2)
  plt.plot(epochs,train_acc,label="train_accuracy")
  plt.plot(epochs,test_acc,label="test accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  plt.legend()