from torch import optim, nn
import torch
from transfer_learning import create_tranfer_learning_model
from train_test_fun import train
from data_setup import create_dataloader
from utils import train_transform, test_transform, plot_loss_curves
import os
import matplotlib.pyplot as plt
from timeit import default_timer as timer


if __name__ == "__main__":
    start_time = timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, test_dataloader, class_names = create_dataloader(
        train_path="./data_split/train",
        test_path="./data_split/test",
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=32,
        pin_memory=True
    )

    model = create_tranfer_learning_model(device=device)
    LR = 0.001
    LOSS_FN = nn.CrossEntropyLoss()
    OPTIMIZER = optim.Adam(model.parameters(), lr=LR)
    EPOCHS = 10

    results = train(
        model=model,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        optimizer=OPTIMIZER,
        loss_fn=LOSS_FN,
        num_epochs=EPOCHS,
        device=device
    )
    end_time = timer()

    print(f"Results:\n{results}\n")

    print(f"Time taken: {end_time-start_time:.3f} seconds")
    # Save plots
    os.makedirs("plots", exist_ok=True)
    plot_loss_curves(results)
    plt.savefig("plots/training_curves.png")
    plt.close()
