import argparse
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import csv

from bayesViT import bayesViT
from data import get_dataloaders
from plot import plot_loss_accuracy
from randomaug import RandAugment

def train(epochs=100, learning_rate=0.0001, batch_size=64, experiment_name="experiment", dropout_rate=0.1, patience=5, min_lr = 1e-5):

    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomHorizontalFlip(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # Choose 2 random augmentations with magnitude 14
    transform_train.transforms.insert(0, RandAugment(2, 14))
    
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    trainloader, valloader, testloader, classes = get_dataloaders(batch_size, transform_train=transform_train, transform_test=transform_test)

    model = bayesViT(
        image_size=32, patch_size=4, num_classes=10, dim=384, depth=6, heads=8,
        mlp_dim=384, pool='cls', channels=3, dim_head=64, dropout=dropout_rate, emb_dropout=dropout_rate
    ).cuda()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    criterion = nn.CrossEntropyLoss()

    # Create directory for experiment
    experiment_dir = f"./results/{experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # CSV setup
    csv_file = os.path.join(experiment_dir, "training_metrics.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Training Loss", "Training Accuracy", "Validation Loss", "Validation Accuracy"])

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # Iterate over the training data
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", ncols=100):
            inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass

            optimizer.step()  # Update the weights

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate the average loss and accuracy for the epoch
        epoch_loss = running_loss / len(trainloader)
        epoch_accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validation step 
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(valloader)
        val_accuracy = 100 * val_correct / val_total

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        model.train()  # Switch back to training mode

        # Save metrics to CSV
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, epoch_loss, epoch_accuracy, val_loss, val_accuracy])

        # Early stopping: Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model based on validation loss
            model_save_path = os.path.join(experiment_dir, "best_model.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Best model saved at {model_save_path}")
        else:
            epochs_without_improvement += 1

        # Early stopping: Stop if no improvement for 'patience' epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

        # Step the learning rate scheduler
        scheduler.step()

    print("Training Complete!")

    print("Plotting...")
    plot_loss_accuracy(csv_file=csv_file, dir=experiment_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bayesian Vision Transformer (bayesViT) model.")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--experiment_name', type=str, default="Dropout0.1_test1", help="Name of the experiment. A directory with this name will be created.")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for the model (applies to both dropout and emb_dropout).")
    parser.add_argument('--patience', type=int, default=35, help="Number of epochs with no improvement after which training will be stopped.")

    args = parser.parse_args()

    train(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, experiment_name=args.experiment_name, 
          dropout_rate=args.dropout_rate, patience=args.patience)
