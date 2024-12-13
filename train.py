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

def train(epochs=100, learning_rate=0.0001, batch_size=64, experiment_name="experiment", dropout_rate=0.1, patience=5, lr_decay_step=10, lr_decay_gamma=0.5):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainloader, valloader, testloader, classes = get_dataloaders(batch_size, transform)

    model = bayesViT(
        image_size=32, patch_size=4, num_classes=10, dim=128, depth=6, heads=4,
        mlp_dim=1024, pool='cls', channels=3, dim_head=32, dropout=dropout_rate, emb_dropout=dropout_rate
    ).cuda()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
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

        # Validation step (using dropout at test time for Monte Carlo estimates)
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

    # Save the final model
    print("Saving final model....")
    model_save_path = os.path.join(experiment_dir, "model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

    print("Plotting...")
    plot_loss_accuracy(csv_file=csv_file, dir=experiment_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bayesian Vision Transformer (bayesViT) model.")
    parser.add_argument('--epochs', type=int, default=250, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--experiment_name', type=str, default="Dropout0.1", help="Name of the experiment. A directory with this name will be created.")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for the model (applies to both dropout and emb_dropout).")
    parser.add_argument('--patience', type=int, default=5, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--lr_decay_step', type=int, default=50, help="Number of epochs between learning rate decays.")
    parser.add_argument('--lr_decay_gamma', type=float, default=0.1, help="Factor by which the learning rate will be decayed.")

    args = parser.parse_args()

    train(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size, experiment_name=args.experiment_name, dropout_rate=args.dropout_rate, patience=args.patience, lr_decay_step=args.lr_decay_step, lr_decay_gamma=args.lr_decay_gamma)
