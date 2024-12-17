import argparse
import os
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.calibration import calibration_curve

from bayesViT import bayesViT
from data import get_dataloaders

def compute_metrics(y_true, y_pred, average='macro'):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, precision, recall, f1

def save_softmax_to_csv(probabilities, targets, output_path):
    """Save softmax probabilities and targets to a CSV file."""
    df = pd.DataFrame(probabilities, columns=[f'class_{i}' for i in range(probabilities.shape[1])])
    df['target'] = targets
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

def test(args):
    # Load data
    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomHorizontalFlip(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    # Choose 2 random augmentations with magnitude 14
    #transform_train.transforms.insert(0, RandAugment(2, 14))
    #transform_train.transforms.insert(0, CIFAR10Policy())
    
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainloader, valloader, testloader, classes = get_dataloaders(
        args.batch_size, transform_train=transform_train, transform_test=transform_test
    )

    # Initialize model
    model = bayesViT(
        image_size=32, patch_size=4, num_classes=10, dim=384, depth=6, heads=8,
        mlp_dim=384, pool='cls', channels=3, dim_head=64, dropout=args.dropout_rate, emb_dropout=args.dropout_rate
    )

    # Load model weights
    ckpt_path = os.path.join('./results/', args.experiment_name,'best_model.pth')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint)

    # Set up output paths
    deterministic_output_path = f'results/{args.experiment_name}./deterministic.csv'
    probabilistic_output_path = f'results/{args.experiment_name}./probabilistic.csv'

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluation in deterministic mode
    model.eval()
    deterministic_probs, targets = [], []

    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            deterministic_probs.append(probabilities.cpu().numpy())
            targets.append(labels.cpu().numpy())

    deterministic_probs = np.vstack(deterministic_probs)
    targets = np.concatenate(targets)

    # Compute metrics
    preds = deterministic_probs.argmax(axis=1)
    acc, precision, recall, f1 = compute_metrics(targets, preds)
    print(f"Deterministic Metrics: Accuracy={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    # Save deterministic probabilities to CSV
    save_softmax_to_csv(deterministic_probs, targets, deterministic_output_path)

    #Evaluation in probabilistic mode
    model.train()

    def compute_entropy(probabilities):
        # Clip probabilities to avoid log(0) which is undefined
        probabilities = torch.clamp(probabilities, min=1e-10, max=1-1e-10)
        return -torch.sum(probabilities * torch.log(probabilities), dim=1)

    # Initialize variables to accumulate metrics over the entire test set
    total_entropy = 0.0
    total_variance = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            
            # Collect predictions from multiple forward passes with dropout
            outputs = []
            for _ in range(args.num_samples):
                outputs.append(F.softmax(model(images), dim=1))
            
            outputs = torch.stack(outputs)  # Shape: [num_samples, batch_size, num_classes]
            
            # Compute mean and variance across MC samples
            mean_output = torch.mean(outputs, dim=0)  # Mean prediction
            variance_output = torch.var(outputs, dim=0)  # Variance

            # 1. Compute entropy for each sample in the batch
            entropy = compute_entropy(mean_output)
            
            # 2. Compute average variance across classes
            average_variance = torch.mean(variance_output, dim=0)

            # Accumulate the metrics for the whole test set
            total_entropy += entropy.sum().item()  # Sum entropy for all samples in the batch
            total_variance += average_variance.sum().item()  # Sum average variance across classes
            total_samples += images.size(0)  # Batch size (number of samples in this batch)

    # Compute the average metrics over the entire test set
    avg_entropy = total_entropy / total_samples
    avg_variance = total_variance / total_samples

    # Print the average metrics
    print(f"Average Entropy over the Test Set: {avg_entropy}")
    print(f"Average Variance over the Test Set: {avg_variance}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bayesian Vision Transformer (bayesViT) model.")
    parser.add_argument('--epochs', type=int, default=400, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for training.")
    parser.add_argument('--experiment_name', type=str, default="FINAL_BayesViT_dropout0.3", help="Name of the experiment. A directory with this name will be created.")
    parser.add_argument('--dropout_rate', type=float, default=0.3, help="Dropout rate for the model (applies to both dropout and emb_dropout).")
    parser.add_argument('--num_samples', type=int, default=10, help="Number of samples to take during variational inference")
    args = parser.parse_args()

    test(args)

