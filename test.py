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
    trainloader, valloader, testloader, classes = get_dataloaders(
        args.batch_size, transform_train=transform_train, transform_test=transform_test
    )

    # Initialize model
    model = bayesViT(
        image_size=32, patch_size=4, num_classes=10, dim=384, depth=6, heads=8,
        mlp_dim=384, pool='cls', channels=3, dim_head=64, dropout=args.dropout_rate, emb_dropout=args.dropout_rate
    )

    # Load model weights
    checkpoint = torch.load(args.path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set up output paths
    deterministic_output_path = f'results/{args.experiment_name}_deterministic.csv'
    probabilistic_output_path = f'results/{args.experiment_name}_probabilistic.csv'

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Evaluation in deterministic mode
    model.eval()
    deterministic_probs, targets = [], []

    with torch.no_grad():
        for images, labels in testloader:
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

    # Evaluation in probabilistic mode
    model.train()
    uncertainty_scores, mutual_info_scores = [], []

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = []
        for _ in range(args.num_samples):
            outputs.append(model(images))
        outputs = torch.stack(outputs)  # Shape: [num_samples, batch_size, num_classes]

        # Compute probabilistic metrics
        mean_output = torch.mean(outputs, dim=0)  # Mean prediction
        variance_output = torch.var(outputs, dim=0)  # Variance

        probabilities = F.softmax(mean_output, dim=1).cpu().numpy()
        uncertainty = variance_output.sum(dim=1).cpu().numpy()  # Summed variance as uncertainty
        mutual_info = (F.softmax(outputs, dim=2) * F.log_softmax(outputs, dim=2)).mean(0).sum(1).cpu().numpy()

        uncertainty_scores.extend(uncertainty)
        mutual_info_scores.extend(mutual_info)

    # Save probabilistic results
    probabilistic_df = pd.DataFrame({
        'uncertainty': uncertainty_scores,
        'mutual_information': mutual_info_scores,
        'target': targets
    })
    probabilistic_df.to_csv(probabilistic_output_path, index=False)

    print(f"Probabilistic results saved to {probabilistic_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Bayesian Vision Transformer (bayesViT) model.")
    parser.add_argument('--epochs', type=int, default=400, help="Number of epochs to train the model.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training.")
    parser.add_argument('--experiment_name', type=str, default="Dropout0.1_test2", help="Name of the experiment. A directory with this name will be created.")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="Dropout rate for the model (applies to both dropout and emb_dropout).")
    parser.add_argument('--patience', type=int, default=35, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("--cutmix", action="store_true")
    parser.add_argument("--mixup", action="store_true")

    args = parser.parse_args()

    test(args)

