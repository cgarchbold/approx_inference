import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_loss_accuracy(csv_file, dir='./results/'):
    # Plot the training and validation metrics
    data = pd.read_csv(csv_file)

    # Set default font sizes for plots
    plt.rcParams.update({
        'font.size': 32,          # Default font size for text
        'axes.titlesize': 32,     # Font size for plot titles
        'axes.labelsize': 32,     # Font size for axis labels
        'xtick.labelsize': 28,    # Font size for x-axis tick labels
        'ytick.labelsize': 28,    # Font size for y-axis tick labels
        'legend.fontsize': 32     # Font size for legend
    })


    # Plot training and validation loss
    plt.figure(figsize=(16, 9))
    plt.plot(data['Epoch'], data['Training Loss'], label='Training Loss', color='blue', alpha=0.7)
    plt.plot(data['Epoch'], data['Validation Loss'], label='Validation Loss', color='orange', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "training_loss.png"), dpi=400, transparent=True)
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(16, 9))
    plt.plot(data['Epoch'], data['Training Accuracy'], label='Training Accuracy', color='green', alpha=0.7)
    plt.plot(data['Epoch'], data['Validation Accuracy'], label='Validation Accuracy', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "training_accuracy.png"), dpi=400, transparent=True)
    plt.close()