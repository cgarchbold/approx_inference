import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from bayesViT import bayesViT
from data import get_dataloaders


def train(epochs=100, learning_rate = 0.0001, batch_size = 64):

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainloader, testloader, classes = get_dataloaders(batch_size, transform)

    model = bayesViT(
        image_size =32, patch_size = 4, num_classes =10, dim=128, depth=6, heads =4,
         mlp_dim=1024, pool = 'cls', channels = 3, dim_head = 32, dropout = 0.1, emb_dropout = 0.1
    ).cuda()

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Iterate over the training data
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
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
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        
        # Validation step (using dropout at test time for Monte Carlo estimates)
        if (epoch + 1) % 10 == 0:  # You can adjust how often you want validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_accuracy = 100 * val_correct / val_total
            print(f"Validation Accuracy: {val_accuracy:.2f}%")
            model.train()  # Switch back to training mode

    print("Training Complete!")

if __name__ == "__main__":
    train()