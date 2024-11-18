# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # For mixed-precision training
from data_loader import get_data_loaders
from models.cmil import FeatureExtractor, SliceFusionTransformer, EndToEndModel

def train():
    # Configuration
    data_dir = '/path/to/ct/volumes'  # Replace with your data directory
    batch_size = 1
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters for the transformer
    embed_dim = 768  # Must match the embed_dim in FeatureExtractor (DINOv2 output)
    num_heads = 8
    hidden_dim = 2048
    num_layers = 1
    patch_size = 1  # Patch size for the SliceFusionTransformer

    # Get DataLoaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)

    # Instantiate models
    feature_extractor = FeatureExtractor().to(device)
    # Assuming maximum sequence length (number of slices) is known or set a reasonable maximum
    max_seq_len = 256  # Adjust based on your data
    transformer_model = SliceFusionTransformer(
        seq_len=max_seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        patch_size=patch_size
    ).to(device)
    model = EndToEndModel(feature_extractor, transformer_model, num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # For mixed-precision training

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for i, batch_data in enumerate(train_loader):
            images = batch_data['image']  # Shape: (batch_size, C, D, H, W)
            labels = batch_data['label']
            images = images.to(device)
            labels = labels.to(device)

            # Prepare images for the model
            batch_size_curr, C, D, H, W = images.shape
            images = images.permute(0, 2, 1, 3, 4)  # Shape: (batch_size, D, C, H, W)

            optimizer.zero_grad()

            with autocast():
                # Forward pass
                outputs = model(images)  # Shape: (batch_size, num_classes)

                # Compute loss
                loss = criterion(outputs, labels)

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Loss: {running_loss / (i + 1):.4f}, '
                      f'Accuracy: {100 * correct_preds / total_samples:.2f}%')

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_data in val_loader:
                images = batch_data['image']
                labels = batch_data['label']
                images = images.to(device)
                labels = labels.to(device)

                batch_size_curr, C, D, H, W = images.shape
                images = images.permute(0, 2, 1, 3, 4)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    print('Training complete.')

if __name__ == '__main__':
    train()
