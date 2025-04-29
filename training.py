import torch
import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim
from Dataset import CustomCityscapesDataset
from utils import get_train_transform, get_train_target_transform, get_val_transform, get_val_target_transform, SegmentationLoss, Config
from SETR_model import SETR


# Learning rate scheduler
def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power=0.9):
    lr = init_lr * (1 - iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


########## Training and Validation Functions ##########
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    
    for images, targets in tqdm(dataloader):
        images = images.to(device)
        targets = targets.to(device).long()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        if isinstance(outputs, tuple):
            pred, aux_outputs = outputs
            loss = criterion(pred, targets, aux_outputs)
        else:
            loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def compute_confusion_matrix(preds, labels, num_classes, ignore_index=255):
    mask = (labels >= 0) & (labels != ignore_index) & (labels < num_classes)
    preds = preds[mask]
    labels = labels[mask]
    return torch.bincount(
        num_classes * labels + preds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def validate(model, dataloader, criterion, device, num_classes):
    model.eval()
    val_loss = 0.0
    confusion_matrix = torch.zeros(num_classes, num_classes).to(device)

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            targets = targets.to(device).long()

            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):
                preds, _ = outputs
            else:
                preds = outputs

            loss = criterion(preds, targets)
            val_loss += loss.item()

            # Get predictions
            preds = torch.argmax(preds, dim=1)

            # Vectorized confusion matrix update
            confusion_matrix += compute_confusion_matrix(preds, targets, num_classes)

    # mIoU computation
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(1) + confusion_matrix.sum(0) - intersection
    iou = intersection / (union + 1e-8)
    mean_iou = iou.mean().item()

    return val_loss / len(dataloader), mean_iou


########## Training Loop ##########
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, num_classes):
    best_miou = 0.0
    
    for epoch in range(num_epochs):
        # Update learning rate
        if scheduler == 'poly':
            curr_lr = poly_lr_scheduler(
                optimizer, config.LEARNING_RATE, 
                epoch * len(train_loader), num_epochs * len(train_loader)
            )
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_miou = validate(model, val_loader, criterion, device, num_classes)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), f'best_{config.MODEL_TYPE}_{config.TRANSFORMER_VARIANT}.pth')
    
    return model


########## Main ##########
# Main function to run the training pipeline
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.cuda.empty_cache()
    
    # Create datasets and data loaders
    train_transform = get_train_transform(config.IMAGE_SIZE)
    val_transform = get_val_transform(config.IMAGE_SIZE)
    train_target_transform = get_train_target_transform(config.IMAGE_SIZE)
    val_target_transform = get_val_target_transform(config.IMAGE_SIZE)
    
    train_dataset = CustomCityscapesDataset(
        root=config.DATASET_ROOT,
        split='train',
        transform=train_transform,
        target_transform=train_target_transform
    )
    
    val_dataset = CustomCityscapesDataset(
        root=config.DATASET_ROOT,
        split='val',
        transform=val_transform,
        target_transform=val_target_transform
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create model
    model = SETR(config).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # Initialize loss function, optimizer, and learning rate scheduler
    criterion = SegmentationLoss(ignore_index=255)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Train model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        config.LR_SCHEDULE,
        device,
        config.NUM_EPOCHS,
        config.NUM_CLASSES
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), f'final_{config.MODEL_TYPE}_{config.TRANSFORMER_VARIANT}.pth')
    
    print("Training completed!")
    
    
if __name__ == "__main__":
    config = Config()
    config.IMAGE_SIZE = (768, 768)  # Suggested by the paper
    config.MODEL_TYPE = 'SETR-Naive'  # Options: 'SETR-Naive', 'SETR-PUP', 'SETR-MLA' according to the paper
    main()