import random
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import json
import torchvision.models as models
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import wandb
from utils import (
    EarlyStopping,
    sample_hyperparameters,
    augment_subset,
    configure_loss,
    initialize_run_directory,
    CLASS_COUNTS,
    calculate_sample_weights,
    transform
)
from torch.utils.data import WeightedRandomSampler

torch.set_float32_matmul_precision("medium")

hyperparameter_space = {
    "learning_rate": [0.0001, 0.0005, 0.001],
    "batch_size": [16, 32, 64],
    "criterion": ["CrossEntropyLoss"]
    # Other hyperparameters here
}


wandb.login(key="Here goes wandb API key")

# Set random seeds for reproducibility
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
epochs = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the data directory
data_dir = "/net/tscratch/people/plgmnkpltrz/gliomaclass/gliomaclass/datasplitted/cropped"
#data_dir = "/net/tscratch/people/plgmnkpltrz/gliomaclass/gliomaclass/HLA_hsv/HLA_hsv"

# Define the class names
class_names = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"]

# Load the dataset
#dataset = ImageFolder(data_dir, transform=transform)
dataset = ImageFolder(data_dir, transform=transforms.ToTensor())

# Define augmentation transformations

# Define the k-fold cross-validation parameters
num_folds = 10
num_classes = len(class_names)
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=123)


# Initialize variables to track the best fold and its accuracy
best_fold = -1
best_fold_accuracy = 0.0
best_fold_hyperparameters = None
fold_accuracies = []
# Perform k-fold cross-validation
hyperparams = sample_hyperparameters(hyperparameter_space)
batch_size = hyperparams["batch_size"]
learning_rate = hyperparams["learning_rate"]
criterion = configure_loss(hyperparams["criterion"]).to(device)
result_save_path = initialize_run_directory(params=hyperparams)
for fold, (train_indexes, test_indexes) in enumerate(
    kfold.split(np.array(dataset.targets), dataset.targets)
):
    print(f"Fold: {fold + 1}")
    fold_folder = os.path.join(result_save_path, f"Fold_{fold+1}")
    os.makedirs(fold_folder, exist_ok=True)
    # Initialize a new W&B run
    wandb.init(
        project="resnet18pretrained", name=f"Fold_{fold + 1}", config=hyperparams
    )

    # Initialize model for this fold
    model = models.resnet18(pretrained=True).to(device) # Load a pretrained ResNet18
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    # Unfreeze the last layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.requires_grad = True # Unfreeze only the final layer
    model = model.to(device)

    print(
        "Model device after initialization:", next(model.parameters()).device
    )

    # Split dataset into train and validation subsets
    train_subset_general = Subset(dataset, train_indexes)
    test_subset = Subset(dataset, test_indexes)
    train_labels = np.array([y for (x, y) in train_subset_general])

    current_train_indices = np.array(list(range(len(train_subset_general))))
    train_indices, val_indices = train_test_split(
        current_train_indices, test_size=0.1, stratify=train_labels
    )
    train_subset = Subset(train_subset_general, train_indices)
    val_subset = Subset(train_subset_general, val_indices)
    # Augment the training subset with weighted class augmentation
    augmented_train_subset = augment_subset(train_subset, CLASS_COUNTS)

    # Convert augmented training subset to DataLoader
    train_loader = DataLoader(
        augmented_train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        prefetch_factor=2,
    )

    # Calculate sample weights and create the sampler
    sample_weights = calculate_sample_weights(augmented_train_subset)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Update the DataLoader to use the weighted random sampler
    train_loader = DataLoader(
        augmented_train_subset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(
        delta=0.01, patience=7, path="temp_chkpt/res18pre_checkpoint.pt"
    )

    for epoch in range(epochs):
        # Training phase
        train_preds = []
        train_gt = []
        train_loss = 0
        train_correct = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.detach(), 1)
            train_preds.extend(predicted.cpu().numpy())
            train_gt.extend(labels.cpu().numpy())
            loss = criterion(outputs, labels)
            train_loss += loss.detach().cpu().item()
            train_correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        train_loss /= i + 1
        accuracy_train = np.mean(np.array(train_preds) == np.array(train_gt))
        precision_train = precision_score(
            train_gt, train_preds, average="macro"
        )
        recall_train = recall_score(train_gt, train_preds, average="macro")
        f1_train = f1_score(train_gt, train_preds, average="macro")
        conf_matrix_train = confusion_matrix(train_gt, train_preds)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}")
        print(f"Train accuracy: {accuracy_train :.4f}")
        print(f"Train precision: {precision_train :.4f}")
        print(f"Train recall: {recall_train :.4f}")
        print(f"Train f1: {f1_train :.4f}")

        # Validation phase
        model.eval()
        val_preds = []
        val_gt = []
        val_correct = 0
        val_loss = 0
        with torch.no_grad():
            for n, (images, labels) in enumerate(val_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_loss += criterion(outputs, labels).cpu().item()
                val_preds.extend(predicted.cpu().numpy())
                val_gt.extend(labels.cpu().numpy())
                val_correct += (predicted == labels).sum().item()
            val_loss /= n + 1
        accuracy_val = np.mean(np.array(val_preds) == np.array(val_gt))
        precision_val = precision_score(val_gt, val_preds, average="macro")
        recall_val = recall_score(val_gt, val_preds, average="macro")
        f1_val = f1_score(val_gt, val_preds, average="macro")
        conf_matrix_val = confusion_matrix(val_gt, val_preds)
        print(f"Epoch: {epoch + 1}, Val Loss: {val_loss:.4f}")
        print(f"Val accuracy: {accuracy_val:.4f}")
        print(f"Val precision: {precision_val:.4f}")
        print(f"Val recall: {recall_val:.4f}")
        print(f"Val f1: {f1_val:.4f}")

        wandb.log(
            {
                "train_accuracy": accuracy_train,
                "train_loss": train_loss,
                "train_precision": precision_train,
                "train_recall": recall_train,
                "train_f1": f1_train,
                "val_accuracy": accuracy_val,
                "val_loss": val_loss,
                "val_precision": precision_val,
                "val_recall": recall_val,
                "val_f1": f1_val,
            },
            commit=False,
        )
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        # after training, load the model with the lowest validation loss
    best_model_state = torch.load("temp_chkpt/res18pre_checkpoint.pt")
    model.load_state_dict(best_model_state)

    # Testing
    test_preds = []
    test_gt = []
    test_correct = 0
    test_loss = 0
    with torch.no_grad():
        for n, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            test_loss += criterion(outputs, labels).cpu().item()
            test_preds.extend(predicted.cpu().numpy())
            test_gt.extend(labels.cpu().numpy())
            test_correct += (predicted == labels).sum().item()
        test_loss /= n + 1
    accuracy = np.mean(np.array(test_preds) == np.array(test_gt))
    precision = precision_score(test_gt, test_preds, average="macro")
    recall = recall_score(test_gt, test_preds, average="macro")
    f1 = f1_score(test_gt, test_preds, average="macro")
    conf_matrix = confusion_matrix(test_gt, test_preds)
    print(f"Epoch: {epoch + 1}, Test Loss: {test_loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    print(f"Test f1: {f1:.4f}")

    wandb.log(
        {
            "test_accuracy": accuracy,
            "test_loss": test_loss,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1,
            "test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_gt,
                preds=test_preds,
                class_names=class_names,
            ),
        },
    )
    # Finish the W&B run
    wandb.finish()
    # save the best model, train indices, val indices, test indices
    indices_dict = {
        "train_indices": [float(i) for i in train_indices],
        "val_indices": [float(i) for i in val_indices],
        "test_indices": [float(i) for i in test_indexes],
    }
    with open(os.path.join(fold_folder, "indices.json"), "w") as f:
        json.dump(indices_dict, f)
    torch.save(model.state_dict(), os.path.join(fold_folder, "res18pre_model.pt"))
