print("Code started")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, SubsetRandomSampler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import logging
import os
import shutil
from distutils.dir_util import copy_tree
import sys
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename='logname.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger()
logger.info("Sanity check")
memfs_path = os.environ.get("MEMFS")
logger.info(memfs_path)

# # Define the data directory
# data_dir = '/content/drive/MyDrive/cropped_testexamples/'

# Define the data directory
data_dir = '/net/ascratch/people/plgmnkpltrz/pretrainedResNet18/cropped'
copy_tree(data_dir,memfs_path)
logger.info(os.listdir(memfs_path))
data_dir = memfs_path


# Define the class names
class_names = ['_0_grade', '_1_grade', '_2_grade', '_3_grade', '_4_grade']
logger.info(device)
# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize(224),  # Resize the images to 224x224
    transforms.ToTensor(),   # Convert images to tensors
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Normalize image channels
        std=[0.229, 0.224, 0.225]
    )
])

# Load the dataset
dataset = ImageFolder(data_dir, transform=transform)

labels = dataset.targets  # Get the labels for the dataset

# Define the k-fold cross-validation parameters
num_folds = 10
# kfold = KFold(n_splits=num_folds, shuffle=True, random_state=123)
kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=123)

# For fold results
results = {}

# Define data augmentation configuration
data_augmentation = {
    '_0_grade': 2,  # Augment _0_grade class three times
    '_1_grade': 2,  # Augment _1_grade class three times
    '_2_grade': 0,  # No augmentation for _2_grade class
    '_3_grade': 2,  # Augment _3_grade class three times
    '_4_grade': 1   # Augment _4_grade class two times
}

# Define the ResNet-18 model
resnet_model = models.resnet18(pretrained=True)

# Unfreeze additional layers
for param in resnet_model.layer3.parameters():
    param.requires_grad = False

for param in resnet_model.layer4.parameters():
    param.requires_grad = True

# Replace the last fully connected layer with a new untrained one
num_features = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_features, len(class_names))
resnet_model = resnet_model.to(device)
dataset_indexes = list(range(len(dataset)))

# Lists to store accuracy results for each fold
train_accs = []
val_accs = []

# # Perform k-fold cross-validation
# for fold, (train_indexes, val_indexes) in enumerate(kfold.split(dataset_indexes)):
# Perform k-fold cross-validation
for fold, (train_indexes, val_indexes) in enumerate(kfold.split(dataset_indexes, labels)):
    train_sub = [dataset[n] for n in train_indexes]
    val_sub= [dataset[p] for p in val_indexes]

    train_images, train_labels = zip(*train_sub)
    val_images, val_labels = zip(*val_sub)

    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    val_images = torch.stack(val_images)
    val_labels = torch.tensor(val_labels)

    train_subset = torch.torch.utils.data.TensorDataset(train_images, train_labels)
    val_subset = torch.torch.utils.data.TensorDataset(val_images, val_labels)
    logger.info("Created subsets")

    print(f"Fold: {fold + 1}")

    # Print the number of images for each class before augmentation
    print("Number of images per class before augmentation:")
    for class_name in class_names:
        logger.info("Augmenting")
        class_indices = [idx for idx, (image, label) in enumerate(train_subset) if class_names[label] == class_name]
        print(f"{class_name}: {len(class_indices)}")

    # Apply data augmentation to specific classes
    augmented_images = []
    augmented_labels = []
    augmented_data = []
    for class_name, augment_times in data_augmentation.items():
        class_indices = [idx for idx, (image, label) in enumerate(train_subset) if class_names[label] == class_name]
        if augment_times > 0:
            for idx in class_indices:
                image, label = train_subset[idx]
                augmented_data.append((image, label))  # Add the original image once
                if class_name != '_2_grade':
                    for _ in range(augment_times):
                        augmented_image = transforms.RandomRotation(90)(image)
                        augmented_data.append((augmented_image, label))
    logger.info("Getting past augmentation")

    # Convert augmented images and labels to tensors
    augmented_images = torch.stack([data[0] for data in augmented_data])
    augmented_labels = torch.tensor([data[1] for data in augmented_data])

    # Create a new augmented subset
    augmented_subset = torch.utils.data.TensorDataset(augmented_images, augmented_labels)

    # Combine the original train subset and augmented subset
    train_subset = torch.utils.data.ConcatDataset([train_subset, augmented_subset])

    # Print the number of images for each class after augmentation
    print("Number of images per class after augmentation:")
    for class_name in class_names:
        class_indices = [idx for idx, (image, label) in enumerate(train_subset) if class_names[label] == class_name]
        print(f"{class_name}: {len(class_indices)}")
    print()

    def custom_collate(batch):
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.int64)
        return images, labels

    # Create data loaders for training and validation
    batch_size = 20
    train_dataloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=custom_collate)
    val_dataloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Compile and train the model on the current fold
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(resnet_model.fc.parameters(), lr=0.0001)  # Specify a lower learning rate

    # Variables to track best model and accuracies
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_model = None
    train_preds = []
    val_preds = []
    train_true_labels = []
    val_true_labels = []

    logging.info("Getting here to the training part")

    val_loss = 0.0
    correct = 0
    total = 0
    
    # Training loop for each fold
    for epoch in range(50):
        # Training phase
        resnet_model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = resnet_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # Store predictions and true labels for calculating confusion matrix
            
        # Calculate accuracy and loss for the training set
        train_acc = 100. * correct / total
        train_loss /= len(train_dataloader)

        # Validation phase
        resnet_model.eval()

        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = resnet_model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Store predictions and true labels for calculating confusion matrix
                

        # Calculate accuracy and loss for the validation set
        val_acc = 100. * correct / total
        val_loss /= len(val_dataloader)

        # Print training and validation accuracy for each fold
        print(f"Fold: {fold + 1}, Epoch: {epoch + 1}, Train Accuracy: {train_acc:.2f}%, Val Accuracy: {val_acc:.2f}%")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_model = resnet_model.state_dict()

            # Save the best model
            torch.save(best_model, "transfer_learn_ResNet18_best_model.pth")
    
    # Calculate confusion matrices
    resnet_model.eval()
    with torch.no_grad():
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = resnet_model(images)
            _, predicted = outputs.max(1)
            train_preds.extend(predicted.tolist())
            train_true_labels.extend(labels.tolist())

        for images_val, labels_val in val_dataloader:
            images_val = images_val.to(device)
            labels_val = labels_val.to(device)
            outputs_val = resnet_model(images_val)
            _, predicted_val = outputs_val.max(1)
            val_preds.extend(predicted_val.tolist())
            val_true_labels.extend(labels_val.tolist())
    train_confusion_matrix = confusion_matrix(train_true_labels, train_preds)
    val_confusion_matrix = confusion_matrix(val_true_labels, val_preds)

    print(train_confusion_matrix)
    print(val_confusion_matrix)

    # Append accuracy results for each fold
    results[f"fold {fold + 1} train accuracy"] = best_train_acc
    results[f"fold {fold + 1} val accuracy"] = best_val_acc

# Calculate mean training and validation accuracy from folds
mean_train_acc = np.mean([results[f"fold {i + 1} train accuracy"] for i in range(num_folds)])
mean_val_acc = np.mean([results[f"fold {i + 1} val accuracy"] for i in range(num_folds)])

# Calculate standard deviation of training and validation accuracy from folds
std_train_acc = np.std([results[f"fold {i + 1} train accuracy"] for i in range(num_folds)])
std_val_acc = np.std([results[f"fold {i + 1} val accuracy"] for i in range(num_folds)])

# Print best accuracy obtained within folds on train and validation datasets
print("Results from all folds:")
for i in range(num_folds):
    print(f"Fold {i + 1} Train Accuracy: {results[f'fold {i + 1} train accuracy']:.2f}%")
    print(f"Fold {i + 1} Val Accuracy: {results[f'fold {i + 1} val accuracy']:.2f}%")

# Print mean and standard deviation of training and validation accuracy
print("Mean Training Accuracy: {:.2f}% (Std: {:.2f})".format(mean_train_acc, std_train_acc))
print("Mean Validation Accuracy: {:.2f}% (Std: {:.2f})".format(mean_val_acc, std_val_acc))
