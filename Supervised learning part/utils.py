import datetime
import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Tuple
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
])

# Define augmentation transformations
AUGMENTATION_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Horizontal reflection
    transforms.RandomVerticalFlip(),    # Vertical reflection
    #transforms.RandomRotation(30)      # Rotation up to 30 degrees
])

# Calculate class weights for weighted loss
CLASS_COUNTS = [17, 20, 68, 21, 41]
CLASS_WEIGHTS = torch.tensor(
    [sum(CLASS_COUNTS) / c for c in CLASS_COUNTS], dtype=torch.float
)
INVERSE_WEIGHTS = 1.0 / CLASS_WEIGHTS
INVERSE_WEIGHTS = INVERSE_WEIGHTS / INVERSE_WEIGHTS.sum()  # Normalize weights


class EarlyStopping:
    """Credit to https://github.com/Bjarten/early-stopping-pytorch"""

    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model) -> None:
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        return None


# Function to randomly select hyperparameters
def sample_hyperparameters(space: Dict[str, List]) -> Dict[str, Any]:
    return {param: random.choice(values) for param, values in space.items()}


def configure_loss(loss_name: str) -> nn.Module:
    """Configure the loss function based on the loss name
    Args:
        loss_name (str): Name of the loss function
    Returns:
        nn.Module: The loss function
    """

    if loss_name == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:  # WeightedCrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=INVERSE_WEIGHTS)
    return criterion


# Function to augment a subset based on class counts
def augment_subset(subset, class_counts):
    augmented_images = []
    for idx in subset.indices:
        image, label = subset.dataset[idx]
        # Determine the number of times to augment each image based on its class
        augment_times = round(sum(class_counts) / class_counts[label])
        for _ in range(augment_times):
            augmented_image = AUGMENTATION_TRANSFORMS(image)
            augmented_images.append((augmented_image, label))
    return augmented_images


def initialize_run_directory(
    params: Dict, resutls_save_dir: str = "results"
) -> str:
    """Initialize a new directory to save the results of the run.
    Args:
        resutls_save_dir (str): Directory to save the results
    Returns:
        str: Path to the directory to save the results
    """
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    loss = params["criterion"]
    folder_name = f"batch_size_{batch_size}_lr_{learning_rate}_loss_{loss}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_path = os.path.join(resutls_save_dir, folder_name)
    os.makedirs(save_path, exist_ok=True)
    return save_path


# import os
# from datetime import datetime

# def initialize_run_directory(params, base_dir="/net/tscratch/people/plgmnkpltrz/gliomaclass/gliomaclass/results"):
#     # Generate a timestamp
#     current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
#     # Create a directory name based on hyperparameters and timestamp
#     dir_name = f"run_lr_{params['learning_rate']}_bs_{params['batch_size']}_{current_time}"
#     path = os.path.join(base_dir, dir_name)
#     os.makedirs(path, exist_ok=True)
#     return path

def calculate_sample_weights(dataset):
    class_weights = INVERSE_WEIGHTS
    sample_weights = [0] * len(dataset)

    for idx, (_, label) in enumerate(dataset):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    return sample_weights