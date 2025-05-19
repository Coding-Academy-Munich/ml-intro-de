# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>CIFAR-100 mit Skorch</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import numpy as np
import sklearn.metrics
import torch
import torchvision
import torchvision.transforms as transforms

from skorch import NeuralNetClassifier
from torch.utils.data import DataLoader, Subset
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# %%
# Dictionary mapping CIFAR-100 class indices to names
CIFAR100_CLASSES = torchvision.datasets.CIFAR100(
    root='./localdata', train=True, download=True).class_to_idx

# %%
def get_class_indices(class_names: list[str]) -> list[int]:
    """Convert class names to CIFAR-100 indices."""
    return [CIFAR100_CLASSES[name.lower()] for name in class_names]

# %%
def create_data_loaders(class_names: list[str], batch_size: int = 32):
    """Create training and test data loaders for specified classes."""
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR100(root='./localdata', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./localdata', train=False,
                                           download=True, transform=transform_test)

    # Get indices for desired classes
    class_indices = get_class_indices(class_names)

    # Filter datasets
    train_mask = np.isin(trainset.targets, class_indices)
    test_mask = np.isin(testset.targets, class_indices)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    trainset = Subset(trainset, train_indices)
    testset = Subset(testset, test_indices)

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size,
                            shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size,
                           shuffle=False, num_workers=2)

    return train_loader, test_loader

# %%
def create_classifier_skorch(class_names: list[str]):
    """Create a fine-tunable classifier using skorch."""
    # Load pre-trained EfficientNetV2
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier = torch.nn.Linear(model.classifier[1].in_features, len(class_names))

    clf = NeuralNetClassifier(
        module=model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=0.001,
        batch_size=32,
        max_epochs=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        train_split=None,  # We'll handle validation separately
        iterator_train__num_workers=2,
        iterator_valid__num_workers=2,
    )

    return clf


# %%
def train_and_evaluate_skorch(class_names: list[str]):
    """Train and evaluate a classifier using skorch."""
    # Create data loaders
    train_loader, test_loader = create_data_loaders(class_names)

    # Create classifier
    clf = create_classifier_skorch(class_names)

    # Convert data to numpy arrays
    X_train = np.array([x for x, _ in train_loader.dataset])
    y_train = np.array([y for _, y in train_loader.dataset])
    X_test = np.array([x for x, _ in test_loader.dataset])
    y_test = np.array([y for _, y in test_loader.dataset])

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy:.4f}")

    return clf

# %%
class_names = ["lion", "tiger", "leopard"]

# %%
train_and_evaluate_skorch(class_names)

# %%
