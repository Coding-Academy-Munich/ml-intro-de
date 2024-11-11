# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>CIFAR-100 mit Keras</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# %%
os.environ["KERAS_BACKEND"] = "torch"

# %%
import keras
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import Sequence
from torch.utils.data import DataLoader, Subset

# %%
# Dictionary mapping CIFAR-100 class indices to names
CIFAR100_CLASSES = torchvision.datasets.CIFAR100(
    root='./localdata', train=True, download=True).class_to_idx

# %%
def get_class_indices(class_names: list[str]) -> list[int]:
    """Convert class names to CIFAR-100 indices."""
    return [CIFAR100_CLASSES[name.lower()] for name in class_names]


# %%
class_idx_mapping: dict[int, int] = {}

# %%
def target_transform(label):
    return class_idx_mapping.get(label, -1)  # Map to new index, or -1

# %%
def create_data_loaders(class_names: list[str], batch_size: int = 32):
    """Create training and test data loaders for specified classes."""
    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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

    # Get class indices for desired classes
    class_indices = get_class_indices(class_names)

    # Create mapping from original indices to 0..len(class_names)-1
    class_idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(class_indices)}

    # Load full datasets without target_transform
    trainset_full = torchvision.datasets.CIFAR100(
        root='./localdata', train=True,
        download=True,
        transform=transform_train
    )
    testset_full = torchvision.datasets.CIFAR100(
        root='./localdata', train=False,
        download=True,
        transform=transform_test
    )

    # Filter indices of samples belonging to desired classes
    train_indices = [i for i, lbl in enumerate(trainset_full.targets) if lbl in class_indices]
    test_indices = [i for i, lbl in enumerate(testset_full.targets) if lbl in class_indices]

    # Create subsets of the data
    trainset = Subset(trainset_full, train_indices)
    testset = Subset(testset_full, test_indices)

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    return train_loader, test_loader, class_idx_mapping




# %%
def create_classifier_keras(class_names: list[str], learning_rate: float = 0.0001, fine_tune_at: int = 100):
    """Create a fine-tunable classifier using Keras 3."""
    # Load pre-trained EfficientNetV2S with correct input shape
    base_model = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg"
    )

    # Initially freeze base model
    base_model.trainable = False

    # Create new model
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    # Fine-tune the base model
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model



# %%
class DataGenerator(Sequence):
    def __init__(self, loader, class_idx_mapping):
        super().__init__()
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.indices = np.arange(len(self.dataset))  # Local indices for the subset
        self.shuffle = True
        self.class_idx_mapping = class_idx_mapping
        self.on_epoch_end()

    def __len__(self):
        # Return the number of batches per epoch
        return max(1, int(np.ceil(len(self.dataset) / self.batch_size)))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_idx:end_idx]

        batch_samples = [self.dataset[i] for i in batch_indices]
        images = torch.stack([sample[0].permute(1, 2, 0) for sample in batch_samples])
        labels = torch.tensor([self.class_idx_mapping[sample[1]] for sample in batch_samples], dtype=torch.long)
        return images, labels

    def on_epoch_end(self):
        # Shuffle indices after each epoch if needed
        if self.shuffle:
            np.random.shuffle(self.indices)




# %%
def train_and_evaluate(
        class_names: list[str], epochs: int = 10, batch_size: int = 64, learning_rate: float = 0.0001, fine_tune_at: int = 100):
    # Create data loaders and get the class index mapping
    train_loader, test_loader, class_idx_mapping = create_data_loaders(class_names, batch_size=batch_size)

    # Create model
    model = create_classifier_keras(class_names, learning_rate=learning_rate, fine_tune_at=fine_tune_at)

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        mode='max'
    )

    model_checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )

    # Create data generators with class index mapping
    train_gen = DataGenerator(train_loader, class_idx_mapping)
    test_gen = DataGenerator(test_loader, class_idx_mapping)

    # Train model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint],
    )

    # Load best model
    model = keras.models.load_model('best_model.keras')

    # Evaluate model
    results = model.evaluate(
        test_gen,
        steps=len(test_gen)
    )
    print(f"\nTest accuracy: {results[1]:.4f}")

    return model, history


# %%
if __name__ == "__main__" and False:
    # Example 1: Big cats
    for fine_tune_at in [100, 50, 0]:
        for lr in [1e-4, 1e-3, 1e-5, 1e-2, 1e-6]:
            print(f"\nTraining classifier for big cats with lr={lr} and ft={fine_tune_at}...")
            model1, history1 = train_and_evaluate(["leopard", "lion"], learning_rate=lr, fine_tune_at=fine_tune_at)

    # Example 2: Furniture
    for fine_tune_at in [100, 50, 0]:
        for lr in [1e-4, 1e-3, 1e-5, 1e-2, 1e-6]:
            print(f"\nTraining classifier for furniture with lr={lr} and ft={fine_tune_at}...")
            model2, history2 = train_and_evaluate(
                ["bed", "chair", "couch", "table", "wardrobe"], learning_rate=lr, fine_tune_at=fine_tune_at
            )
