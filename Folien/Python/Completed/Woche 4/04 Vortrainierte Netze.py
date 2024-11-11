# %%
# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Vortrainierte Netze</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# - Trainieren von neuronalen Netzen ist aufwändig
# - Können wir Netze verwenden, die bereits trainiert wurden?
# - Potentielle Probleme:
#   - Eingabeformat?
#   - Wie weiß das Netz, was es tun soll?
#   - Ausgabeformat?

# %% [markdown]
#
# - Viele Aufgaben verarbeiten ähnliche Eingabedaten
#   - Bilder
#   - Videos
#   - Texte
#   - Audiosignale
# - Ausgabeformat wird von letzter Schicht bestimmt
#   - Hängt eng mit der Aufgabe zusammen
# - Was lernen die ersten Schichten?
#   - Können wir das Wissen für andere Aufgaben nutzen?

# %% [markdown]
#
# ### Schichten eines CNN
#
# <img src="img/Figure-21-058.png" style="float: center; width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# <img src="img/Figure-21-059.png" style="float: center; width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# - Die ersten Schichten lernen einfache Muster (Linien, Kanten, ...)
# - Tiefere Schichten lernen komplexere Muster (Augen, Nase, ...)

# %% [markdown]
#
# ## Vortrainierte Netze
#
# ### Idee
#
# - Trainiere ein Netz auf einer großen Menge an Daten
# - Für ein Problem, das viel Kenntnisse über die Eingabedaten erfordert
# - Nutze die ersten Schichten des Netzes für ein anderes Problem

# %% [markdown]
#
# ### Vorgehensweise: Transfer-Lernen (Transfer Learning)
#
# - Entferne die letzte Schicht des Netzes
# - Füge eine neue letzte Schicht hinzu
# - Trainiere das Netz für das neue Problem
# - (Oft werden die Gewichte der ersten Schichten eingefroren)

# %% [markdown]
#
# ### Vorteile
#
# - Schnelleres Training
# - Bessere Generalisierung
# - Weniger Daten benötigt

# %% [markdown]
#
# ### Beispiele für vortrainierte Netze
#
# - Modelle zur Bildklassifikation
#   - VGG, ResNet, Inception, EfficientNet, ...
# - Modelle zur Objekterkennung
#   - Faster R-CNN, YOLO, ...
# - Modelle zur Verarbeitung von Texten
#   - BERT, GPT, Llama, Qwen, ...
# - Modelle zur Generierung von Bildern
#   - DALL-E, Stable Diffusion, Flux
# - ...

# %% [markdown]
#
# ## Zero-Shot-Lernen
#
# - Modelle, die Texte verarbeiten:
#   - Beschreibe die Aufgabe als Text
#   - Potentiell mit Beispielen
# - Beispiele:
#   - Text-to-Text: GPT, Llama, Qwen
#   - Text-to-Image: DALL-E, Stable Diffusion, Flux

# %% [markdown]
#
# ## Wo finde ich vortrainierte Netze?
#
# - [Hugging Face](https://huggingface.co/models)
# - [Kaggle(https://www.kaggle.com/models)
# - [PyTorch Hub](https://pytorch.org/hub/)
# - ...

# %% [markdown]
#
# ## Beispiel: MNIST mit vortrainiertem Netz

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.resnet import ResNet18_Weights
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# %%
# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(2),  # Pad to make 32x32
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to 3 channels
])
train_dataset = torchvision.datasets.MNIST(root='data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='data', train=False, transform=transform, download=True)

# %%
# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %%
# Create a predefined network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(512, 10)
model = model.to(device)

# %%
# Show the number of trainable and fixed parameters
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_fixed_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f'Number of trainable parameters: {num_trainable_params}')
print(f'Number of fixed parameters: {num_fixed_params}')

# %%
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
# Training loop
num_epochs = 2
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average training loss for the epoch
    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # Evaluation phase
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_test_loss = test_loss / len(test_loader)
    test_losses.append(epoch_test_loss)

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_train_loss:.4f}, '
          f'Test Loss: {epoch_test_loss:.4f}, '
          f'Test Accuracy: {accuracy:.2f}%')

# %%
# Plot training and test losses
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Test Losses')

plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.tight_layout()
plt.show()

# %%
# Generate confusion matrix
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(10, 10))
disp.plot()
plt.title('Confusion Matrix')
plt.show()

# %%
# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# %%
# Visualize some test images with predictions
images = np.array(test_dataset.data)[:6]
labels = np.array(y_true)[:6]
preds = np.array(y_pred)[:6]

# %%
# Show images
fig = plt.figure(figsize=(12, 6))

for idx in np.arange(len(images)):
    ax = fig.add_subplot(2, 3, idx+1)
    img = images[idx]
    ax.imshow(img, cmap="binary")
    ax.set_title(f'Predicted: {predicted[idx]}, Actual: {labels[idx]}')
    ax.axis('off')

plt.show()

# %%
# Visualize images that were misclassified
misclassified = np.where(np.array(y_true) != np.array(y_pred))[0]
misclassified_images = np.array(test_dataset.data)[misclassified]
misclassified_labels = np.array(y_true)[misclassified]
misclassified_preds = np.array(y_pred)[misclassified]

fig = plt.figure(figsize=(12, 12))

for i in range(12):
    ax = fig.add_subplot(4, 3, i+1)
    img = misclassified_images[i]
    ax.imshow(img, cmap='binary')
    ax.set_title(f'Predicted: {misclassified_preds[i]}, Actual: {misclassified_labels[i]}')
    ax.axis('off')

# %%
# Visualize the filters of the first convolutional layer
model.eval()
conv1 = model.conv1.weight.detach().cpu().numpy()
conv1 = conv1 - conv1.min()
conv1 = conv1 / conv1.max()

fig = plt.figure(figsize=(12, 6))
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(conv1[i, 0], cmap='gray')
    ax.axis('off')

plt.show()

# %%
# Visualize the filters of the second convolutional layer
conv2 = model.layer1[0].conv1.weight.detach().cpu().numpy()
conv2 = conv2 - conv2.min()
conv2 = conv2 / conv2.max()

fig = plt.figure(figsize=(12, 6))
for i in range(64):
    ax = fig.add_subplot(8, 8, i+1)
    ax.imshow(conv2[i, 0], cmap='gray')
    ax.axis('off')

plt.show()

# %%
