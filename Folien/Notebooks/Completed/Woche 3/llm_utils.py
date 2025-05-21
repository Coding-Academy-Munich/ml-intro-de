import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns


def plot_neuron_2d(neuron):
    # Create a grid of input points
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # Prepare input for the neuron
    xy = torch.tensor(np.column_stack((X.ravel(), Y.ravel())), dtype=torch.float32)

    # Pass the input through the neuron
    with torch.no_grad():
        Z = neuron(xy).numpy().reshape(X.shape)

    # Create the 3D plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add a color bar
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Activation')
    ax.set_title('2D Neuron Activation')

    # Enable rotation
    ax.mouse_init()

    # Adjust the view angle for better initial visualization
    ax.view_init(elev=20, azim=45)

    # Tight layout to ensure everything fits
    plt.tight_layout()

    # Show the plot
    plt.show()


# %%
def evaluate_model(model, test_dataset, batch_size=100):
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    y_pred = model.predict(test_dataset)
    y_true = torch.cat([y for _, y in test_loader], dim=0).numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    return accuracy, precision, recall, f1
