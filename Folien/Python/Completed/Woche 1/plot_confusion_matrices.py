# %% tags=["subslide", "keep"]
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


# %% tags=["keep"]
def plot_confusion_matrices(clf, x_train, x_test, y_train, y_test, normalize=None):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 9))
    fig.tight_layout()
    fig.suptitle(f"Confusion Matrices (Normalize={normalize})")
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        clf.predict(x_test),
        normalize=normalize,
        ax=ax1,
        colorbar=False,
        values_format=".3f" if normalize else None,
    )
    ax1.set_title("Test Data")
    ConfusionMatrixDisplay.from_predictions(
        y_train,
        clf.predict(x_train),
        normalize=normalize,
        ax=ax2,
        colorbar=False,
        values_format=".3f" if normalize else None,
    )
    ax2.set_title("Training Data")
    plt.show()
