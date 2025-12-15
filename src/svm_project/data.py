import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

def generate_training_data_binary(num=1):
    """
    Generates binary classification training data using the Breast Cancer dataset.

    Reduces the dataset to 2 dimensions using PCA and maps labels to {1, -1}.
    Subsamples the data to a smaller size for performance.

    Args:
        num (int): Number of samples to generate (if > 10), otherwise defaults to 50.

    Returns:
        np.ndarray: Combined array of shape (N, 3) containing [x, y, label].
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Scale data for better SVM performance/meaningful distances
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    y_mapped = np.where(y == 0, 1, -1)
    
    n_samples = num if num > 10 else 50
    
    X_res, y_res = resample(X_pca, y_mapped, n_samples=n_samples, random_state=42, replace=False)
    
    data_combined = np.column_stack((X_res, y_res))
    return data_combined



def plot_training_data_binary(data):
    """
    Plots binary training data.

    Args:
        data (np.ndarray): Data array of shape (N, 3) where columns are [x, y, label].
                           Labels should be 1 (Malignant) or -1 (Benign).
    """
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+', alpha=0.6, label='Malignant')
        else:
            plt.plot(item[0], item[1], 'ro', alpha=0.6, label='Benign')
            
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    plt.axis([x_min, x_max, y_min, y_max])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Breast Cancer (PCA 2D)")
    plt.show()


