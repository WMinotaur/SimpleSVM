import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from svm_project.data import generate_training_data_binary
from svm_project.svm import svm_train_brute, plot_hyper_binary

def compare_binary():
    """
    Compares custom SVM implementation with Scikit-learn's SVM for binary classification.

    Generates binary training data, trains both models, and plots the decision boundaries
    and support vectors for visual comparison.
    """
    print("Generating binary data...")
    data = generate_training_data_binary(num=4)
    X = data[:, :2]
    y = data[:, 2]

    print("Training Custom SVM...")
    w, b, _ = svm_train_brute(data)

    print("Training Scikit-learn SVM...")
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Custom SVM")
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    
    line = np.linspace(x_min, x_max)
    if w[1] != 0:
        plt.plot(line, (-w[0] * line - b) / w[1], 'k-')
    else:
        plt.axvline(x=b, color='k')
    
    for item in data:
        if item[-1] == 1:
            plt.plot(item[0], item[1], 'b+')
        else:
            plt.plot(item[0], item[1], 'ro')
    
    plt.axis([x_min, x_max, y_min, y_max])

    plt.subplot(1, 2, 2)
    plt.title("Scikit-learn SVM")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)
    
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    plt.show()



if __name__ == "__main__":
    compare_binary()
