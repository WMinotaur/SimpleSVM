import numpy as np
import matplotlib.pyplot as plt
from .data import *



def svm_train_brute(training_data, C=50):
    """
    Trains a linear SVM using a brute-force approach to find the optimal hyperplane
    by minimizing a Cost Function (Soft Margin).

    Cost = 0.5 * (1 / margin)^2 + C * TotalError

    Args:
        training_data (np.ndarray): Training data of shape (N, 3) with [x, y, label].
        C (float): Penalty parameter. Higher C means more penalty for misclassification.

    Returns:
        tuple: (w, b, support_vectors) where w is the weight vector, b is the bias,
               and support_vectors is a list of support vector points.
    """
    training_data = np.asarray(training_data)

    positive = training_data[training_data[:, 2] == 1]
    negative = training_data[training_data[:, 2] == -1]

    min_cost = float('inf')
    s_last, w_last, b_last = None, None, None

    def update_best(w_cand, b_cand, sv_cand):
        nonlocal min_cost, s_last, w_last, b_last
        current_cost = compute_cost(training_data, w_cand, b_cand, C)
        
        if current_cost < min_cost:
            min_cost = current_cost
            s_last = sv_cand
            w_last = w_cand
            b_last = b_cand

    for pos in positive:
        for neg in negative:
            mid_point = (pos[0:2] + neg[0:2]) / 2
            w = np.array(pos[:-1] - neg[:-1])
            w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
            b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])
            
            update_best(w, b, np.array([pos, neg]))

    for pos in positive:
        for pos1 in positive:
            for neg in negative:
                if (pos[0] != pos1[0]) and (pos[1] != pos1[1]):
                    separator = (pos1 - pos)[:2]
                    ws = separator / np.sqrt(separator.dot(separator))
                    projection = np.append(pos[:2] + (np.dot(ws, (neg[:2] - pos[:2]))) * ws, [1])

                    mid_point = (projection[0:2] + neg[0:2]) / 2
                    w = np.array(projection[:-1] - neg[:-1])
                    w = w / np.sqrt(w.dot(w))
                    b = -1 * (w.dot(mid_point))

                    update_best(w, b, np.array([pos, pos1, neg]))

    for neg in negative:
        for neg1 in negative:
            for pos in positive:
                if neg[0] != neg1[0] and neg[1] != neg1[1]:
                    separator = (neg1[:2] - neg[:2])
                    ws = separator / np.sqrt(separator.dot(separator))
                    projection = np.append(neg[:2] + (np.dot(ws, (pos[:2] - neg[:2]))) * ws, [-1])
                    mid_point = (pos[0:2] + projection[0:2]) / 2
                    w = np.array(pos[:-1] - projection[:-1])
                    w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
                    b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])

                    update_best(w, b, np.array([pos, neg, neg1]))

    return w_last, b_last, s_last


def distance_point_to_hyperplane(pt, w, b):
    """
    Calculates the absolute distance from a point to the hyperplane defined by w and b.

    Args:
        pt (np.ndarray): Point coordinates [x, y].
        w (np.ndarray): Weight vector.
        b (float): Bias term.

    Returns:
        float: Absolute distance.
    """
    return np.abs(((pt[0] * w[0]) + (pt[1] * w[1]) + b) / (np.sqrt((w[0] * w[0]) + (w[1] * w[1]))))


def compute_cost(data, w, b, C=50):
    """
    Computes the Cost Function for the hyperplane.
    
    Cost = 0.5 * (1 / Margin)^2 + C * TotalError
    
    Where:
    - Margin is the min distance to correctly classified points.
    - TotalError is sum of distances of misclassified points.

    args:
        data (np.ndarray): Dataset.
        w (np.ndarray): Weight vector.
        b (float): Bias term.
        C (float): Penalty parameter.

    Returns:
        float: The computed Cost.
    """
    total_error = 0
    count_misclassified = 0
    min_dist = float('inf')
    
    for pt in data:
        dist = distance_point_to_hyperplane(pt[:-1], w, b)
        predicted = svm_test_brute(w, b, pt)
        
        if predicted != pt[2]:
            total_error += dist
            count_misclassified += 1
        else:
            if dist < min_dist:
                min_dist = dist
    final_total_error = total_error + (count_misclassified * min_dist)

    margin_cost = 0.5 * (1 / min_dist)**2
    penalty_cost = C * final_total_error
      
    return margin_cost + penalty_cost


def svm_test_brute(w, b, x):
    """
    Classifies a new point x using the trained SVM parameters w and b.

    Args:
        w (np.ndarray): Weight vector.
        b (float): Bias term.
        x (np.ndarray): Point to classify (can include label, but only coords used).

    Returns:
        int: Predicted label (1 or -1).
    """
    if np.dot(w, x[:-1]) + b > 0:
        return 1
    else:
        return -1





def plot_hyper_binary(w, b, data):
    """
    Plots the decision boundary and training data for binary classification.

    Args:
        w (np.ndarray): Weight vector.
        b (float): Bias term.
        data (np.ndarray): Training data.
    """
    line = np.linspace(-100, 100)
    if w[1] != 0:
        plt.plot(line, (-w[0] * line - b) / w[1])
    else:
        plt.axvline(x=b)
    plot_training_data_binary(data)



