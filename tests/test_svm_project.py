import numpy as np
import pytest

from svm_project.data import generate_training_data_binary
from svm_project.svm import (
    svm_train_brute,
    svm_test_brute,
    compute_cost,
    distance_point_to_hyperplane
)

# ----------------------------
# FIXTURES
# ----------------------------

@pytest.fixture
def simple_separable_data():
    """
    Prosty, liniowo separowalny zbiór danych
    """
    return np.array([
        [2, 2,  1],
        [3, 3,  1],
        [-2, -2, -1],
        [-3, -3, -1],
    ])


@pytest.fixture
def generated_data():
    """
    Dane wygenerowane z Breast Cancer (PCA 2D)
    """
    return generate_training_data_binary(num=30)


# ----------------------------
# TESTY DATA
# ----------------------------

def test_generate_training_data_shape(generated_data):
    assert isinstance(generated_data, np.ndarray)
    assert generated_data.shape[1] == 3  # x, y, label
    assert generated_data.shape[0] == 30


def test_generate_training_data_labels(generated_data):
    labels = generated_data[:, 2]
    assert set(labels).issubset({1, -1})


def test_generate_training_data_default_size():
    data = generate_training_data_binary(num=5)
    assert data.shape[0] == 50


# ----------------------------
# TESTY GEOMETRII
# ----------------------------

def test_distance_point_to_hyperplane():
    w = np.array([1.0, 0.0])
    b = 0.0
    pt = np.array([3.0, 0.0])

    dist = distance_point_to_hyperplane(pt, w, b)

    assert dist == pytest.approx(3.0)


# ----------------------------
# TESTY KLASYFIKACJI
# ----------------------------

def test_svm_test_brute_prediction():
    w = np.array([1.0, 1.0])
    b = 0.0

    pt_pos = np.array([2, 2, 1])
    pt_neg = np.array([-2, -2, -1])

    assert svm_test_brute(w, b, pt_pos) == 1
    assert svm_test_brute(w, b, pt_neg) == -1


# ----------------------------
# TESTY TRENINGU
# ----------------------------

def test_svm_train_brute_returns_valid_params(simple_separable_data):
    w, b, sv = svm_train_brute(simple_separable_data, C=10)

    assert w is not None
    assert isinstance(w, np.ndarray)
    assert w.shape == (2,)
    assert isinstance(b, float)
    assert sv is not None
    assert len(sv) >= 2


def test_svm_train_brute_classifies_training_data(simple_separable_data):
    w, b, _ = svm_train_brute(simple_separable_data, C=10)

    correct = 0
    for pt in simple_separable_data:
        if svm_test_brute(w, b, pt) == pt[2]:
            correct += 1

    accuracy = correct / len(simple_separable_data)
    assert accuracy == 1.0


# ----------------------------
# TESTY FUNKCJI KOSZTU
# ----------------------------

def test_compute_cost_positive(simple_separable_data):
    w = np.array([1.0, 1.0])
    b = 0.0

    cost = compute_cost(simple_separable_data, w, b, C=10)

    assert cost > 0


def test_compute_cost_penalty_increases(simple_separable_data):
    w = np.array([1.0, 1.0])
    b = 0.0

    cost_low_C = compute_cost(simple_separable_data, w, b, C=1)
    cost_high_C = compute_cost(simple_separable_data, w, b, C=100)

    assert cost_high_C >= cost_low_C


# ----------------------------
# TEST INTEGRACYJNY (REAL DATA)
# ----------------------------

def test_training_accuracy_on_generated_data(generated_data):
    w, b, _ = svm_train_brute(generated_data, C=50)

    correct = 0
    for pt in generated_data:
        if svm_test_brute(w, b, pt) == pt[2]:
            correct += 1

    accuracy = correct / len(generated_data)

    assert accuracy > 0.7
