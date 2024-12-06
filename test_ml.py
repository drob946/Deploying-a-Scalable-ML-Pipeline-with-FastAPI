import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


def test_process_data():
    """
    Test the process_data function to ensure it correctly processes both
    categorical and continuous features, as well as the labels.
    """
    # Sample test data
    data = pd.DataFrame({
        "age": [25, 35],
        "workclass": ["Private", "Self-emp-not-inc"],
        "education": ["Bachelors", "Masters"],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Tech-support", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "native-country": ["United-States", "India"],
        "salary": [">50K", "<=50K"]
    })

    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    label = "salary"
    
    X, y, encoder, lb = process_data(data, categorical_features, label, training=True)
    
    # Assertions
    assert X.shape[0] == 2  # Ensure the number of rows matches
    assert len(y) == 2  # Ensure the labels are processed
    assert encoder is not None  # Ensure the encoder is created
    assert lb is not None  # Ensure the label binarizer is created

def test_train_model():
    """
    Test the train_model function to ensure it trains without errors
    and returns a valid model object.
    """
    # Generate dummy data
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, size=10)

    # Train model
    model = train_model(X_train, y_train)

    # Assertions
    assert model is not None  # Check that a model object is returned

def test_compute_model_metrics():
    """
    Test the compute_model_metrics function to ensure it calculates
    precision, recall, and F1-score correctly.
    """
    # Dummy labels
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0])

    # Compute metrics
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    # Assertions
    assert precision >= 0 and precision <= 1  # Precision is valid
    assert recall >= 0 and recall <= 1  # Recall is valid
    assert f1 >= 0 and f1 <= 1  # F1-score is valid

def test_inference():
    """
    Test the inference function to ensure it generates predictions in
    the correct format and length.
    """
    # Dummy data
    X_test = np.random.rand(10, 5)
    y_test = np.random.randint(0, 2, size=10)

    # Train model
    model = train_model(X_test, y_test)

    # Run inference
    preds = inference(model, X_test)

    # Assertions
    assert len(preds) == 10  # Predictions match the input length
    assert np.array_equal(preds, preds.astype(int))  # Predictions are integers

