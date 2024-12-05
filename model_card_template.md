# Model Card

## Model Details
This model predicts whether an individual's income exceeds $50,000 based on census data. The model is built using a Random Forest Classifier with 100 estimators. It uses categorical and continuous features from the census dataset to make predictions. The model was trained and evaluated as part of a pipeline designed to process data, train the model, and evaluate its performance on both overall metrics and slice-specific metrics.

### Framework
- **Framework**: scikit-learn
- **Algorithm**: Random Forest Classifier
- **Version**: Python 3.10, scikit-learn 1.5.1

---

## Intended Use
This model is intended to classify individuals into two income categories:
1. `<=50K`
2. `>50K`

### Applications
- Can be used for demographic analysis and income prediction.
- Provides insights into income distribution across categorical features.

### Limitations
- Should not be used for making high-stakes financial decisions without rigorous validation.
- Model performance might degrade when applied to populations significantly different from the training dataset.

---

## Training Data
The model was trained using a subset of the UCI Adult Census dataset. The dataset contains features like age, education, marital status, occupation, race, and sex.

- **Training Size**: 26,048 samples

### Features
- **Categorical**: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- **Continuous**: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week

---

## Evaluation Data
The model was evaluated on a test set created using a 20% split of the original dataset.

- **Test Size**: 6,513 samples
- The evaluation includes overall metrics and slice-specific metrics across categorical features.

---

## Metrics
The model's performance was evaluated using the following metrics:

### Overall Metrics
- **Precision**: 0.7419
- **Recall**: 0.6384
- **F1-Score**: 0.6863

### Slice-Specific Metrics
Metrics were computed for slices based on unique values of each categorical feature. For example:
- **workclass = Private**: Precision = 0.74, Recall = 0.64, F1-Score = 0.69
- **education = Bachelors**: Precision = 0.75, Recall = 0.73, F1-Score = 0.74

Full details can be found in the `slice_output.txt`.

---

## Ethical Considerations
- The model may inherit biases present in the dataset, such as historical disparities in income across gender or race.
- Sensitive features like race and sex should be carefully analyzed for fairness concerns.
- Predictions should not be used to discriminate or harm individuals or groups.

### Mitigation
- Regular monitoring of model predictions on sensitive slices (e.g., race, sex).
- Use fairness-aware machine learning techniques if deployed in high-impact applications.

---

## Caveats and Recommendations

### Caveats
- Sparse data in some categories (e.g., `education = 1st-4th`) may lead to unreliable predictions.
- Performance on categories with fewer samples might not generalize well.

### Recommendations
- Collect more diverse training data to improve generalization.
- Perform hyperparameter tuning and feature engineering for better accuracy.
- Periodically evaluate the model on new data to ensure continued reliability.
