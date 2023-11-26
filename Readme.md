# Auto Logistic Regression

## Overview
This project implements an Auto Logistic Regression framework for easy model training, evaluation, and prediction. The framework includes two main classes: `AutoLogisticRegression` and `AutoPreprocessor`. 

### `AutoLogisticRegression` Class
The `AutoLogisticRegression` class is designed to automate the process of logistic regression model training and evaluation. Key functionalities include:
- **Initialization**: Accepts the path to the training dataset (`data_path`), the target column name (`target_column`), the output path for the model and predictions (`output_path`), and an optional parameter for the number of folds in cross-validation (`num_folds`).
- **Training**: Uses logistic regression with cross-validation to train a model on the provided dataset. The best model is saved as a pickle file for future use.
- **Prediction**: Given a new dataset, the trained model can be used to make predictions. The predictions are saved to a CSV file at the specified output path.

### `AutoPreprocessor` Class
The `AutoPreprocessor` class handles the preprocessing steps required before training or making predictions with the logistic regression model. Key functionalities include:
- **Initialization**: Accepts optional parameters for specifying categorical columns (`categorical_columns`), numeric columns (`numeric_columns`), and whether to perform oversampling (`oversample`).
- **Fit and Transform**: Fits an imputer, encoder (for categorical columns), and scaler (for numeric columns) on the provided data. The transformations are then applied to the data.
- **Oversampling**: Optionally applies oversampling to balance the class distribution in the training data.
- **Split Data**: Splits the data into training and testing sets.
## Installation
```python
pip install autolr
```
## Usage
1. **Importing**:
```python
from autolr import AutoLogisticRegression
```
2. **Initialization**: Create an instance of the `AutoLogisticRegression` class by providing the path to the training dataset, the target column name, and the output path for the model and predictions. Optionally, you can specify the number of folds for cross-validation.

```python
auto_lr = AutoLogisticRegression(data_path='path/to/training_data.csv', target_column='target', output_path='output', num_folds=5)
```
3. **Training Model**: After training, the model is automatically evaluated using metrics such as accuracy, classification report, and confusion matrix. The feature importance is also displayed.

```python
# Training the model and evaluating it
auto_lr.train()
```
4. **Prediction**: After training, you can use the trained model to make predictions on new data by providing the path to the new dataset.
```python
predictions = auto_lr.predict(data_path='path/to/test_data.csv')
```
## Dependencies
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [pickle](https://docs.python.org/3/library/pickle.html)
- [os](https://docs.python.org/3/library/os.html)

## Notes
- Ensure that the required dependencies are installed before running the code.
- The `AutoPreprocessor` class is used internally for data preprocessing and is not intended for standalone use.
- Customize and extend this framework based on your specific needs.
- If you encounter any issues or have suggestions for improvements, please let us know.
