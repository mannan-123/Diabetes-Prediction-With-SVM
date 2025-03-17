# Diabetes Prediction Using SVM

## Overview

This project implements a **Support Vector Machine (SVM) classifier** to predict diabetes using the **PIMA Indian Diabetes Dataset**. The model is trained using **scikit-learn** and evaluates its performance on training and test data.

## Dataset

The dataset used is the **PIMA Indian Diabetes Dataset**, which contains health-related attributes such as glucose levels, BMI, and insulin levels to predict whether a person has diabetes.

## Installation & Requirements

Make sure you have Python installed along with the necessary libraries:

```bash
pip install numpy pandas scikit-learn
```

## Importing the Necessary Libraries

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
```

## Data Collection and Analysis

Load the dataset and explore its structure:

```python
dataset = pd.read_csv('/content/diabetes.csv')

# Display first 5 rows
dataset.head()

# Check dataset shape
dataset.shape

# Summary statistics
dataset.describe()

# Count of diabetic (1) and non-diabetic (0) cases
dataset['Outcome'].value_counts()

# Mean values of features grouped by outcome
dataset.groupby('Outcome').mean()
```

## Feature Selection

Separate the target variable (**Outcome**) from the feature set:

```python
Y = dataset['Outcome']
X = dataset.drop(columns='Outcome', axis=1)
```

## Data Standardization

Feature scaling is performed using `StandardScaler` to normalize the data:

```python
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
```

## Splitting the Dataset

The dataset is split into **training** and **testing** sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

## Model Training

We use a **Support Vector Classifier (SVC)** with a linear kernel:

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

## Model Evaluation

Evaluate the model's performance on training and test data:

```python
# Accuracy on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Training Data Accuracy: {training_data_accuracy:.4f}')

# Accuracy on test data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Test Data Accuracy: {testing_data_accuracy:.4f}')
```

## Making Predictions

Test the model on a **new sample**:

```python
model_input = [6,148,72,35,0,33.6,0.627,50]  # Example input
model_input = np.asarray(model_input).reshape(1, -1)
model_input = scaler.transform(model_input)

prediction = classifier.predict(model_input)
if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```

## Conclusion

- The **SVM classifier** is trained on the **PIMA Indian Diabetes Dataset**.
- **Data standardization** improves model performance.
- **Evaluation metrics** provide insight into model accuracy.
- The trained model can predict diabetes based on new input data.

## Future Improvements

- Try different **kernels** (e.g., RBF, polynomial) to improve accuracy.
- Perform **feature engineering** for better model performance.
- Implement **hyperparameter tuning** using Grid Search.

---

### 💡 Feel free to contribute to this project by improving the model or exploring different approaches!
