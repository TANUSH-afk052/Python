# 🌸 Iris Flower Classification Project

## 📌 Objective:
To build a simple Machine Learning model that classifies **Iris flowers** into 3 species:
- **Setosa**
- **Versicolor**
- **Virginica**

based on their petal and sepal measurements.

---

## 📊 Dataset Information:
- Dataset: [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- Features:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- Target: Flower species (Setosa, Versicolor, Virginica)

---

## 🧪 Code Overview:

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load the data
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## ✅ Classification Report:

```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

---

## 📈 Result:
- **Perfect classification (100%)** on the test set.
- The model is very effective for this dataset due to the clear separation between classes.

---

## 📌 Notes:
- The Iris dataset is often used for learning ML basics.
- You can visualize the dataset using seaborn or matplotlib.
- Try testing different models like `LogisticRegression`, `KNN`, or `SVM`.

---

## 🧠 Learnings:
- Supervised classification
- Train-test split
- Model evaluation using accuracy and F1-score
- Understanding scikit-learn ML workflow
