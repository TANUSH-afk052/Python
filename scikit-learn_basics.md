
# 📘 Machine Learning & AI Basics with scikit-learn

## 1. 🤖 What is AI, ML, and DL?

- **Artificial Intelligence (AI)**: Broad field aiming to create machines that can perform tasks requiring human-like intelligence (reasoning, problem-solving, decision-making).
- **Machine Learning (ML)**: Subset of AI — systems learn from data without being explicitly programmed.
- **Deep Learning (DL)**: Subset of ML — uses multi-layered neural networks to learn complex patterns.

---

## 2. 📂 Types of Machine Learning

| Type | Description | Examples |
|------|-------------|----------|
| **Supervised Learning** | Model learns from labeled data (features + target). | Predicting house prices, spam detection |
| **Unsupervised Learning** | Model finds patterns in unlabeled data. | Customer segmentation, clustering |
| **Reinforcement Learning** | Agent learns by interacting with environment and receiving rewards/penalties. | Game-playing AI, robotics |

---

## 3. 📊 Common Terms

- **Feature (X)**: Input variable.
- **Label/Target (y)**: Output variable we want to predict.
- **Model**: Algorithm trained on data to make predictions.
- **Training**: Feeding data to a model to learn patterns.
- **Testing**: Evaluating the model on unseen data.
- **Overfitting**: Model performs well on training data but poorly on new data.
- **Underfitting**: Model is too simple, performs poorly on both training and new data.
- **Accuracy**: % of correct predictions.
- **Precision, Recall, F1-score**: Metrics for classification performance.

---

## 4. ⚙️ scikit-learn Basics

`scikit-learn` is a popular Python library for ML.  
Install it:
```bash
pip install scikit-learn
````

---

### 4.1 🚀 Basic Workflow in scikit-learn

```python
# Step 1: Import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Prepare data (X = features, y = labels)
X = [[1], [2], [3], [4], [5]]
y = [0, 0, 1, 1, 1]

# Step 3: Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose and train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

### 4.2 📈 Common Algorithms in scikit-learn

| Task                     | Algorithm           | Import                                                |
| ------------------------ | ------------------- | ----------------------------------------------------- |
| Classification           | Logistic Regression | `from sklearn.linear_model import LogisticRegression` |
| Classification           | Random Forest       | `from sklearn.ensemble import RandomForestClassifier` |
| Regression               | Linear Regression   | `from sklearn.linear_model import LinearRegression`   |
| Clustering               | KMeans              | `from sklearn.cluster import KMeans`                  |
| Dimensionality Reduction | PCA                 | `from sklearn.decomposition import PCA`               |

---

### 4.3 🔍 Example: Classification with scikit-learn

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred, target_names=data.target_names))
```

---

## 5. 📌 Tips for Learning ML

* Understand **data preprocessing** (scaling, encoding, missing values).
* Learn **model evaluation metrics**.
* Experiment with **different algorithms**.
* Use **cross-validation** to get more reliable results.
* Practice with datasets from [Kaggle](https://www.kaggle.com/) or `sklearn.datasets`.

---

## 6. 📚 Resources

* [scikit-learn Documentation](https://scikit-learn.org/stable/user_guide.html)
* [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
* [Kaggle Learn Micro-Courses](https://www.kaggle.com/learn)
* [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

---
