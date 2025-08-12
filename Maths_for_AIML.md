
# Math Foundations for Machine Learning

Understanding the math behind Machine Learning (ML) helps you:

- Choose the right algorithms  
- Debug and optimize models  
- Understand why models behave the way they do

---

## 1. Linear Algebra

### 1.1 Vectors  
An ordered list of numbers used to represent features (inputs) or weights in ML models.  
Example:  
`x = [3, 5, 2]`

### 1.2 Matrices  
2D arrays of numbers used to store datasets or transformation rules.  
Example: Dataset with 3 samples and 2 features:  
```

X = \[\[1, 2],
\[3, 4],
\[5, 6]]

```

### 1.3 Matrix Operations  
- **Addition:** Element-wise addition  
- **Multiplication:** Dot product  
- **Transpose:** Flip rows and columns  
- **Inverse:** “Undo” matrix transformation  

### 1.4 Why It Matters in ML  
- Datasets → matrices  
- Model parameters → vectors/matrices  
- Predictions:  
`y_hat = X * W + b`

---

## 2. Probability & Statistics

### 2.1 Probability Basics  
- Probability P(A) = likelihood of event A (between 0 and 1)  
- Joint Probability P(A, B) — both A and B happen  
- Conditional Probability P(A|B) — A happens given B  

### 2.2 Distributions  
- Normal Distribution: bell-shaped, common in nature  
- Bernoulli Distribution: binary outcomes (0 or 1)  
- Binomial Distribution: number of successes in fixed trials  
- Poisson Distribution: number of events in a fixed interval  

### 2.3 In ML  
- Naive Bayes classifier uses conditional probabilities  
- Logistic regression predicts class probabilities  

---

## 3. Calculus

### 3.1 Derivatives  
Measures how a function changes with respect to its inputs. Example: slope of a curve.

### 3.2 Gradient  
Vector of partial derivatives that shows how to change each parameter to improve the model.

### 3.3 Gradient Descent  
Optimization algorithm to minimize loss function:  
```

theta := theta - alpha \* gradient\_J(theta)

```
Where:  
- theta = parameters  
- alpha = learning rate  
- J(theta) = cost/loss function  

### 3.4 Why It Matters  
Training ML models means minimizing a loss function using calculus.

---

## 4. Norms and Distances

### 4.1 Euclidean Distance  
Straight-line distance between points:  
```

d = sqrt((x1 - y1)^2 + (x2 - y2)^2)

```

### 4.2 Manhattan Distance  
Distance along axes:  
```

d = |x1 - y1| + |x2 - y2|

```

### 4.3 Cosine Similarity  
Measures angle between vectors:  
```

cos(theta) = (x · y) / (|x| \* |y|)

```

---

## 5. Optimization

### 5.1 Loss Functions  
- Regression: Mean Squared Error (MSE)  
```

MSE = (1/n) \* sum((y\_i - y\_hat\_i)^2)

```
- Classification: Cross-Entropy Loss  
```

L = -sum(y \* log(y\_hat))

```

### 5.2 Gradient Descent Variants  
- Batch Gradient Descent — whole dataset each step  
- Stochastic Gradient Descent (SGD) — one sample at a time  
- Mini-batch Gradient Descent — small batches  

---

## Resources

- [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra)  
- [3Blue1Brown: Essence of Linear Algebra (YouTube)](https://www.youtube.com/watch?v=kjBOesZCoqc)  
- [StatQuest: Statistics & Probability](https://www.youtube.com/user/joshstarmer)  
- [Calculus for Machine Learning (Free PDF)](https://www.manning.com/books/calculus-for-machine-learning)  

---

*Mastering these math concepts will make ML much easier to understand, debug, and improve.*

---

