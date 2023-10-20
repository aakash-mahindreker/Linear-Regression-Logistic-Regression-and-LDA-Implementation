# Linear-Regression-Logistic-Regression-and-LDA-Implementation

## Iris Dataset:
<img width="809" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/8c665140-1cd8-4d33-83c1-c9025f36e02d">


## Linear Regression:
Linear Regression class is implemented in which there are following necessary functions:

● __init__ : To initialize the weight and bias.

● calculate_loss : Takes two input arguments “y_pred” and “y_val”. The loss is calculated using the formula: loss = (np.square(y_pred - y_val)).mean().

● mean_squared_error: To calculate score, calculated using
score = ((y_pred - y_test) ** 2).mean()

● fit: Takes arguments “X_train”, “y_train”, “epochs”, “learning_rate”, “regularization”,
“batch_size” and “val_split”.

● predict : To predict the value for given input using the weight and bias. Calculated as
follows: preds = np.dot(X_test, self.weights) + self.bias.

 
 Combinations Used as cases:
1. Case 1: sepal width vs petal width:
 In this case, the mean squared error is 1.2784744874824914. The losses vs epochs graph for Case 1: “sepal width vs petal width”. <img width="807" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/d25b19ad-65d2-4019-beef-34715cc61550">

2. Case 2: Sepal length vs Petal:
 In this case, the mean squared error is 4.2309774828907205. The losses vs epochs graph for Case 2: “Sepal length vs Petal”. <img width="810" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/181d9996-6cb4-4320-a10e-0d9461169606">


3. Case 3: Sepal length, Sepal width vs Petal Length:
 In this case, the mean squared error is 0.797179913006021. The losses vs epochs graph for Case 3: “Sepal length, Sepal width vs Petal Length”. <img width="813" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/85e9fb21-cbdf-4b2d-8658-24378417b16e">

4. Case 4: Sepal length, Sepal width vs Petal Width:
 In this case, the mean squared error is 1.1649302442602993. The losses vs epochs graph for Case 4: “Sepal length, Sepal width vs Petal Width”. <img width="817" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/20ad7efc-5721-436f-87e5-4afe78285e2c">


 5. Case 5: Sepal length, Petal length, Petal width vs Sepal Width:
 In this case, the mean squared error is 0.7921599867407249. The losses vs epochs graph for Case 5: “Sepal length, Petal width vs Sepal Width”. <img width="812" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/99c3f755-1a89-4581-b2f0-d8e743b6693f">

 
 Comparison of Cases Based on MSE: <img width="778" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/76d970fd-ed1d-436d-9acd-72b6defa8134">


 Observation:
The case 2: “Sepal length vs Petal” has greater MSE, and loss. Therefore, it is the most inefficient model.

## Logistic Regression:
Logistic Regression class is implemented in which there are following necessary functions:

● __init__ : To initialize the weight and bias.

● calculate_loss : Takes two input arguments “y_pred” and “y_val”. The lossiscalculated using the formula: 
loss = (np.square(y_pred - y_val)).mean().

● Sigmoid: Is the logistic function calculated 1 / (1 + np.exp(-z)).

● Calculate_cost: To calculate the cost as -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon)) where epsilon = 1e-5.

● fit: Takes arguments “X”, “y”, and fits the model by calculating costs on each step.

● predict : To predict the value for given input using the weight, bias and sigmoid.

Calculated as follows: preds = np.where(sigmoid(z), 1, 0) where z is calculated using np.dot(X, weights).
 
 Classification using Logistic Regression:
1. Case 1: Petal Length and Petal Width: <img width="631" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/d22dc94b-1268-4a01-baaa-b552674668f1">
 The Accuracy for case 1 using Logistic Regression is 0.4.
 
2. Case 2: Petal Length and Petal Width: <img width="579" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/caf85069-de1e-4b8d-a1b5-0a1a41fc82a2">
 The Accuracy of Case 2 using Logistic Regression.

3. Case 3: All features
The accuracy remains the same i.e 0.4.
 
 Comparison of Case1, Case2 and Case3:
Observation: <img width="573" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/60d301a1-6f7a-4d0f-9792-92ee0557098a">

Looks like all the models are similar.

## Linear Discriminant Analysis (LDA)
Linear Discriminant Analysis (LDA) class is implemented in which there are following necessary functions:

● __init__ : To initialize the weight and bias.

● fit: Takes arguments “X”, “y”, and fits the model by calculating costs on each step.

● predict : To predict the value for given input by calculating class scatter matrices Sb, Sw and gradient descent.

● _calculate_within_class_scatter_matrix : To calculate Sw as (class_samples - class_mean).T.dot(class_samples - class_mean).

● _calculate_between_class_scatter_matrix(self, X, y, class_means) : To calculate Sb as np.outer(class_mean - self.mean_overall, class_mean - self.mean_overall) * class_samples.shape[0].


1. Case 1: Petal Length and Petal Width:<img width="622" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/c266ddff-3011-483a-bbfc-8f199b1b16fd">
 The Accuracy for case 1 using Logistic Regression is 0.4. 

2. Case 2: Petal Length and Petal Width: <img width="640" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/5035b1dc-8c1e-45f5-bd73-9be90f6f58b6">

 The Accuracy of Case 2 using LDA.

3. Case 3: All features
The accuracy remains the same i.e 0.4.
 
Comparison of Case1, Case2 and Case3: <img width="705" alt="image" src="https://github.com/aakash-mahindreker/Linear-Regression-Logistic-Regression-and-LDA-Implementation/assets/70765660/6ffbe226-f8ee-4f36-a58d-567c1ea30860">

Observation:
Looks like all the models are similar.
