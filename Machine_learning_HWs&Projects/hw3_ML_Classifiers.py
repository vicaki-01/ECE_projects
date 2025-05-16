import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data[:, [0, 2]]  # Use only sepal length and petal length for visualization
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Standardize features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolor='black')

    # Highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=100, label='test set')
        
        # Train Perceptron
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)


# Train Logistic Regression
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train_std, y_train)

# Train SVM with Linear Kernel
svm_linear = SVC(kernel='linear', C=1.0, random_state=1)
svm_linear.fit(X_train_std, y_train)

# Train SVM with RBF Kernel
svm_rbf = SVC(kernel='rbf', gamma=0.2, C=1.0, random_state=1)
svm_rbf.fit(X_train_std, y_train)

# Train Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=1)
tree.fit(X_train_std, y_train)

# Train Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=1)
forest.fit(X_train_std, y_train)

# Predict and evaluate
y_pred = ppn.predict(X_test_std)
print("Perceptron Accuracy:", accuracy_score(y_test, y_pred))
print("Perceptron Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot decision boundary
plot_decision_regions(X_train_std, y_train, classifier=ppn)
plt.title('Perceptron Decision Boundary')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Petal Length (standardized)')
plt.legend(loc='upper left')
plt.show()

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test_std)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("KNN Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot decision boundary
plot_decision_regions(X_train_std, y_train, classifier=knn)
plt.title('KNN Decision Boundary')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Petal Length (standardized)')
plt.legend(loc='upper left')
plt.show()


# Create a summary table
results = {
    'Classifier': ['Perceptron', 'Logistic Regression', 'SVM (Linear)', 'SVM (RBF)', 'Decision Tree', 'Random Forest', 'KNN'],
    'Accuracy': [accuracy_score(y_test, ppn.predict(X_test_std)),
                 accuracy_score(y_test, lr.predict(X_test_std)),
                 accuracy_score(y_test, svm_linear.predict(X_test_std)),
                 accuracy_score(y_test, svm_rbf.predict(X_test_std)),
                 accuracy_score(y_test, tree.predict(X_test_std)),
                 accuracy_score(y_test, forest.predict(X_test_std)),
                 accuracy_score(y_test, knn.predict(X_test_std))],
    'Misclassified': [(y_test != ppn.predict(X_test_std)).sum(),
                      (y_test != lr.predict(X_test_std)).sum(),
                      (y_test != svm_linear.predict(X_test_std)).sum(),
                      (y_test != svm_rbf.predict(X_test_std)).sum(),
                      (y_test != tree.predict(X_test_std)).sum(),
                      (y_test != forest.predict(X_test_std)).sum(),
                      (y_test != knn.predict(X_test_std)).sum()]
}

results_df = pd.DataFrame(results)
print(results_df)