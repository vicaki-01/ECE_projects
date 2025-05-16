import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from itertools import combinations
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        dim = X_train.shape[1]  # Number of features
        self.indices_ = tuple(range(dim))  # Initial feature indices
        self.subsets_ = [self.indices_]  # List to store feature subsets
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]  # List to store scores

        # Perform sequential backward selection
        while dim > self.k_features:
            scores = []
            subsets = []

            # Evaluate all possible subsets with one less feature
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            # Find the best subset
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            # Store the best score
            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]  # Best score for the final subset
        return self

    def transform(self, X):
        # Return the dataset with the selected features
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        # Train the estimator on the selected features and calculate the score
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score


# Example dataset with missing values
data = {
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [10, 11, 12, np.nan]
}
df = pd.DataFrame(data)

# Impute missing values using the mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Drop rows with missing values
df_dropped_rows = df.dropna(axis=0)

# Drop columns with missing values
df_dropped_cols = df.dropna(axis=1)

print("Original DataFrame:\n", df)
print("\nDataFrame after imputation:\n", df_imputed)
print("\nDataFrame after dropping rows with missing values:\n", df_dropped_rows)
print("\nDataFrame after dropping columns with missing values:\n", df_dropped_cols)



from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Example DataFrame with categorical data
df = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'blue'],
    'size': ['S', 'M', 'L', 'M'],
    'price': [10, 20, 30, 40]
})

# Label Encoding
label_encoder = LabelEncoder()
df_label_encoded = df.copy()
df_label_encoded['color'] = label_encoder.fit_transform(df['color'])

# One-Hot Encoding
one_hot_encoder = OneHotEncoder()
color_encoded = one_hot_encoder.fit_transform(df[['color']]).toarray()
df_one_hot_encoded = pd.concat([df.drop('color', axis=1), pd.DataFrame(color_encoded)], axis=1)

print("Original DataFrame:\n", df)
print("\nDataFrame after Label Encoding:\n", df_label_encoded)
print("\nDataFrame after One-Hot Encoding:\n", df_one_hot_encoded)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Example dataset
X = np.array([[1, 2], [3, 4], [5, 6]])

# Standardization
scaler_std = StandardScaler()
X_std = scaler_std.fit_transform(X)

# Normalization
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

print("Original Data:\n", X)
print("\nStandardized Data:\n", X_std)
print("\nNormalized Data:\n", X_minmax)

from sklearn.ensemble import RandomForestClassifier

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

# Split the dataset into features (X) and labels (y)
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Assuming X_train_std and y_train are already defined
knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# Plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# Optimal number of features
optimal_num_features = k_feat[np.argmax(sbs.scores_)]
print("Optimal number of features:", optimal_num_features)


# Train a RandomForestClassifier
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_

feat_labels = df_wine.columns[1:]
# Plot feature importance
indices = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# Top 3 features
top_3_features = feat_labels[indices[:3]]
print("Top 3 features:", top_3_features)


from sklearn.linear_model import LogisticRegression

# Train a LogisticRegression model with L1 regularization
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
lr.fit(X_train_std, y_train)

# Visualize the feature weights
plt.figure(figsize=(10, 6))
plt.plot(range(X_train_std.shape[1]), lr.coef_.T, 'o')
plt.xticks(range(X_train_std.shape[1]), feat_labels, rotation=90)
plt.hlines(0, 0, X_train_std.shape[1])
plt.xlabel('Feature')
plt.ylabel('Coefficient')
plt.title('Feature Weights with L1 Regularization')
plt.show()