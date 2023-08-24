#%%
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit

#%% Fix random seed
rng = np.random.default_rng(0)


# (b) Implementation of bagging
class BaggingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, B=10):
        self.B = B
        self.estimator = estimator
        self.estimators = []

    def fit(self, X, y):
        # Initialize list of estimators
        self.estimators = [clone(self.estimator) for b in range(self.B)]

        for estimator in self.estimators:
            # Create bootstrap dataset of same size as input data, by drawing random indices with replacement
            y_len = len(y)
            sample_indices = rng.integers(0, y_len, y_len)
            # Fit an estimator to the bootstrap sample
            estimator.fit(X[sample_indices], y[sample_indices])

    def predict_proba(self, X):
        # Probability of a given label is the average probability assigned by all estimators
        return np.average(np.array([estimator.predict_proba(X) for estimator in self.estimators]), 0)

    def predict(self, X):
        # Predicted label is the one with the highest probability (this way voting is avoided)
        return np.argmax(self.predict_proba(X), 1)


# (c) Implementation of AdaBoost
class AdaBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, M=10):
        self.M = M
        self.estimator = estimator
        self.estimators = []
        self.a = []

    def fit(self, X, y):
        # Initialize list of estimators
        self.estimators = [clone(self.estimator) for m in range(self.M)]

        # Initialize weights of models and samples
        self.a = np.zeros((self.M, 1))
        N = len(y)
        w = np.full(N, 1/N)

        for m, estimator in enumerate(self.estimators):
            # Fit estimator and get predictions
            estimator.fit(X, y, sample_weight=w)
            y_pred = estimator.predict(X)

            # Calculate error and weights
            err_m = np.sum(w * (y != y_pred)) / np.sum(w)
            self.a[m] = np.log((1 - err_m) / err_m)
            w = w * np.exp(self.a[m] * (y != y_pred))

    def predict(self, X):
        # Labels are {0,1}, but algorithm requires {-1,1},
        # below we make use of the fact that {0,1} * 2 - 1 = {-1,1}
        return (np.sign(np.sum(self.a *
                               np.array([estimator.predict(X) * 2 - 1 for estimator in self.estimators]),
                               axis=0)) + 1) / 2


data = np.genfromtxt("spambase.data", delimiter=',')
X, y = np.array_split(data, [data.shape[1] - 1], axis=1)
y = y.reshape(-1, ).astype(int)

# (a) Splitting data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# This is a workaround to enable the use of GridSearchCV with a fixed validation set
# Size of the validation set is 1/9 * 0.9 = 0.1
validation_fold = rng.choice([-1, 0], size=len(y_train_val), p=[8/9, 1/9])
pds = PredefinedSplit(validation_fold)

# Parameters to be searched
# Random state is set to 0 for all to ensure reproducibility

param_grid = {
    "random_state": [0],
    "criterion": ["gini", "entropy"],
    "splitter": ["best", "random"],
    "max_depth": [None, *range(1, 30)],
    "max_features": [None, "sqrt", "log2"]
}

bagging_param_grid = {
    "estimator__random_state": [0],
    "estimator__criterion": [
        "gini",
        "entropy"
    ],
    "estimator__splitter": [
        "best",
        "random"
    ],
    "estimator__max_depth": range(1, 15),
    "estimator__max_features": [
        None,
        "sqrt",
        "log2"
    ],
    "B": [*range(1, 20), *range(20, 200, 20)]
}

boosting_param_grid = {
    "estimator__random_state": [0],
    "estimator__criterion": [
        "gini",
        "entropy"
    ],
    "estimator__splitter": [
        "best",
        "random"
    ],
    "estimator__max_depth": range(1, 11),
    "estimator__max_features": [
        None,
        "sqrt",
        "log2"
    ],
    "M": [*range(1, 20), *range(20, 200, 20)]
}

n_jobs = -1
verbose = 0
return_train_score = True

#%% Finding best parameters

tree_clf = GridSearchCV(DecisionTreeClassifier(), param_grid,
                        n_jobs=n_jobs, cv=pds,verbose=verbose, return_train_score=return_train_score)
tree_clf.fit(X_train_val, y_train_val)
print("Decision Tree results:")
tree_cv_results = pd.DataFrame(tree_clf.cv_results_)
print(f"Best params: {tree_cv_results.iloc[tree_clf.best_index_]['params']}")
print(f"Train score: {tree_cv_results.iloc[tree_clf.best_index_]['mean_train_score']}")
print(f"Validation score: {tree_cv_results.iloc[tree_clf.best_index_]['mean_test_score']}")
print(f"Test score: {tree_clf.score(X_test, y_test)}")

bagging_clf = GridSearchCV(BaggingModel(DecisionTreeClassifier()), bagging_param_grid,
                           n_jobs=n_jobs, cv=pds, verbose=verbose, return_train_score=return_train_score)
bagging_clf.fit(X_train_val, y_train_val)
print("Bagging results:")
bagging_cv_results = pd.DataFrame(bagging_clf.cv_results_)
print(f"Best params: {bagging_cv_results.iloc[bagging_clf.best_index_]['params']}")
print(f"Train score: {bagging_cv_results.iloc[bagging_clf.best_index_]['mean_train_score']}")
print(f"Validation score: {bagging_cv_results.iloc[bagging_clf.best_index_]['mean_test_score']}")
print(f"Test score: {bagging_clf.score(X_test, y_test)}")

adaboost_clf = GridSearchCV(AdaBoost(DecisionTreeClassifier()), boosting_param_grid,
                            n_jobs=n_jobs, cv=pds, verbose=verbose, return_train_score=return_train_score)
adaboost_clf.fit(X_train_val, y_train_val)
print("AdaBoost results:")
adaboost_cv_results = pd.DataFrame(adaboost_clf.cv_results_)
print(f"Best params: {adaboost_cv_results.iloc[adaboost_clf.best_index_]['params']}")
print(f"Train score: {adaboost_cv_results.iloc[adaboost_clf.best_index_]['mean_train_score']}")
print(f"Validation score: {adaboost_cv_results.iloc[adaboost_clf.best_index_]['mean_test_score']}")
print(f"Test score: {adaboost_clf.score(X_test, y_test)}")
