# Author: Mehdi Abbassi <mehdi.abbassi@studenti.unimi.it>
#
# License: GNU/GPL 3

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from AdaBoost_binary import AdaBoost
from multiprocessing import Pool

# Download the Cover Type Dataset for multiclass classification.
#
# Implement AdaBoost from scratch and run it using decision stumps
# (binary classification rules based on single features) as base classifiers to
#
# train seven binary classifiers, one for each of the seven classes
# (one-vs-all encoding).
#
# Use external cross-validation to evaluate the multiclass classification
# performance (zero-one loss) for different values of the number T of AdaBoost
# rounds.
#
# In order to use the seven binary classifiers to make multiclass
# predictions, use the fact that binary classifiers trained by AdaBoost have
# the form h(x)=sgn(g(x)) and predict using argmax_i g_i(x) where g_i
# corresponds to the binary classifier for class.

T = 200  # number of base classifiers
K = 5  # number of folds for cross-validation
gamma = 16
beta = 0.6

dataset = pd.read_csv('covtype.csv')
X, y = dataset.iloc[:, :-1].to_numpy(), dataset.iloc[:, -1].to_numpy()
Y = np.unique(y)


def cross_validation(
        X_train, X_test,
        y_train, y_test,
        Y, results, T,
        random_state):
    """
    cross-validation
    """
    training_predicts = []
    testing_predicts = []
    complexity = []
    for label in Y:
        y_train_ = np.ones(y_train.shape[0])
        y_train_[y_train != label] = -1

        y_test_ = np.ones(y_test.shape[0])
        y_test_[y_test != label] = -1

        Classifier = AdaBoost(
                        n_classifiers=T,
                        gamma=gamma,
                        beta=beta,
                        random_state=random_state)
        Classifier.fit(X_train, y_train_)

        training_predicts.append(Classifier.predict(X_train))
        testing_predicts.append(Classifier.predict(X_test))
        complexity.append(Classifier.complexity)

    train_predict = Y[np.argmax(training_predicts, axis=0)]
    training_accuracy = np.sum(y_train == train_predict) / len(train_predict)

    test_predict = Y[np.argmax(testing_predicts, axis=0)]
    testing_accuracy = np.sum(y_test == test_predict) / len(test_predict)

    results.append(
        (
            T, gamma, beta,
            training_accuracy, testing_accuracy, complexity))

    return results


def seeding(seed):
    print(f'Setting seed to {seed}.')
    results = []
    random_state = seed
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
    skf.get_n_splits(X, y)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        results = cross_validation(
                    X_train, X_test,
                    y_train, y_test,
                    Y, results, T,
                    random_state)
    file_name = f'{seed}_{gamma}_{beta}.json'
    with open(file_name, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    pool = Pool()
    pool.map(seeding, [0, 1, 2, 3])
