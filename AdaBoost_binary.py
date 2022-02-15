# Author: Mehdi Abbassi <mehdi.abbassi@studenti.unimi.it>
#
# License: GNU/GPL 3

import numpy as np


class DecisionStump:
    """
    Base Classifier
    """

    def __init__(self, *, gamma, complexity):
        """
        setting attributes
        """
        self.feature = None
        self.threshold = None
        self.alpha = None
        self.gamma = gamma
        self.complexity = complexity

    def fit(self, X, y, weights, n_samples, thresholds_set):
        """
        constructs a weak base classifier
        """
        # non-greedy search to find useful threshold and feature
        for feature in thresholds_set.keys():

            self.complexity += 1  # add one to the number of calculations

            X_column = X[:, feature]

            thresholds = thresholds_set[feature]

            threshold = np.random.choice(thresholds)

            y_predict = np.ones(n_samples)
            y_predict[X_column < threshold] = -1

            incorrect = y_predict != y

            # error fraction
            error = np.mean(
                np.average(
                    incorrect,
                    weights=weights,
                    axis=0
                    )
                )

            # Stop if classification is better than random guessing
            # In the case of binary classification, a weak hypothesis
            # `h_t` with error significantly larger than 1/2 is of
            # equal value to one with error significantly less than
            # 1/2 since `h_t` can be replaced by its negation `−h_t`
            # (an effect that happens “automatically” in AdaBoost,
            # which chooses αt < 0 in such a case). (p. 305)

            if np.abs(error - 0.5) > self.gamma:
                # Calculate alpha
                epsilon = 1e-10  # using epsilon to avoid dividion by zero
                self.alpha = 0.5 * np.log(
                    (1 - error) / (error + epsilon)
                    )
                self.feature, self.threshold = feature, threshold
                return self
        return self

    def predict(self, X):
        """
        predicts with a simple rule
        """
        n_samples = X.shape[0]
        X_column = X[:, self.feature]
        predictions = np.ones(n_samples)
        predictions[X_column < self.threshold] = -1

        return predictions


class AdaBoost:
    """
    AdaBoost meta-algorithm
    """

    def __init__(self, n_classifiers=200, gamma=12, beta=0.5, random_state=0):
        """
        setting attributes
        """
        self.n_classifiers = n_classifiers
        self.classifiers = []
        self.gamma = gamma
        self.beta = beta
        self.complexity = None
        np.random.seed(seed=random_state)

    def fit(self, X, y):
        """
        calling base learner to construct base classifiers
        """
        self.complexity = 0  # to know the number of calculations

        n_samples, n_features = X.shape
        thresholds = {
            feature: np.sort(np.unique(X[:, feature]))[1:]
            for feature in range(n_features)
            }

        # suggested by Schapire >> YouTube: L6BlpGnCYVg 28:25
        gamma = self.gamma / np.sqrt(n_samples)

        # initialize weights as a uniform distribution
        weights = np.full(n_samples, (1/n_samples))

        self.classifiers = set()

        # iterate through classifiers
        gamma_ = gamma
        n = 0
        while n < self.n_classifiers:

            print(f'calling base classifier: {n = }, {gamma_ = }, {self.complexity = }')
            base_classifier = DecisionStump(
                gamma=gamma_,
                complexity=self.complexity
                )

            base_classifier = base_classifier.fit(
                X,
                y,
                weights,
                n_samples,
                thresholds
                )

            self.complexity = base_classifier.complexity

            # check if `weak learner` succeds in learning
            if base_classifier.feature is None:
                gamma_ *= self.beta
                print(f'ACHTUNG! {gamma_ = }')
                continue

            n += 1
            print(f'{n = }')

            # Calculate predictions and update weights
            y_predict = base_classifier.predict(X)

            """
            On each round, the weights of incorrectly
            classified examples are increased so that,
            effectively, hard examples get successively
            higher weight, forcing the base learner to
            focus its attention on them.
            """
            weights *= np.exp(
                -1 * base_classifier.alpha * y * y_predict
                )
            weights /= np.sum(weights)

            # Save classifiers
            gamma_ = gamma
            self.classifiers.add(base_classifier)

    def predict(self, X):
        """
        predict for multiclass classification
        """
        return np.sum(
            base_classifier.alpha * base_classifier.predict(X)
            for base_classifier in self.classifiers
            )
