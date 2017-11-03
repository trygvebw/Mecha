import numpy as np

import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.core.converters import ndarray_to_instances
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble.forest import ForestRegressor
from sklearn.utils import check_array
from scipy.sparse import issparse

class ModelTreeRegressor(BaseEstimator, RegressorMixin):
    """A frontend for Weka's M5P models."""
    
    def __init__(self):
        self._classifier = None
        self._n_features = None

    def fit(self, X, y, **kwargs):
        if (len(y.shape) > 2) or (len(y.shape) == 2 and y.shape[1] > 1):
            raise Exception('y had dimensions ' + str(y.shape))

        if not jvm.started:
            jvm.start()
        self._classifier = Classifier(classname='weka.classifiers.trees.M5P')
        self._n_features = X.shape[1]
        joined_data = np.hstack((X, y.reshape((-1, 1))))
        weka_data = ndarray_to_instances(joined_data, 'joined_data')
        weka_data.class_is_last()
        self._classifier.build_classifier(weka_data)
        return self

    def predict(self, X, **kwargs):
        assert self._classifier is not None

        y_dummy = np.zeros((X.shape[0], 1))
        joined_data = np.hstack((X, y_dummy))
        weka_data = ndarray_to_instances(joined_data, 'joined_data')

        predictions = []
        for instance in weka_data:
            predictions.append(self._classifier.classify_instance(instance))
        return np.array(predictions)

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X)
            if issparse(X):
                raise ValueError("No support for sparse matrices")

        n_features = X.shape[1]
        if self._n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self._n_features, n_features))
        return X

class ModelTreeForestRegressor(ForestRegressor):
    def __init__(self, n_estimators=10):
        super().__init__(
            base_estimator=ModelTreeRegressor(),
            n_estimators=n_estimators)