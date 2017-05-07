from sklearn import preprocessing
import numpy as np


def fit_transform_column(encoders, input, column):
    lbe = preprocessing.LabelEncoder()
    encoders.append(lbe)
    return lbe.fit_transform(input[:, column])


def transform_column(encoders, input, column):
    return encoders[column].transform(input[column])


def transform_vector(encoders, vector):
    new_vector = np.zeros(vector.shape)
    for i in range(len(vector)):
        new_vector[i] = transform_column(encoders, vector, i)[0]
    return new_vector


def transform_test_features(features, encoders):
    new_features = np.zeros(features.shape)
    for i in range(features.shape[1]):
        new_features[:, i] = transform_column(encoders, features, i)
    return np.array(new_features)


def transform_train_features(features):
    encoders = []
    new_features = np.zeros(features.shape)
    for i in range(features.shape[1]):
        new_features[:, i] = fit_transform_column(encoders, features, i)
    return np.array(new_features), encoders
