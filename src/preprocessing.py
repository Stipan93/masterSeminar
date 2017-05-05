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


def transform_all_features(train_features, validation_features, test_features):
    encoders = []
    train_len, validation_len, test_len = len(train_features), len(validation_features), len(test_features)
    all_features = []
    all_features.extend(train_features)
    all_features.extend(validation_features)
    all_features.extend(test_features)
    all_features = np.array(all_features)
    new_features = np.zeros(all_features.shape)
    for i in range(all_features.shape[1]):
        new_features[:, i] = fit_transform_column(encoders, all_features, i)
    new_features = np.array(new_features)
    ohe = preprocessing.OneHotEncoder()
    new_features = ohe.fit_transform(new_features)
    return new_features[:train_len, :], new_features[train_len:train_len+validation_len, :], \
           new_features[train_len+validation_len:, :], encoders
