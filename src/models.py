import os
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, precision_score, f1_score
from sklearn.model_selection import GridSearchCV

from src.feature_extraction import *
from src.preprocessing import fit_transform_column
from src.utils import current_milli_time, print_ms, Dataset


class Model:

    def __init__(self, alpha_range, penalty, n_iter):
        self._best_estimator = None
        self.alpha_range = alpha_range
        self.penalty = penalty
        self.n_iter = n_iter

    def train(self, train_features, train_y, validation_features, validation_y, cv=True):
        print('\ntraining model')

        t6 = current_milli_time()

        train_validation_features = np.append(train_features, validation_features, axis=0)
        train_validation_y = np.append(train_y, validation_y)

        if not cv:
            self._best_estimator = Perceptron(n_jobs=-1, alpha=0.00001, penalty='l2',
                                              shuffle=True, n_iter=self.n_iter)
        else:
            parameters = {'alpha': self.alpha_range, 'penalty': self.penalty}
            clf = GridSearchCV(Perceptron(n_jobs=-1, shuffle=True, n_iter=self.n_iter), parameters, n_jobs=-1, cv=5)
            clf.fit(train_validation_features, train_validation_y)
            print_ms('\ntraining done: ', t6, current_milli_time())
            self._best_estimator = clf.best_estimator_
            # print(clf.get_params())
            # print(classification_report(validation_y, predicted_y))
            # print("micro: ", precision_score(validation_y, predicted_y, average='micro'))
            # print("macro: ", precision_score(validation_y, predicted_y, average='macro'))
            # print("#"*100)
        self._best_estimator.fit(train_validation_features, train_validation_y)

    def predict(self, x):
        raise NotImplemented('This method is not implemented')

    def _make_feature_vec(self, word, prev, prev_prev, next, next_next, next_next_next):
        raise NotImplemented('This method is not implemented')

    def _get_features(self, data):
        features = []
        Y = []
        print(len(data.documents))
        for doc in data.documents:
            print(len(doc.sentences))
            for sentance in doc.sentences:
                n = len(sentance.words)
                for i in range(n):
                    prev = get_prev(sentance.words, i, 1)
                    prev_prev = get_prev(sentance.words, i, 2)
                    next = get_next(sentance.words, i, 1)
                    next_next = get_next(sentance.words, i, 2)
                    next_next_next = get_next(sentance.words, i, 3)
                    features.append(self._make_feature_vec(sentance.words[i], prev, prev_prev, next,
                                                           next_next, next_next_next))
                    Y.append(sentance.words[i].entity)
        return np.array(features), np.array(Y)

    def _get_all_features(self, train, validation, test):
        t1 = current_milli_time()
        print('\ngetting train features')
        train_features, train_y = self._get_features(train)
        t2 = current_milli_time()
        print_ms('train features: ', t1, t2)

        print('\ngetting validation features')
        validation_features, validation_y = self._get_features(validation)
        t3 = current_milli_time()
        print_ms('validation features: ', t2, t3)

        print('\ngetting test features')
        test_features, test_y = self._get_features(test)
        t4 = current_milli_time()
        print_ms('test features: ', t3, t4)
        return train_features, train_y, validation_features, validation_y, test_features, test_y

    def _add_ordinal_features(self, new_features, train, validation, test):
        return new_features

    def get_transform_features(self, train, validation, test):
        train_features, train_y, validation_features, validation_y, test_features, test_y = \
            self._get_all_features(train, validation, test)
        print('\ntransforming features')
        t5 = current_milli_time()

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
        ohe = preprocessing.OneHotEncoder()
        new_features = ohe.fit_transform(new_features).toarray()

        new_features = self._add_ordinal_features(new_features, train, validation, test)
        print_ms('Features transform: ', t5, current_milli_time())

        return new_features[:train_len, :], train_y, \
               new_features[train_len:train_len + validation_len, :], validation_y, \
               new_features[train_len + validation_len:, :], test_y, encoders

    def eval(self, true_y, predicted_y, filename='../results/temp_eval'):
        with open(filename, 'w') as f:
            f.write(str(self._best_estimator.get_params()) + '\n')
            f.write(str(classification_report(true_y, predicted_y)) + '\n')
            f.write("micro: " + str(precision_score(true_y, predicted_y, average='micro')) + '\n')
            f.write("macro: " + str(precision_score(true_y, predicted_y, average='macro')) + '\n')

        print(self._best_estimator.get_params())
        print("micro: ", precision_score(true_y, predicted_y, average='micro'))
        print("macro: ", precision_score(true_y, predicted_y, average='macro'))
        print("#" * 150)


class BaseLine(Model):

    def __init__(self, alpha_range=[10 ** i for i in range(-10, -1)], penalty=list(['l1', 'l2']), n_iter=5):
        super().__init__(alpha_range, penalty, n_iter)

    def predict(self, test_features):
        return self._best_estimator.predict(test_features)

    def _make_feature_vec(self, word, prev, prev_prev, next, next_next, next_next_next):
        vec = []
        vec.append(get_entity(prev_prev))
        vec.append(get_entity(prev))

        vec.append(alpha_numeric(word.token))
        vec.append(all_digits(word.token))
        vec.append(all_capitalized(word.token))

        # extract 3gram chars from and group the by entity

        # vec.append(get_stem(prev_prev))
        # vec.append(get_stem(prev))
        # vec.append(get_stem(word))
        # vec.append(get_stem(next))
        # vec.append(get_stem(next_next))

        vec.append(is_capitalized(get_token(prev_prev)))
        vec.append(is_capitalized(get_token(prev)))
        vec.append(is_capitalized(word.token))
        vec.append(is_capitalized(get_token(next)))
        vec.append(is_capitalized(get_token(next_next)))
        return vec


def gazetters_for_entity(entity):
    dir = '../gazetters/'+entity
    gazetters = []
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'r') as f:
            gazetters.extend([line.lower() for line in f.readlines()])
    return gazetters


def get_features_for_entity(words, gazetters):
    while None in words:
        words.remove(None)
    while len(words) > 0:
        phrase = " ".join(words).lower()
        if phrase in gazetters:
            return len(words)
        words.pop(-1)
    return 0


def populate_topics(lines, file, topics):
    for line in lines:
        words = line.lower().split()
        for i in range(len(words)):
            words_on_place_i = topics.get(i, {})
            list_of_topics = words_on_place_i.get(words[i], [])
            list_of_topics.append(file)
            words_on_place_i[words[i]] = list_of_topics
            topics[i] = words_on_place_i


class BaseLineAndGazetters(BaseLine):
    def __init__(self, alpha_range=[10 ** i for i in range(-10, -1)], penalty=list(['l1', 'l2']), n_iter=5):
        super().__init__(alpha_range, penalty, n_iter)
        self.topics = {}
        topics_dir = '../gazetters/topics'
        for file in os.listdir(topics_dir):
            with open(os.path.join(topics_dir, file), 'r') as f:
                populate_topics(f.readlines(), file, self.topics)
        self.per = gazetters_for_entity('per')
        self.loc = gazetters_for_entity('loc')
        self.org = gazetters_for_entity('org')

    def _get_gazetters_features(self, words):
        return [get_features_for_entity(words, self.per),
                get_features_for_entity(words, self.org),
                get_features_for_entity(words, self.loc)]

    def _add_ordinal_features(self, new_features, train, validation, test):
        dataset = Dataset()
        dataset.merge(train)
        dataset.merge(validation)
        dataset.merge(test)
        ordinal_features = []
        for doc in dataset.documents:
            for sentance in doc.sentences:
                n = len(sentance.words)
                for i in range(n):
                    word = sentance.words[i]
                    next = get_next(sentance.words, i, 1)
                    next_next = get_next(sentance.words, i, 2)
                    next_next_next = get_next(sentance.words, i, 3)
                    ordinal_features.append(self._get_gazetters_features([word, next, next_next, next_next_next]))
        return np.append(new_features, np.array(ordinal_features), axis=1)

    def predict(self, test_features):
        return self._best_estimator.predict(test_features)

    def _make_feature_vec(self, word, prev, prev_prev, next, next_next, next_next_next):
        vec = []
        vec.append(get_entity(prev_prev))
        vec.append(get_entity(prev))

        vec.append(alpha_numeric(word.token))
        vec.append(all_digits(word.token))
        vec.append(all_capitalized(word.token))

        vec.append(is_capitalized(get_token(prev_prev)))
        vec.append(is_capitalized(get_token(prev)))
        vec.append(is_capitalized(word.token))
        vec.append(is_capitalized(get_token(next)))
        vec.append(is_capitalized(get_token(next_next)))
        vec.extend(get_topic(word, self.topics.get(0, {})))
        vec.append(get_topic(next, self.topics.get(1, {})))
        vec.append(get_topic(next_next, self.topics.get(2, {})))
        vec.append(get_topic(next_next_next, self.topics.get(3, {})))
        return vec
