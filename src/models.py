from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, precision_score, f1_score

from src.feature_extraction import *
from src.preprocessing import fit_transform_column
from src.utils import current_milli_time, print_ms


class Model:

    def __init__(self, alpha_range, penalty, n_iter):
        self._best_estimator = None
        self.alpha_range = alpha_range
        self.penalty = penalty
        self.n_iter = n_iter

    def train(self, train_features, train_y, validation_features, validation_y, cv=True):
        print('\ntraining model')

        t6 = current_milli_time()
        # lr = LogisticRegressionCV(class_weight='balanced', n_jobs=-1, max_iter=300000, multi_class='multinomial')
        # lr.fit(train_features, train_y)
        if not cv:
            self._best_estimator = Perceptron(n_jobs=-1, alpha=0.00001, penalty=self.penalty,
                                              shuffle=True, n_iter=self.n_iter)
        else:
            max_f1_micro = None
            max_f1_macro = None
            for alpha in self.alpha_range:
                p = Perceptron(n_jobs=-1, alpha=alpha, penalty=self.penalty, shuffle=True, n_iter=self.n_iter)
                p.fit(train_features, train_y)

                print_ms('\ntraining done: ', t6, current_milli_time())
                predicted_y = p.predict(validation_features)
                temp_f1 = f1_score(validation_y, predicted_y, average='micro')
                if max_f1_micro is None or max_f1_micro < temp_f1:
                    max_f1_micro = temp_f1
                    max_f1_macro = f1_score(validation_y, predicted_y, average='macro')
                    self._best_estimator = p
                    # print(p.get_params())
                    # print(classification_report(validation_y, predicted_y))
                    # print("micro: ", precision_score(validation_y, predicted_y, average='micro'))
                    # print("macro: ", precision_score(validation_y, predicted_y, average='macro'))
                    # print("#"*100)
        self._best_estimator.fit(np.append(train_features, validation_features, axis=0),
                                 np.append(train_y, validation_y))

    def predict(self, x):
        raise NotImplemented('This method is not implemented')

    def _make_feature_vec(self, word, prev, prev_prev, next, next_next):
        raise NotImplemented('This method is not implemented')

    def _get_features(self, data):
        features = []
        Y = []
        for doc in data.documents:
            for sentance in doc.sentences:
                n = len(sentance.words)
                for i in range(n):
                    prev = get_prev(sentance.words, i, 1)
                    prev_prev = get_prev(sentance.words, i, 2)
                    next = get_next(sentance.words, i, 1)
                    next_next = get_next(sentance.words, i, 2)
                    features.append(self._make_feature_vec(sentance.words[i], prev, prev_prev, next, next_next))
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

    def __init__(self, alpha_range=[10 ** i for i in range(-10, -1)], penalty='l2', n_iter=5):
        super().__init__(alpha_range, penalty, n_iter)

    def predict(self, test_features):
        return self._best_estimator.predict(test_features)

    def _make_feature_vec(self, word, prev, prev_prev, next, next_next):
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


class BaseLineAndGazetters(Model):
    def __init__(self, alpha_range=[10 ** i for i in range(-10, -1)], penalty='l2', n_iter=5):
        super().__init__(alpha_range, penalty, n_iter)

    def predict(self, test_features):
        return self._best_estimator.predict(test_features)

    def _make_feature_vec(self, word, prev, prev_prev, next, next_next):
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

    def _add_ordinal_features(self, new_features, train, validation, test):
        # add new features
        return new_features