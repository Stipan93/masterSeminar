import os
import numpy as np
import pickle

from nltk import SnowballStemmer
from sklearn import preprocessing
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from src.feature_extraction import *
from src.preprocessing import fit_transform_column
from src.utils import current_milli_time, print_ms, Dataset, Eval


def exact_scorer(true_y, predicted_y):
    scorer = Eval(true_y, predicted_y, exact=True)
    return scorer.get_avg_f1_score()


def generate_trigrams(token):
    trigrams = []
    for i in range(3, len(token)+1):
        trigrams.append(token[i-3:i])
    return trigrams


def _avg(index, exact_metrics):
    suma = 0
    for i in exact_metrics:
        suma += exact_metrics[i][index]
    return suma/4


class Model:

    def __init__(self, alpha_range, penalty, n_iter):
        self._best_estimator = None
        self.alpha_range = alpha_range
        self.penalty = penalty
        self.n_iter = n_iter
        self.trigrams = {}

    def train(self, train_features, train_y, validation_features, validation_y, cv=True):
        print('\ntraining model')

        t6 = current_milli_time()

        train_validation_features = np.append(train_features, validation_features, axis=0)
        train_validation_y = np.append(train_y, validation_y)

        if not cv:
            # self._best_estimator = Perceptron(n_jobs=-1, alpha=0.00001, penalty='l2', class_weight='balanced',
            #                                   shuffle=True, n_iter=self.n_iter)
            self._best_estimator = LogisticRegression(n_jobs=-1, multi_class='ovr', max_iter=100,
                                                      C=10, penalty='l2', class_weight='balanced')
        else:
            best_score = 0

            # for alp in self.alpha_range:
            #     for pen in self.penalty:
            #         temp_model = Perceptron(n_jobs=-1, shuffle=True, n_iter=self.n_iter,
            #                               class_weight='balanced', alpha=alp, penalty=pen)
            for c in [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
                for pen in ['l1', 'l2']:
                    temp_model = LogisticRegression(n_jobs=-1, max_iter=self.n_iter,
                                                      C=c, penalty=pen, class_weight='balanced')
                    cv_scores = []
                    kfold = KFold(n_splits=5)
                    splits = kfold.split(train_validation_features, train_validation_y)
                    for cv_train_index, cv_test_index in splits:
                        cv_train_x = train_validation_features[cv_train_index]
                        cv_test_x = train_validation_features[cv_test_index]
                        cv_train_y = train_validation_y[cv_train_index]
                        cv_test_y = train_validation_y[cv_test_index]
                        temp_model.fit(cv_train_x, cv_train_y)
                        cv_predicted_y = temp_model.predict(cv_test_x)
                        cv_scores.append(exact_scorer(cv_test_y, cv_predicted_y))
                    score = np.average(cv_scores)
                    if score > best_score:
                        self._best_estimator = temp_model
                        best_score = score
            # parameters = {'alpha': self.alpha_range, 'penalty': self.penalty}
            # clf = GridSearchCV(Perceptron(n_jobs=1, shuffle=True, n_iter=self.n_iter),
            #                    parameters, n_jobs=1, cv=5, scoring=make_scorer(exact_scorer))
            # clf.fit(train_validation_features, train_validation_y)
            # self._best_estimator = clf.best_estimator_
            # print(classification_report(validation_y, predicted_y))
            # print("micro: ", precision_score(validation_y, predicted_y, average='micro'))
            # print("macro: ", precision_score(validation_y, predicted_y, average='macro'))
            # print("#"*100)

        self._best_estimator.fit(train_validation_features, train_validation_y)
        print_ms('\ntraining done: ', t6, current_milli_time())

    def predict(self, x):
        raise NotImplemented('This method is not implemented')

    def _make_feature_vec(self, word, prev, prev_prev, next, next_next, next_next_next):
        raise NotImplemented('This method is not implemented')

    def _get_features(self, data):
        features = []
        Y = []
        for doc in data.documents:
            for sentance in doc.sentences:
                n = len(sentance.words)
                for i in range(n):
                    token = get_token(sentance.words[i])
                    entity_type = get_entity_type(get_entity(sentance.words[i]))
                    entity_trigrams = self.trigrams.get(entity_type, {})
                    for tri in generate_trigrams(token):
                        entity_trigrams[tri] = entity_trigrams.get(tri, 0) + 1
                    self.trigrams[entity_type] = entity_trigrams

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

    def _add_ordinal_features(self, new_features, train, validation, test, scaled=True):
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
        scorer = Eval(true_y, predicted_y, exact=True)
        scorer_normal = Eval(true_y, predicted_y)
        with open(filename, 'w') as f:
            f.write(str(self._best_estimator.get_params()) + '\n\n')
            # f.write(str(classification_report(true_y, predicted_y)) + '\n')\
            f.write('      P')
            f.write('    R')
            f.write('    A')
            f.write('    F1\n')
            for ent in scorer.exact_metrics:
                a = scorer.exact_metrics[ent]
                metrics_str = "%.2f %.2f %.2f %.2f" % (a[0], a[1], a[2], a[3])
                f.write(ent+' '+metrics_str+'\n')
            avg_metrics_str = "%.2f %.2f %.2f %.2f" % (_avg(0, scorer.exact_metrics), _avg(1, scorer.exact_metrics),
                                                   _avg(2, scorer.exact_metrics), _avg(3, scorer.exact_metrics))
            f.write('\nAVG  ' + avg_metrics_str + '\n')
            f.write("\n#######################################################\n")
            classes = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
            f.write('        P   R   A   F1\n')
            for index, clazz in enumerate(classes):
                f.write(clazz)
                f.write(' %.2f' % (scorer_normal.precisions[index],))
                f.write(' %.2f' % (scorer_normal.recalls[index],))
                f.write(' %.2f' % (scorer_normal.accuracies[index],))
                f.write(' %.2f' % (scorer_normal.f1_scores[index],))
                f.write("\n")
            f.write("\nAVG  ")
            f.write(' %.2f' % (np.average(scorer_normal.precisions),))
            f.write(' %.2f' % (np.average(scorer_normal.recalls),))
            f.write(' %.2f' % (np.average(scorer_normal.accuracies),))
            f.write(' %.2f' % (np.average(scorer_normal.f1_scores),))

        print(self._best_estimator.get_params())
        # print("micro: ", precision_score(true_y, predicted_y, average='micro'))
        # print("macro: ", precision_score(true_y, predicted_y, average='macro'))
        # print("precision: " + str(scorer.get_precision()))
        # print("recall: " + str(scorer.get_recall()))
        # print("f1 score: " + str(scorer.get_f1_score()))
        # print("accuracy: " + str(scorer.get_accuracy()))
        print("#" * 150)


def sum_trigrams_score(map, trigrams):
    score = 0
    for i in trigrams:
        score += map.get(i, 0)
    return score


class BaseLine(Model):

    def __init__(self, alpha_range=[10 ** i for i in range(-10, -1)], penalty=list(['l1', 'l2']), n_iter=5):
        super().__init__(alpha_range, penalty, n_iter)

    def predict(self, test_features):
        return self._best_estimator.predict(test_features)

    def _add_ordinal_features(self, new_features, train, validation, test, scaled=True):
        dataset = Dataset()
        dataset.merge(train)
        dataset.merge(validation)
        dataset.merge(test)
        ordinal_features = []
        for doc in dataset.documents:
            for sentance in doc.sentences:
                n = len(sentance.words)
                for i in range(n):
                    word = get_token(sentance.words[i])
                    feature_vector = []
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('PER', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('LOC', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('ORG', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('MISC', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('O', {}), generate_trigrams(word)))
                    ordinal_features.append(feature_vector)
        # scale ordinal features
        if scaled:
            scaler = MinMaxScaler()
            ordinal_features = scaler.fit_transform(ordinal_features)
        return np.append(new_features, np.array(ordinal_features), axis=1)

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
    gazetters = {}
    for file in os.listdir(dir):
        stemmer = SnowballStemmer(file.split('-')[0])
        with open(os.path.join(dir, file), 'r') as f:
            for line in f.readlines():
                words = line.lower().split()
                for i in range(len(words)):
                    word = stemmer.stem(words[i])
                    positions = gazetters.get(word, {})
                    counter = positions.get(i, 0)
                    counter += 1
                    positions[i] = counter
                    gazetters[word] = positions
    return gazetters


# def get_features_for_entity(words, gazetters):
#     while None in words:
#         words.remove(None)
#     while len(words) > 0:
#         phrase = " ".join(words).lower()
#         if phrase in gazetters:
#             return len(words)
#         words.pop(-1)
#     return 0


def populate_topics(lines, file, topics, stemmer):
    for line in lines:
        words = line.lower().split()
        for i in range(len(words)):
            stem = stemmer.stem(words[i])
            words_on_place_i = topics.get(i, {})
            list_of_topics = words_on_place_i.get(stem, [])
            list_of_topics.append(file)
            words_on_place_i[stem] = list_of_topics
            topics[i] = words_on_place_i


def get_entity_count(word, index, gazetters):
    return gazetters.get(word.stem, {}).get(index, 0)


class BaseLineAndGazetters(BaseLine):
    def __init__(self, alpha_range=[10 ** i for i in range(-10, -1)], penalty=list(['l1', 'l2']), n_iter=5):
        super().__init__(alpha_range, penalty, n_iter)

        with open('../gazetters/topics.bin', 'rb') as handle:
            self.topics = pickle.load(handle)

        with open('../gazetters/per.bin', 'rb') as handle:
            self.per = pickle.load(handle)

        with open('../gazetters/loc.bin', 'rb') as handle:
            self.loc = pickle.load(handle)

        with open('../gazetters/org.bin', 'rb') as handle:
            self.org = pickle.load(handle)
        # self.topics = {}
        # topics_dir = '../gazetters/topics'
        # for file in os.listdir(topics_dir):
        #     stemmer = SnowballStemmer('english')
        #     with open(os.path.join(topics_dir, file), 'r') as f:
        #         populate_topics(f.readlines(), file, self.topics, stemmer)
        # with open('../gazetters/topics.bin', 'wb') as handle:
        #     pickle.dump(self.topics, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.per = gazetters_for_entity('per')
        # with open('../gazetters/per.bin', 'wb') as handle:
        #     pickle.dump(self.per, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.loc = gazetters_for_entity('loc')
        # with open('../gazetters/loc.bin', 'wb') as handle:
        #     pickle.dump(self.loc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.org = gazetters_for_entity('org')
        # with open('../gazetters/org.bin', 'wb') as handle:
        #     pickle.dump(self.org, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # print('over')

    def _get_gazetters_features(self, word, index):
        return [get_entity_count(word, index, self.per),
                get_entity_count(word, index, self.org),
                get_entity_count(word, index, self.loc)]
        # return [get_features_for_entity(words, self.per),
        #         get_features_for_entity(words, self.org),
        #         get_features_for_entity(words, self.loc)]

    def _add_ordinal_features(self, new_features, train, validation, test, scaled=True):
        dataset = Dataset()
        dataset.merge(train)
        dataset.merge(validation)
        dataset.merge(test)
        ordinal_features = []
        for doc in dataset.documents:
            for sentance in doc.sentences:
                n = len(sentance.words)
                for i in range(n):
                    word = get_token(sentance.words[i])
                    feature_vector = self._get_gazetters_features(sentance.words[i], i)
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('PER', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('LOC', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('ORG', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('MISC', {}), generate_trigrams(word)))
                    feature_vector.append(sum_trigrams_score(self.trigrams.get('O', {}), generate_trigrams(word)))
                    ordinal_features.append(feature_vector)
        # scale ordinal features
        if scaled:
            scaler = MinMaxScaler()
            ordinal_features = scaler.fit_transform(ordinal_features)
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
        vec.append(get_topic(word, self.topics.get(0, {})))
        vec.append(get_topic(next, self.topics.get(1, {})))
        vec.append(get_topic(next_next, self.topics.get(2, {})))
        vec.append(get_topic(next_next_next, self.topics.get(3, {})))
        return vec
