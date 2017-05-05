import argparse
import pickle
import sys
import numpy as np
import time

from sklearn.linear_model import Perceptron, LogisticRegressionCV
from sklearn.metrics import classification_report, precision_score, f1_score

from src.preprocessing import transform_train_features, transform_test_features, transform_all_features
from src.utils import get_data, Dataset
from src.feature_extraction import get_features


def current_milli_time():
    return int(round(time.time() * 1000))


def print_ms(message, t1, t2):
    print(message, t2-t1, 'ms')


def print_help(parser, message):
    parser.print_help()
    print(message)
    exit(1)


def check_argument_set(arg_set, choices, parser):
    for arg in arg_set:
        if arg not in choices:
            print_help(parser, "'"+arg+"' is not in possible choices: "+str(choices))


def get_set(_set, languages):
    dataset = Dataset()
    for lang in languages:
        data = get_data(lang)
        dataset.merge(data.get_set(_set))
    return dataset


def get_serialized_sets(_set, languages):
    dataset = Dataset()
    for lang in languages:
        with open('../serialization/' + _set + '.' + lang, 'rb') as handle:
            train = pickle.load(handle)
            dataset.merge(train)
    return dataset


def save_data(train, name1, validation, name2, test, name3):
    with open('../serialization/'+name1, 'wb') as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../serialization/'+name2, 'wb') as handle:
        pickle.dump(validation, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../serialization/'+name3, 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(name1, name2, name3):
    with open('../serialization/'+name1, 'rb') as handle:
        train = pickle.load(handle)
    with open('../serialization/' + name2, 'rb') as handle:
        validation = pickle.load(handle)
    with open('../serialization/'+name3, 'rb') as handle:
        test = pickle.load(handle)
    return train, validation, test


def parse_arguments(args):
    choices = ['eng', 'esp', 'ned']
    parser = argparse.ArgumentParser()
    parser.add_argument('-train')
    parser.add_argument('-validation')
    parser.add_argument('-test')

    parsed_args = parser.parse_args(args[1:])
    if None in [parsed_args.train, parsed_args.test]:
        print_help(parser, 'Must provide both train and test sets.')

    train_sets = parsed_args.train.split(',')
    validation_sets = parsed_args.validation.split(',')
    test_sets = parsed_args.test.split(',')

    print('checking arguments')
    check_argument_set(train_sets, choices, parser)
    check_argument_set(validation_sets, choices, parser)
    check_argument_set(test_sets, choices, parser)
    return train_sets, validation_sets, test_sets


def get_datasets(train_sets, validation_sets, test_sets):
    print('loading train, validation and test set')
    t0 = current_milli_time()
    train = get_serialized_sets('train', train_sets)
    validation = get_serialized_sets('validation', validation_sets)
    test = get_serialized_sets('test', test_sets)
    # train = get_set('train', train_sets)
    # validation = get_set('validation', validation_sets)
    # test = get_set('test', test_sets)
    # train, validation, test = load_data('train.'+parsed_args.train, 'validation.'+parsed_args.validation,
    #                                     'test.'+parsed_args.test)
    print_ms('data loaded in: ', t0, current_milli_time())

    # save_data(train, 'train.'+parsed_args.train, validation, 'validation.'+parsed_args.validation,
    #           test, 'test.'+parsed_args.test)
    return train, validation, test


def get_all_features(train, validation, test):
    t1 = current_milli_time()
    print('\ngetting train features')
    train_features, train_y = get_features(train)
    t2 = current_milli_time()
    print_ms('train features: ', t1, t2)

    print('\ngetting validation features')
    validation_features, validation_y = get_features(validation)
    t3 = current_milli_time()
    print_ms('validation features: ', t2, t3)

    print('\ngetting test features')
    test_features, test_y = get_features(test)
    t4 = current_milli_time()
    print_ms('test features: ', t3, t4)
    return train_eatures, train_y, validation_features, validation_y, test_features, test_y


def main(args):
    train_sets, validation_sets, test_sets = parse_arguments(args)
    train, validation, test = get_datasets(train_sets, validation_sets, test_sets)
    train_features, train_y, validation_features, validation_y, test_features, test_y = \
        get_all_features(train, validation, test)

    print('\ntransforming features')
    print(len(train_features))
    print(len(validation_features))
    print(len(test_features))
    print('###################')
    t5 = current_milli_time()
    train_features, validation_features, test_features, encoders = \
        transform_all_features(train_features, validation_features, test_features)
    print_ms('Features transform: ', t5, current_milli_time())
    print(train_features.shape)
    print(validation_features.shape)
    print(test_features.shape)

    print('\ntraining model')

    t6 = current_milli_time()
    # lr = LogisticRegressionCV(class_weight='balanced', n_jobs=-1, max_iter=300000, multi_class='multinomial')
    # lr.fit(train_features, train_y)
    best_estimator = None
    max_f1_micro = None
    max_f1_macro = None
    for alpha in [10**i for i in range(-10, -1)]:
        p = Perceptron(n_jobs=-1, alpha=alpha, penalty='l2', shuffle=True)
        p.fit(train_features, train_y)

        print_ms('\ntraining done: ', t6, current_milli_time())
        predicted_y = p.predict(validation_features)
        temp_f1 = f1_score(validation_y, predicted_y, average='micro')
        if max_f1_micro is None or max_f1_micro < temp_f1:
            max_f1_micro = temp_f1
            max_f1_macro = f1_score(validation_y, predicted_y, average='macro')
            best_estimator = p
        print(p.get_params())
        print(classification_report(validation_y, predicted_y))
        print("micro: ", precision_score(validation_y, predicted_y, average='micro'))
        print("macro: ", precision_score(validation_y, predicted_y, average='macro'))
        print("#"*100)
    # print(len(train.documents))
    # print(len(validation.documents))
    # print(len(test.documents))
    a = np.hstack((train_features, validation_features))
    best_estimator.fit(a, train_y+validation_y)
    predicted_y = best_estimator.predict(test_features)
    print(classification_report(test_y, predicted_y))
    print("micro: ", precision_score(test_y, predicted_y, average='micro'))
    print("macro: ", precision_score(test_y, predicted_y, average='macro'))
    print("#" * 100)


def make_args(train, test):
    return ['/home/stipan/dev/fer/seminar/src/baseline.py', '-train', train, '-validation', train, '-test', test]


def all_combinations():
    for i in ['eng', 'esp', 'ned']:
        for j in ['eng', 'esp', 'ned']:
            main(make_args(i, j))


if __name__ == '__main__':
    # main(sys.argv)
    all_combinations()
