import argparse
import pickle

from src.utils import get_data, Dataset, print_ms, current_milli_time
from src.models import BaseLine, BaseLineAndGazetters


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


def main(args, model, results_file):
    train_sets, validation_sets, test_sets = parse_arguments(args)
    train, validation, test = get_datasets(train_sets, validation_sets, test_sets)
    # model = BaseLine()
    train_features, train_y, validation_features, validation_y, test_features, test_y, encoders = \
        model.get_transform_features(train, validation, test)

    model.train(train_features, train_y, validation_features, validation_y)
    predicted_y = model.predict(test_features)
    model.eval(test_y, predicted_y, results_file)


def make_args(train, test):
    return ['/home/stipan/dev/fer/seminar/src/runner.py', '-train', train, '-validation', train, '-test', test]


def all_combinations():
    for i in ['eng', 'esp', 'ned']:
        for j in ['eng', 'esp', 'ned']:
            # main(make_args(i, j), BaseLine(n_iter=50), '../results/temp/' + i + '_' + j)
            main(make_args(i, j), BaseLineAndGazetters(), '../results/temp/' + i + '_' + j)


if __name__ == '__main__':
    # main(sys.argv)
    all_combinations()
