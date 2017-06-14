import time
from nltk import SnowballStemmer
import numpy as np
from src.project_path import project_path


def current_milli_time():
    return int(round(time.time() * 1000))


def print_ms(message, t1, t2):
    print(message, t2-t1, 'ms')


class FeatureSet:
    def __init__(self, data, train_len, validation_len, test_len):
        self.data = np.array(data)
        self.train_len = train_len
        self.validation_len = validation_len
        self.test_len = test_len


class Data:
    def __init__(self, corpus):
        self.corpus = corpus
        self.train = read_datasets(project_path.get_dataset(corpus, 'train'))
        self.validation = read_datasets(project_path.get_dataset(corpus, 'validation'))
        self.test = read_datasets(project_path.get_dataset(corpus, 'test'))
        print("test")

    def print_stats(self):
        print("Corpus: ", self.corpus)
        print("\ntrain:")
        self.train.print_stats()
        print("\nvalidation:")
        self.validation.print_stats()
        print("\ntest:")
        self.test.print_stats()

    def get_set(self, set):
        if set == 'train':
            return self.train
        elif set == 'validation':
            return self.validation
        elif set == 'test':
            return self.test
        else:
            return None


class Dataset:
    def __init__(self):
        self.documents = []
        self.entity_counter = {}

    def add_document(self, document):
        self.documents.append(document)

    def print_stats(self):
        print("Entity counters:")
        for key in self.entity_counter:
            print(key, " -> ", self.entity_counter[key])

    def last_doc(self):
        return self.documents[len(self.documents) - 1]

    def merge(self, dataset):
        self.documents.extend(dataset.documents)
        self.entity_counter.update(dataset.entity_counter)


class Document:
    def __init__(self):
        self.sentences = []

    def add_sentence(self, sentence):
        self.sentences.append(sentence)

    def remove_last_sentence(self):
        self.sentences.pop()

    def __repr__(self):
        return " ".join(self.sentences)


class Sentence:
    def __init__(self):
        self.words = []

    def add_word(self, word):
        self.words.append(word)

    def __repr__(self):
        return " ".join(self.words)


class Word:
    def __init__(self, token, pos, entity, stem):
        self.token = token
        self.pos = pos
        self.entity = entity
        self.stem = stem
        # self.tag = tag

    def __repr__(self):
        return self.token


def set_recall(tp, fp):
    if tp == 0:
        return 0
    else:
        return tp / (tp + fp)


def set_precision(tp, fn):
    if tp == 0:
        return 0
    else:
        return tp / (tp + fn)


def set_f1_score(precision, recall):
    f1_score = 0
    if precision != 0 and recall != 0:
        f1_score = 2*precision*recall/(precision + recall)
    return f1_score


def set_accuracy(tp, tn, fp, fn):
    if tp != 0 and tn != 0:
        return (tp + tn) / (tp + tn + fp + fn)
    return 0


def get_metrics_for_entity(param):
    # tp = fp = fn = tn = 0
    precision = set_precision(param[0], param[2])
    recall = set_recall(param[0], param[1])
    return [precision, recall, set_accuracy(param[0], param[3], param[1], param[2]),
            set_f1_score(precision, recall)]


def _get_exact_metrics(true_y, predicted_y):
    mapa = {}
    mapa['PER'] = [0, 0, 0, 0]
    mapa['LOC'] = [0, 0, 0, 0]
    mapa['ORG'] = [0, 0, 0, 0]
    mapa['MISC'] = [0, 0, 0, 0]
    tp = fp = fn = tn = 0
    i = 0
    while i < len(true_y):
        if true_y[i] == 'O':
            if predicted_y[i] == 'O':
                for index in mapa:
                    lista = mapa[index]
                    lista[3] += 1
                    mapa[index] = lista
            else:
                entity_type = predicted_y[i].split('-')[1]
                lista = mapa[entity_type]
                lista[1] += 1
                mapa[entity_type] = lista
            i += 1
        elif true_y[i].startswith('B'):
            entity_type = true_y[i].split('-')[1]

            start = i
            entity = [true_y[i]]
            i += 1
            while i < len(true_y) and true_y[i] != 'O' and entity_type == true_y[i].split('-')[1] and \
                    true_y[i].startswith('I'):
                entity.append(true_y[i])
                i += 1
            if len([i for i, j in zip(entity, predicted_y[start:i]) if i == j]) == len(entity):
                lista = mapa[entity_type]
                lista[0] += 1
                mapa[entity_type] = lista
            else:
                lista = mapa[entity_type]
                lista[2] += 1
                mapa[entity_type] = lista
    entity_metrics = {}
    entity_metrics['PER'] = get_metrics_for_entity(mapa['PER'])
    entity_metrics['LOC'] = get_metrics_for_entity(mapa['LOC'])
    entity_metrics['ORG'] = get_metrics_for_entity(mapa['ORG'])
    entity_metrics['MISC'] = get_metrics_for_entity(mapa['MISC'])
    return entity_metrics


class Eval:
    def __init__(self, true_y, predicted_y, exact=False):
        self.class_index = self._init_class_index()
        self.exact = exact
        if self.exact:
            self.exact_metrics = _get_exact_metrics(true_y, predicted_y)
        else:
            self.confusion_matrix = np.zeros((9, 9))
            self.instances = np.zeros(9)
            for i in range(len(true_y)):
                self.confusion_matrix[self.class_index[true_y[i]]][self.class_index[predicted_y[i]]] += 1
                self.instances[self.class_index[true_y[i]]] += 1
            self.precisions, self.recalls, self.f1_scores, self.accuracies = self._init_metrics()

    def get_confusion_matrix(self):
        return self.confusion_matrix

    @staticmethod
    def _init_class_index():
        return {'B-LOC': 0, 'B-MISC': 1, 'B-ORG': 2, 'B-PER': 3,
                'I-LOC': 4, 'I-MISC': 5, 'I-ORG': 6, 'I-PER': 7, 'O': 8}

    def _init_metrics(self):
        precisions = np.zeros(len(self.class_index))
        recalls = np.zeros(len(self.class_index))
        f1_scores = np.zeros(len(self.class_index))
        accuracies = np.zeros(len(self.class_index))
        for clazz, index in self.class_index.items():
            tp, fp, fn, tn = self._class_stats_from_confusion_matrix(index)
            precisions[index] = set_precision(tp, fp)
            recalls[index] = set_recall(tp, fn)
            f1_scores[index] = set_f1_score(precisions[index], recalls[index])
            accuracies[index] = set_accuracy(tp, tn, fp, fn)
        return precisions, recalls, f1_scores, accuracies

    def _class_stats_from_confusion_matrix(self, index):
        tp = self.confusion_matrix[index][index]
        fp = self._sum_column(index) - self.confusion_matrix[index][index]
        fn = self._sum_row(index) - self.confusion_matrix[index][index]
        tn = np.sum(self.confusion_matrix) - fn - fp + tp
        return tp, fp, fn, tn

    def _sum_row(self, index):
        return np.sum(self.confusion_matrix[index, :])

    def _sum_column(self, index):
        return np.sum(self.confusion_matrix[:, index])

    def get_avg_f1_score(self):
        if self.exact:
            f1_sum = 0
            for i in self.exact_metrics:
                f1_sum = self.exact_metrics[i][3]
            return f1_sum/4
        else:
            return np.average(self.f1_scores)

# Eval(['O', 'B-PER','I-PER','B-LOC','O','B-ORG','I-ORG','O'], ['O', 'B-PER','I-PER','B-LOC','B-PER','B-ORG','I-ORG','B-PER'], exact=True)
# Eval(['O', 'B-PER','I-PER','B-LOC','O','B-ORG','I-ORG','O'], ['O', 'B-PER','I-PER','B-LOC','O','B-ORG','I-ORG','O'], exact=True)


def get_data(corpus):
    return Data(corpus)


def read_datasets(path):
    filename = path.rsplit('/', 1)[-1]
    if filename.startswith('eng'):
        return read_eng_dataset(path)
    elif filename.startswith('esp'):
        return read_esp_dataset(path)
    elif filename.startswith('ned'):
        return read_ned_dataset(path)


def read_eng_dataset(path):
    dataset = Dataset()
    document = sentence = None
    stemmer = SnowballStemmer("english")
    with open(path, encoding='utf-8', mode='r') as f:
        new_sentence = True
        while True:
            line = f.readline()
            if line == '':
                break
            if line.startswith('-DOCSTART-'):
                document = Document()
                dataset.add_document(document)
            elif line.strip() == '':
                new_sentence = True
            else:
                if new_sentence:
                    sentence = Sentence()
                    document.add_sentence(sentence)
                    new_sentence = False
                args = line.split()
                ent_type = entity = None
                sentence.add_word(Word(args[0], args[1], args[3], stemmer.stem(args[0])))
                if args[3] != 'O':
                    ent_type, entity = args[3].split('-', 2)
                if ent_type == 'B':
                    dataset.entity_counter[entity] = dataset.entity_counter.get(entity, 0) + 1
    return dataset


def read_esp_dataset(path):
    dataset = Dataset()
    document = Document()
    sentence = None
    stemmer = SnowballStemmer("spanish")
    dataset.add_document(document)
    with open(path, encoding='latin-1', mode='r') as f:
        new_sentence = True
        while True:
            line = f.readline()
            if line == '':
                break
            if line.strip() == '':
                new_sentence = True
            else:
                if new_sentence:
                    sentence = Sentence()
                    document.add_sentence(sentence)
                    new_sentence = False
                args = line.split()
                ent_type = entity = None
                sentence.add_word(Word(args[0], None, args[1], stemmer.stem(args[0])))
                if args[1] != 'O':
                    ent_type, entity = args[1].split('-', 2)
                if ent_type == 'B':
                    dataset.entity_counter[entity] = dataset.entity_counter.get(entity, 0) + 1
    return dataset


def read_ned_dataset(path):
    dataset = Dataset()
    document = sentence = None
    stemmer = SnowballStemmer("dutch")
    with open(path, encoding='latin-1', mode='r') as f:
        new_sentence = True
        while True:
            line = f.readline()
            if line == '':
                break
            if line.startswith('-DOCSTART-'):
                document = Document()
                dataset.add_document(document)
            elif line.strip() == '':
                new_sentence = True
            else:
                if new_sentence:
                    sentence = Sentence()
                    document.add_sentence(sentence)
                    new_sentence = False
                args = line.split()
                ent_type = entity = None
                sentence.add_word(Word(args[0], args[1], args[2], stemmer.stem(args[0])))
                if args[2] != 'O':
                    ent_type, entity = args[2].split('-', 2)
                if ent_type == 'B':
                    dataset.entity_counter[entity] = dataset.entity_counter.get(entity, 0) + 1
    return dataset


def get_starting(entity):
    return entity.replace('I-', 'B-')


def normalize_flags(flags):
    new_flags = []
    prev_flag = -2
    for flag in flags:
        if flag != prev_flag + 1:
            new_flags.append(flag)
        prev_flag = flag
    return new_flags


def next_entity(index, words, entity, f, counter):
    while index < len(words) and words[index].entity == entity:
        f.write(words[index].token + " " + words[index].pos + " - " + entity + "\n")
        index += 1
        counter += 1
    return index - 1, counter


def transfer_eng_to_bio(dataset, output):
    counter = 0
    flags = []
    eng = read_datasets(project_path.get_dataset('eng_old_encoding', dataset))
    with open(output, 'w') as f:
        for doc in eng.documents:
            f.write("-DOCSTART- -X- -X- O\n\n")
            counter += 2
            for sentence in doc.sentences:
                index = 0
                sentence_length = len(sentence.words)
                words = sentence.words
                while index < sentence_length:
                    entity = words[index].entity
                    if entity.startswith('B-'):
                        flags.append(counter)
                    if entity == 'O':
                        f.write(words[index].token + " " + words[index].pos + " - " + entity + "\n")
                    else:
                        if entity.startswith('B'):
                            f.write(words[index].token + " " + words[index].pos + " - " + entity + "\n")
                            index, counter = next_entity(index + 1, words, entity.replace("B-", "I-"), f, counter)
                        else:
                            f.write(words[index].token + " " + words[index].pos + " - " + get_starting(entity) + "\n")
                            index, counter = next_entity(index + 1, words, entity, f, counter)
                    index += 1
                    counter += 1

                f.write("\n")
                counter += 1
    return flags


def count_entity_size_dataset(dataset, counter):
    for document in dataset.documents:
        for sentence in document.sentences:
            i = size = 0
            n = len(sentence.words)
            entity = None
            while i < n:
                word = sentence.words[i]
                if word.entity == 'O':
                    if entity is not None:
                        entity_size_list = counter.get(entity, [])
                        entity_size_list.append(size)
                        counter[entity] = entity_size_list
                    entity = None
                    size = 0
                else:
                    if word.entity.startswith('I-'):
                        size += 1
                    elif word.entity.startswith('B-'):
                        if entity is not None:
                            entity_size_list = counter.get(entity, [])
                            entity_size_list.append(size)
                            counter[entity] = entity_size_list
                        entity = word.entity.split('-', 2)[-1]
                        size = 1
                i += 1
            if entity is not None:
                entity_size_list = counter.get(entity, [])
                entity_size_list.append(size)
                counter[entity] = entity_size_list


def count_entity_size(data):
    counter = {}
    count_entity_size_dataset(data.train, counter)
    count_entity_size_dataset(data.validation, counter)
    count_entity_size_dataset(data.test, counter)
    return counter


def plot_correllation_matrix():
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(context="paper", font="monospace")

    # Load the datset of correlations between cortical brain networks
    df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)
    corrmat = df.corr()

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 9))

    # Draw the heatmap using seaborn
    sns.heatmap(corrmat, vmax=.8, square=True)

    # Use matplotlib directly to emphasize known networks
    networks = corrmat.columns.get_level_values("network")
    for i, network in enumerate(networks):
        if i and network != networks[i - 1]:
            ax.axhline(len(networks) - i, c="w")
            ax.axvline(i, c="w")
    f.tight_layout()
    sns.plt.show()


# plot_correllation_matrix()
