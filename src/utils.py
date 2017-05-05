from nltk import SnowballStemmer

from src.project_path import project_path


class Data:
    def __init__(self, corpus):
        self.corpus = corpus
        self.train = read_datasets(project_path.get_dataset(corpus, 'train'))
        self.validation = read_datasets(project_path.get_dataset(corpus, 'validation'))
        self.test = read_datasets(project_path.get_dataset(corpus, 'test'))

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
