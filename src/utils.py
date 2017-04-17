import os


class Dataset:
    def __init__(self):
        self.documents = []

    def add_document(self, document):
        self.documents.append(document)

    def get_stats(self):
        pass

    def last_doc(self):
        return self.documents[len(self.documents)-1]


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
    def __init__(self, token, pos, entity):
        self.token = token
        self.pos = pos
        self.entity = entity

    def __repr__(self):
        return self.token


class ProjectPath:
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self):
        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", localtime())
        self.datasets = os.path.join(ProjectPath.base, "data", "datasets")

    def get_dataset(self, language, dataset):
        return os.path.join(self.datasets, language, language+"."+dataset)


def read_dataset(path):
    dataset = Dataset()
    document = sentence = None
    filename = path.rsplit('/', 1)[-1]
    with open(path, "r") as f:
        new_sentence = True
        while True:
            line = f.readline()
            if line == '': break
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
                if filename.startswith('eng_old_encoding'):
                    sentence.add_word(Word(args[0], args[1], args[3]))
                elif filename.startswith('esp'):
                    sentence.add_word(Word(args[0], None, args[1]))
                elif filename.startswith('ned'):
                    sentence.add_word(Word(args[0], args[1], args[2]))
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
    eng = read_dataset(project_path.get_dataset('eng_old_encoding', dataset))
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
                            index, counter = next_entity(index+1, words, entity.replace("B-", "I-"), f, counter)
                        else:
                            f.write(words[index].token + " " + words[index].pos + " - " + get_starting(entity) + "\n")
                            index, counter = next_entity(index+1, words, entity, f, counter)
                    index += 1
                    counter += 1

                f.write("\n")
                counter += 1
    return flags

project_path = ProjectPath()
