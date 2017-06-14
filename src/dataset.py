from src.utils import get_data, count_entity_size
# from src.project_path import project_path
import matplotlib.pyplot as plt
import numpy as np


def print_entity_size_counter(counter):
    for key, value in counter.items():
        print(key, ' -> ', len(value))


def plot_entity_counter(id, data, name):
    plt.subplot(id)
    plt.rcParams['figure.facecolor'] = 'white'
    plt.xlabel('size')
    plt.ylabel('count')
    plt.title(name)
    plt.xticks(np.arange(min(data[name]), 10, 1.0))
    plt.hist(data[name], bins=range(min(data[name]), 10), color='royalblue', align='left')


def plot_entity_size_counters(counter):
    plt.figure(1, facecolor='white')

    plot_entity_counter(221, counter, 'PER')
    plot_entity_counter(222, counter, 'ORG')
    plot_entity_counter(223, counter, 'LOC')
    plot_entity_counter(224, counter, 'MISC')

    plt.show()

# print(project_path.get_dataset('eng', 'train'))
# print(project_path.get_dataset('eng', 'validation'))
# print(project_path.get_dataset('eng', 'test'))
print('dataset loading')

eng = get_data('eng')
# eng_counter = count_entity_size(eng)
# eng.print_stats()
# plot_entity_size_counters(eng_counter)
# print_entity_size_counter(eng_counter)

print('#'*40)
esp = get_data('esp')
# esp_counter = count_entity_size(esp)
# esp.print_stats()
# plot_entity_size_counters(esp_counter)
# print_entity_size_counter(count_entity_size(esp))

print('#'*40)
ned = get_data('ned')
# ned_counter = count_entity_size(ned)
# ned.print_stats()
# plot_entity_size_counters(ned_counter)

# print_entity_size_counter(count_entity_size(ned))



# transfer_eng_to_bio('train', 'eng.train')
#
# transfer_eng_to_bio('validation', 'eng.validation')
#
# transfer_eng_to_bio('test', 'eng.test')
