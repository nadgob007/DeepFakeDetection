from DeepFakes.functions import *
import os       # файлы

"""
     сценарий для полной обработки (БПФ, Азимутальное усреднение)
"""


def scenario1(initial_params):
    path = initial_params['path']

    a = 0  # Подаются пути к данным, создаются txt файлы с psd и выборки. Использовать для перерасчёта.
    if a == 1:
        data_to_psd(initial_params['size_of_sample'], initial_params['number_of_samples'],
                    initial_params['p'], path + '\\true2', path + '\\false2', path)

    b = 0  # Классификация по имеющимся txt файлам
    if b == 1:
        interval = [[0, initial_params['count_of_features']]]
        classification(path + '\\split', initial_params['number_of_samples'], interval)

    c = 0  # Отображение данных классификаторов
    if c == 1:
        kn_all, svm_all, dt_all = read_acc(initial_params['path'] + '\\split\\acc.txt')
        show_acc(len(kn_all), kn_all, svm_all, dt_all, title='по всем выборкам')

    d = 0  # Перещёт классификаторами для интервала в 10 признаков
    if d == 1:
        classification10(path + '\\split', initial_params['number_of_samples'])
        kn_all, svm_all, dt_all = read_acc(path + '\\split\\0\\acc.txt')
        show_acc(len(kn_all), kn_all, svm_all, dt_all, title='для 10 признаков')

    e = 0   # Отображение данных классификации для 10 признаков
    if e == 1:
        for i in range(40):
            kn_all, svm_all, dt_all = read_acc(path + f'\\split\\{i}\\acc10.txt')
            show_acc(len(kn_all), kn_all, svm_all, dt_all, title=f'для 10 признаков изображение №{i}')

    f = 0  # Перещёт классификаторами для интервала в 20 признаков из участков по 10 из разных частей вектора признаков.
    if f == 1:
        classification20(path + '\\split', initial_params['number_of_samples'])

    g = 1  # Отображение тепловой карты
    if g == 1:
        all_kn, all_svm, all_dt, intervals = read_acc20(path + '\\split', initial_params['number_of_samples'])
        show_temp(all_kn, all_svm, all_dt, intervals, initial_params['number_of_samples'])


"""
     сценарий для обработки 1K изображений
"""


def scenario2(initial_params):
    path = initial_params['path']
    # Проверка метода с метрикой общая средняя точность
    check_for_comparison_psd(path + '\\datasets_psd', initial_params['number_of_samples'], initial_params['p'])


"""
     Определение номеров бэтта коэффициентов 
"""


def scenario3(initial_params):
    path = initial_params['path']
    data_to_frequencies(path + '\\true2', path + '\\false2', path + '\\datasets', initial_params['size_of_dataset'])


"""
     Эксперементы с классификацией
"""


def scenario4(initial_params):
    path = initial_params['path']
    classification_dct(path + '\\datasets', initial_params['number_of_samples'], initial_params['size_of_sample'],
                       initial_params['p'], initial_params['count_of_features'])


"""
     Проверка метода с метрикой общая средняя точность
"""


def scenario5(initial_params):
    path = initial_params['path']
    check_for_comparison_dct(path + '\\datasets', initial_params['number_of_samples'], initial_params['p'],
                             initial_params['count_of_features'])
