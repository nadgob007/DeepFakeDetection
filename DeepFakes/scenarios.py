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
     Эксперементы с классификацией по всем данным 0.6 ДПФ
"""


def scenario2(initial_params):
    path = initial_params['path']
    # Проверка метода с метрикой общая средняя точность
    check_for_comparison_psd(path + '\\datasets_psd', initial_params['number_of_samples'], initial_params['p'])


"""
_______________________________________________________________________________________________________________________
     Определение номеров бэтта коэффициентов 
"""


def scenario3(initial_params):
    path = initial_params['path']
    data_to_frequencies(path + '\\true2', path + '\\false2', path + '\\datasets', initial_params['size_of_dataset'])


# Отображение графика с бетта коэффициентами для каждого датасета
def scenario4(initial_params):
    path = initial_params['path']
    beta_true, matrices_true = read_beta_matrix_of_images(True, path, file_name=f'dct.txt')
    beta_false, matrices_false = read_beta_matrix_of_images(False, path, file_name=f'dct.txt')

    true_datasets_names = [path + '\\true\\' + dirpath for dirpath in os.listdir(path + '\\true\\')]
    false_datasets_names = [path + '\\false\\' + dirpath for dirpath in os.listdir(path + '\\false\\')]

    # отображение графика с бетта коэффициентами для каждого датасета
    show_beta_statistic(beta_true, beta_false, true_datasets_names, false_datasets_names)


# Отображение графика значений хи квадрат
def scenario5(initial_params, file_read=f'dct.txt', file_save='special_beta.txt'):
    path = initial_params['path']
    beta_true, matrices_true = read_beta_matrix_of_images(True, path, file_name=file_read)
    beta_false, matrices_false = read_beta_matrix_of_images(False, path, file_name=file_read)

    # 4. Вычисляем расстояние χ
    datasets = matrices_true + matrices_false
    x = [[] for i in datasets]
    special_beta = []

    for f in range(1, initial_params['count_of_features']+1):
        count = 0
        # fig, ax = plt.subplots(6, 6)
        all_beta = []
        for i in datasets:
            line = datasets.index(i)
            for j in datasets:
                col = datasets.index(j)
                arr = x_squer(i, j)
                arr = multi_argmax(arr, f)
                x[line].append(arr)
                count += 1
                # print(f'{count}/{len(datasets) * len(datasets)}')
                a = arr
                x1 = [i[0] for i in a]
                y1 = [i[1] for i in a]

                # Только расстояни для T->F и F->T
                if (line < 3 and col > 2) or (line > 2 and col < 3):
                    all_beta = all_beta + x1

                plt.ylim()
                # ax[line, col].bar(x1, y1, width=0.2, label='KN')
                # ax[line, col].legend(loc="upper left")

                # ax[line, col].set_title(f'хи квадрат {line+1} и {col+1}]')
                # ax[line, col].set_facecolor('seashell')
                # fig.set_figwidth(12)  # ширина Figure
                # fig.set_figheight(6)  # высота Figure
                # fig.set_facecolor('floralwhite')
                #plt.show()
        print(f'{f}', np.unique(all_beta), len(np.unique(all_beta)))
        special_beta.append([f, np.unique(all_beta)])

    for i in special_beta:
        dct_save(path, i[0], i[1], file_name=file_save)


"""
     Эксперементы с классификацией
"""


# 1. Эксперементы. 60 выборок по 1000. окно 8. коэф 1-10
def scenario6(initial_params):
    path = initial_params['path']
    classification_dct(path + '\\datasets', initial_params['number_of_samples'], initial_params['size_of_sample'],
                       initial_params['p'])


# Расчёт вектора бета коэфициентов для изображения с разным размером окна. коэф 1-10
def scenario7(initial_params):
    path = initial_params['path']
    data_to_frequencies1(path + '\\true2', path + '\\false2', path + '\\datasets', initial_params['size_of_dataset'])


# Отображение графика бетта коэфициентов с разной величиной окна
def scenario8(initial_params):
    path = initial_params['path'] + '\\datasets'
    sizes = [8 * (2 ** i) for i in range(1, 5)]
    for block_size in sizes:
        print(f'Размер блоков: {block_size}')
        beta_true, matrices_true = read_beta_matrix_of_images(True, path, file_name=f'dct{block_size}.txt')
        beta_false, matrices_false = read_beta_matrix_of_images(False, path, file_name=f'dct{block_size}.txt')

        true_datasets_names = [path + '\\true\\' + dirpath for dirpath in os.listdir(path + '\\true\\')]
        false_datasets_names = [path + '\\false\\' + dirpath for dirpath in os.listdir(path + '\\false\\')]

        # Отображение графика с бетта коэффициентами для каждого датасета
        show_beta_statistic(beta_true, beta_false, true_datasets_names, false_datasets_names)


# Эксперементы с классификацией по выборкам 1000. с разным размером окна. коэф 1-10
def scenario9(initial_params):
    sizes = [8 * (2 ** i) for i in range(1, 5)]

    # for block_size in sizes:
    #     scenario5(initial_params, file_read=f'dct{block_size}.txt', file_save=f'special_beta{block_size}.txt')

    for block_size in sizes:
        if block_size == 8:
            classification_dct(initial_params['path'],
                               initial_params['number_of_samples'],
                               initial_params['size_of_sample'],
                               initial_params['p'], file_read=f'dct.txt',
                               file_special=f'special_beta.txt')
        classification_dct(initial_params['path'],
                           initial_params['number_of_samples'],
                           initial_params['size_of_sample'],
                           initial_params['p'], file_read=f'dct{block_size}.txt', file_special=f'special_beta{block_size}.txt')


"""
     Проверка метода с метрикой общая средняя точность
"""


def scenario10(initial_params):
    path = initial_params['path']
    check_for_comparison_dct(path + '\\datasets',
                             initial_params['number_of_samples'],
                             initial_params['size_of_sample'],
                             initial_params['p'])
