from datetime import datetime  # Время выполнения скрипта
from DeepFakes.scenarios import *

initial_params1 = {
    'number_of_samples': 40,                    # количество выборок(папок) по size_of_sample фотографий
    'size_of_sample': 1000,                     # количество фотографий в выборке (папке)
    'count_of_features': 724,                   # Общее количество признаков для 1 изображения
    'p': 0.8,                                   # Процент тренировочной части выборки
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params2 = {
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params3 = {
    'size_of_dataset': 10000,                   # размер каждого датасета
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params4 = {
    'size_of_dataset': 10000,                   # количество фотографий в датасете
    'number_of_samples': 30,                    # количество выборок(папок) по size_of_sample фотографий
    'size_of_sample': 1000,                     # колличество фотографий в выборке (папке)
    'count_of_features': 5,                     # общее количество признаков для 1 изображения
    'p': 0.8,                                   # Процент тренировочной части выборки
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params5 = {
    'number_of_samples': 5,                     # количество выборок(папок) по size_of_sample фотографий
    'count_of_features': 5,                     # Общее количество признаков для 1 изображения
    'p': 0.6,                                   # Процент тренировочной части выборки
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

if __name__ == '__main__':
    # Начало
    start_time = datetime.now()
    print('Start in:', start_time)

    scenario1(initial_params1)      # ДПФ

    # scenario2(initial_params2)    # Перенос psds

    # scenario3(initial_params3)    # Определение номеров бэтта коэффициентов

    # scenario4(initial_params4)    # Эксперементы с классификацией по выборкам 1000

    # scenario5(initial_params5)    # Эксперементы с классификацией по всем данным 0.6

    # Конец
    print('Main выполнен за:', datetime.now() - start_time)
