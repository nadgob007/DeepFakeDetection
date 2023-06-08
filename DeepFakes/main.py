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
    'number_of_samples': 5,                     # количество выборок(папок) по size_of_sample фотографий
    'p': 0.6,                                   # Процент тренировочной части выборки
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params3 = {
    'size_of_dataset': 10000,                   # размер каждого датасета
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params4 = {
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2\\datasets"  # путь до папки сохраненными значениями
}

initial_params5 = {
    'count_of_features': 10,                             # до скольки параметров взято
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2\\datasets"  # путь до папки сохраненными значениями
}

initial_params6 = {
    'number_of_samples': 59,                    # количество выборок(папок) по size_of_sample фотографий
    'size_of_sample': 1000,                     # колличество фотографий в выборке (папке)
    'p': 0.8,                                   # Процент тренировочной части выборки
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params7 = {
    'size_of_dataset': 1000,                    # размер каждого датасета
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}
# как initial_params5 замени!
initial_params9 = {
    'count_of_features': 10,                            # до скольки параметров взято
    'number_of_samples': 6,                             # количество выборок(папок) по size_of_sample фотографий
    'size_of_sample': 1000,                             # колличество фотографий в выборке (папке)
    'p': 0.8,                                           # Процент тренировочной части выборки
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2\\datasets"  # путь до папки со всеми файлами
}

initial_params10 = {
    'number_of_samples': 5,                     # количество выборок(папок) по size_of_sample фотографий
    'size_of_sample': 60000,                             # колличество фотографий в выборке (папке)
    'p': 0.6,                                   # Процент тренировочной части выборки
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

if __name__ == '__main__':
    # Начало
    start_time = datetime.now()
    print('Start in:', start_time)

    # scenario1(initial_params1)    # ДПФ

    # scenario2(initial_params2)    # test. Эксперементы с классификацией по всем данным 0.6 ДПФ

    # scenario3(initial_params3)    # Расчёт вектора средних бета коэфициентов для всех изображений

    # scenario4(initial_params4)    # Отображение графика вектора средних бета коэфициентов для всех наборов данных

    # scenario5(initial_params5)    # Расчёт и отображение графика значений хи квадрат для 1-10 коэф

    # scenario6(initial_params6)    # 1. Эксперементы с классификацией по выборкам 1000. окно 8. коэф 1-10

    # scenario7(initial_params7)    # Расчёт вектора бета коэфициентов для изображения с разным размером окна. коэф 1-10

    # scenario8(initial_params7)    # Отображение графика бетта коэфициентов с разной величиной окна

    scenario9(initial_params9)    # 2. Эксперементы с классификацией по выборкам 1000. разным размером окна. коэф 1-10

    #scenario10(initial_params10)  # test. Эксперементы с классификацией по всем данным 0.6

    # Конец
    print('Main выполнен за:', datetime.now() - start_time)
