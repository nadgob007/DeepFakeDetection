from datetime import datetime  # Время выполнения скрипта
from DeepFakes.scenarios import *

initial_params1 = {
    'all_images': 20000,                        # колличество всех фотографий
    'size_of_sample': 1000,                     # колличество фотографий в выборке (папке)
    'number_of_samples': 0,                     # колличество выборок(папок) по size_of_sample фотографий
    'p': 0.80,                                  # Процент тренировочной части выборки
    'count_of_features': 724,                   # Общее количество признаков для 1 изображения
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}
initial_params1['number_of_samples'] = int(initial_params1['all_images'] / initial_params1['size_of_sample'])

initial_params2 = {
    'all_images': 1000,                         # колличество всех фотографий
    'size_of_sample': 1000,                     # колличество фотографий в выборке (папке)
    'number_of_samples': 0,                     # колличество выборок(папок) по size_of_sample фотографий
    'p': 0.80,                                  # Процент тренировочной части выборки
    'count_of_features': 724,                   # Общее количество признаков для 1 изображения
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}
initial_params2['number_of_samples'] = int(initial_params2['all_images'] / initial_params2['size_of_sample'])

initial_params3 = {
    'size_of_dataset': 10000,                   # размер каждого датасета
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

initial_params4 = {
    'size_of_dataset': 10000,                   # колличество фотографий в датасете
    'number_of_samples': 30,                    # колличество выборок(папок) по size_of_sample фотографий
    'size_of_sample': 1000,                     # колличество фотографий в выборке (папке)
    'p': 0.80,                                  # Процент тренировочной части выборки
    'count_of_features': 5,                     # Общее количество признаков для 1 изображения ?
    'path': "E:\\NIRS\\Frequency\\Faces-HQ2"    # путь до папки со всеми файлами
}

if __name__ == '__main__':
    # Начало
    start_time = datetime.now()
    print('Start in:', start_time)

    # scenario1(initial_params1) # Для функции классификации дописать более полную точность

    # scenario2(initial_params2)

    # scenario3(initial_params3)

    # scenario4(initial_params4)

    # Конец
    print('Main выполнен за:', datetime.now() - start_time)
