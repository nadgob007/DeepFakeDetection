import os  # файлы
import re  # Регуляоные выражения
from datetime import datetime
import numpy as np
from skimage import color  # Отображение изображений
import matplotlib.pyplot as plt  # Графики
from skimage.io import imread, imshow, show  # Отображение изображений
from scipy.fft import fft2, fftfreq, fftshift, dct  # Преобразование фурье

"""
 Вычисление косинусного преобразования, μ – среднего по выборке и σ - среднеквадратического отклонения.
    Вход:
        img_nogrey - изображение
        is_zigzag - распологать ли элементы в зиг-заг
    Выход: 
        averages_beta - Вектор средних значений бетта коэффициентов 
"""


def cosinus_trans(img_nogrey, is_zigzag=True):
    img = imread(img_nogrey)
    img_grey = color.rgb2gray(img)  # Изображение в оттенках серого

    w, h = img_grey.shape
    f = 8
    count = int(w / f)

    blocks = []
    blocks_dct = []
    averages = [[] for i in range(f * f)]

    for i in range(count):
        for j in range(count):
            blocks.append(img_grey[i * f:(i * f + f), j * f:(j * f + f)])

            # Косинусное преобразование
            block = dct(img_grey[i * f:(i * f + f), j * f:(j * f + f)])
            if is_zigzag:
                block = zigzag(block)
                for k in range(len(block)):
                    averages[k].append(block[k])
            blocks_dct.append(block)    # Можно убрать если не нужны блоки

    # averages_m = [np.mean(i) for i in averages]
    averages_beta = [np.std(i) / (2 ** (1 / 2)) for i in averages]

    return averages_beta


"""
 Демонстрация косинусного преобразования
    Вход:
        img_nogrey - изображение
        is_zigzag - распологать ли элементы в зиг-заг
    Выход: нет
"""


def cosinus_trans_show(img_nogrey, is_zigzag=True):
    img = imread(img_nogrey)
    img_grey = color.rgb2gray(img)  # Изображение в оттенках серого
    w, h = img_grey.shape
    f = 8
    count = int(w / f)
    blocks = []
    blocks_dct = []
    averages = [[] for i in range(f * f)]

    for i in range(count):
        for j in range(count):
            blocks.append(img_grey[i * f:(i * f + f), j * f:(j * f + f)])

            # Косинусное преобразование
            block = dct(img_grey[i * f:(i * f + f), j * f:(j * f + f)])

            # модуль от косинусного перобразования надо ли ?
            # block = np.abs(block)
            if is_zigzag:
                block = zigzag(block)

                # Создать 2д массив из 1д
                # block = np.reshape(block, (-1, 8))
                for i in range(len(block)):
                    averages[i].append(block[i])
            blocks_dct.append(block)

    averages_m = [np.mean(i) for i in averages]
    averages_beta = [np.std(i) / 2 ** (1 / 2) for i in averages]

    # соединяем блоки в 1 изображение
    c = []
    for i in range(0, 128):
        b = c

        a = blocks_dct[i * 128]
        for j in range(1, 128):
            a = np.hstack([a, blocks_dct[i * 128 + j]])

        if i == 0:
            c = a.copy()
        else:
            c = np.vstack([b, a])

    fig = plt.figure(figsize=(8, 8))
    imshow(c, cmap='gray')

    dct2 = np.log(1 + np.abs(dct(img_grey)))
    # Простотранство для отображения
    fig = plt.figure(figsize=(15, 5))

    fig.add_subplot(2, 3, 1)
    plt.title("Изображение до обработки", fontsize=12)
    imshow(img)

    # В оттенках серого. Значения от 0 до 1
    fig.add_subplot(2, 3, 2)
    plt.title("Изображение в отенках серого", fontsize=12)
    imshow(img_grey)

    fig.add_subplot(2, 3, 3)
    imshow(dct2, cmap='gray')  # Отображать серым
    show()


"""
  Зиг-загом переписывает 2d массив в 1d массив, в порядке зиг-заг от левого верхнего к нижнему правому углу. 
    Вход: массив 
    Выход: массив в зиг-заг развёртке
"""


def zigzag(matrix):
    zigzag = []
    for index in range(1, len(matrix) + 1):
        slice = [i[:index] for i in matrix[:index]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag

    for index in range(1, len(matrix)):
        slice = [i[index:] for i in matrix[index:]]
        diag = [slice[i][len(slice) - i - 1] for i in range(len(slice))]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    return zigzag


""" 
 Сохраняет по указанному пути массив бетта с путем к изображению  
    Вход: 
        path - путь до файла сохранения, 
        name - название сохраняемого, 
        averages_beta - массив бета
    Выход: нет
"""


def dct_save(path, name, averages_beta):
    # Сохраняем 1 массив созданый по 1 изображению
    f = open(path + '\\' + 'dct.txt', 'a')
    line = ''
    line += str(name) + '\t' + str([i for i in averages_beta]) + '\n'
    f.write(line)
    f.close()


"""
 Считывает из указанного пути к файлу массив бетта и пути до изображений  
    Вход: path - путь до файла чтения
    Выход: averages - массив матриц beta
"""


def dct_read(path):
    file_path = path + '\\' + 'dct.txt'
    if not os.path.exists(file_path):
        f = open(file_path, 'a')
        f.close()

    f = open(file_path, 'r')
    line = '.'
    averages = []
    names = []
    while line:
        line = f.readline()
        if len(line) == 0:
            break
        result = re.split(r'\t', line)
        names.append(result.pop(0))
        result = result[0][1:-2]
        result = re.split(r', ', result)
        tmp = [float(i) for i in result]
        averages.append(tmp)
    f.close()
    return averages


"""
 Для переданного массива путей к изображениям расчитываются матрицы бетта и сохраняются.
 В случае, если в файле уже есть столько же бетта матриц сколько путей в массиве, то перерасчёт не производится 
    Вход:
        datasets - массив путей к изображениям
        istrue - передаётся  
        path - путь до папки для сохранения бетта матриц
    Выход: 
        beta_matrix - массив средних бетта матриц для каждого датасета  
        matrices_images - массив средних бетта для каждого изображения, для каждого датасета 
"""


def beta_matrix_of_images(images, path):
    beta_matrix = []        # Вектор для средних бетта коэффициентов по датасету
    matrices_images = []    # Массив векторов средних бетта коэффициентов для каждого изображения
    averages = []

    bookmark = len(dct_read(path))  # Изображение на котором остановились расчёты
    if len(images) > bookmark:
        count = bookmark
        remaining_part = images[bookmark:]

        for i in remaining_part:
            averages_beta = cosinus_trans(i)    # Расчёт вектора бета коэфициентов для 1го изображения
            dct_save(path, i, averages_beta)    # Сохранение вектора бета коэфициентов в файл

            for j in range(len(averages_beta)):
                averages.append(averages_beta[j])

            print(count, '/', len(images))
            count += 1

    averages = dct_read(path)               # Считываем из файла без расчётов, если всё посчитано
    matrices_images.append(averages)

    transpose_averages = [*zip(*averages)]
    beta = [np.mean(j) for j in transpose_averages]     # Считаем среднее по всем изображениям в датасете
    beta_matrix.append(beta)

    return beta_matrix, matrices_images


"""
 Строит график средних значений бетта коэффициентов для каждого датасета
    Вход: 
        beta_true - массив бетта матриц для каждого настоящего датасета,
        beta_false - массив бетта матриц для каждого поддельного датасета, 
        true - названия настоящих датасетов, 
        false - названия поддельных датасетов.
    Выход: отсутствует
"""


def show_beta_statistic(beta_true):
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')

    x = beta_true[0][1:]
    y = [j for j in range(1, 64)]
    ax.plot(y, x, label='CelebA(Настоящие)')

    ax.set_xlabel('Номер β коэффициента')
    ax.set_ylabel('Значение β коэффициента')
    ax.set_title("График значений β для каждого задействованного набора данных")
    ax.legend()
    show()


def data_to_frequencies(path_true, path):
    # 1. Получаем массив путей до файлов картинок, по указанному пути
    true = [os.path.join(path_true, image) for image in os.listdir(path_true)]

    # 2. Для каждого изображения высчитываем и сохраняем матрицу средних бетта коэффициентов
    beta_true, matrices_true = beta_matrix_of_images(true, path)

    # 3. Отображаем для датасета график со средними бетта коэффициентами
    show_beta_statistic(beta_true)


def scenario5(initial_params):
    path = initial_params['path']
    data_to_frequencies(path + '\\true', path)


if __name__ == '__main__':
    initial_params5 = {
        'path': "exsample"  # Путь до папки exsample
    }

    # Начало
    start_time = datetime.now()
    print('Start in:', start_time)

    scenario5(initial_params5)

    # Конец
    end_time = datetime.now() - start_time
    print('Main выполнен за:', end_time)
