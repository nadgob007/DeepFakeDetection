import os       # файлы
import re       # Регуляоные выражения
import copy     # глубокое копирование объекта
import numpy as np
from skimage import color  # Отображение изображений
import matplotlib.pyplot as plt  # Графики
from skimage.io import imread, imshow, show  # Отображение изображений
from scipy.fft import fft2, fftfreq, fftshift, dct  # Преобразование фурье
# from sklearn.metrics import classification_report   # отчёт о классификации
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split  # Разбиение данных на обучение и тестирования
from sklearn.neighbors import KNeighborsClassifier  # Классификация ближайших соседей
# from sklearn.pipeline import make_pipeline  # Классификация векторов поддержки С
# from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score  # Классификатор дерева решений
from sklearn.tree import DecisionTreeClassifier
# from datetime import datetime  # Время выполнения скрипта
# import time


"""
    Сторонние функции азимутального усреднения
"""


def azimuthalAverage(image, center=None, stddev=False, median=False, returnradii=False, return_nr=False,
                     binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None,
                     mask=None):
    """
    Calculate the azimuthally averaged radial profile.
    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.
    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...
    :param median:

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape, dtype='bool')
    # obsolete elif len(mask.shape) > 1:
    # obsolete     mask = mask.ravel()

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    # nr = np.bincount(whichbin)[1:]
    nr = np.histogram(r, bins, weights=mask.astype('int'))[0]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat, bins)
        # This method is still very slow; is there a trick to do this with histograms?
        radial_prof = np.array([image.flat[mask.flat * (whichbin == b)].std() for b in range(1, nbins + 1)])
    else:
        if median:
            w, h = r.shape
            med = [np.array([]) for i in range(maxbin)]
            for i in range(h - 1):
                for j in range(w - 1):
                    a = int(np.round(r[i][j]))
                    b = image[i][j]
                    med[a] = np.append(med[a], [b])

            radial_prof = np.empty((maxbin))

            for i in range(1, len(med)):
                a = med[i]
                radial_prof[i - 1] = np.median(med[i])
        else:
            radial_prof = np.histogram(r, bins, weights=(image * weights * mask))[0] / \
                          np.histogram(r, bins, weights=(mask * weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers, bin_centers[radial_prof == radial_prof],
                                radial_prof[radial_prof == radial_prof], left=left, right=right)

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(radial_prof, radial_prof)).ravel()
        return xarr, yarr
    elif returnradii:
        return bin_centers, radial_prof
    elif return_nr:
        return nr, bin_centers, radial_prof
    else:
        return radial_prof


"""
 Вычисляет psd1D. 
    Вход: изображения 
    Выход: psd1D (массив признаков)
"""


# calculations -> calculate_features
def calculate_features(img_nogrey, only_psd=True, isavg=False):
    try:
        img = imread(img_nogrey)  # Цветное изображение
    except:
        f = open('err.txt', 'a')
        f.write(img_nogrey)
        f.close()
        return 0, 0, [], []
    else:
        print('Исключений не произошло')

    img_grey = color.rgb2gray(img)  # Изображение в оттенках серого

    # Быстрое преобразование Фурье FFT
    fft2 = np.fft.fft2(img_grey)  # Использование FFT

    # Перемещение картинки в центр и использование модуля. Спектрально-логарифмическое преобразование
    # 1 + чтоб значения были от 0.Модуль перевод из комплексного

    fft2 = np.fft.fftshift(np.log(1 + np.abs(fft2)))
    # fft2 = np.fft.fftshift(1 + np.abs(fft2))  # Хуже работает

    # Добавить возможность деления на сумму усреднения
    if isavg:
        fft2 = fft2 / sum(fft2, fft2[0])
    psd1D = azimuthalAverage(fft2, binsize=1, median=True)

    if only_psd:
        return psd1D
    else:
        return img, img_grey, fft2, psd1D


""" 
 Рисует Спектрограмму и Азимутальное усреднение для входящего изображения 
    Вход: изображение
    Выход: отсутствует 
"""


def show_img(img_nogrey, isavg):
    img, img_grey, fft2, psd1D = calculate_features(img_nogrey, isavg)

    # Простотранство для отображения
    fig = plt.figure(figsize=(15, 5))

    # Цветная. Значения от 0 до 255
    fig.add_subplot(2, 3, 1)
    plt.title("Изображение до обработки", fontsize=12)
    imshow(img)

    # В оттенках серого. Значения от 0 до 1
    fig.add_subplot(2, 3, 2)
    plt.title("Изображение в отенках серого", fontsize=12)
    imshow(img_grey)

    # Быстрое преобразование Фурье FFT. Значения
    fig.add_subplot(2, 3, 3)
    plt.ylabel('Амплитуда', fontsize=10)
    plt.xlabel('Частота', fontsize=10)
    plt.title("Спектрограмма", fontsize=12)
    imshow(fft2, cmap='gray')  # Отображать серым

    # Азимутальное усреднение
    fig.add_subplot(2, 1, 2)
    plt.plot(psd1D, color='green', linestyle='-', linewidth=1, markersize=4)
    plt.ylabel('Энергетический спектр', fontsize=10)
    plt.xlabel('пространственная частота', fontsize=10)
    plt.title("Азимутальное усреднение", fontsize=12)

    plt.tight_layout()
    show()

    return 0


""" 
 Сохраняет массивы признаков и классов psd1D в текстовый файл и возвращает сколько строк сохранил
    Вход: путь до файла сохранения, массив признаков, массив классов, название файлов
    Выход: сколько строк сохранил
"""


def psd_save(path, name, psd, file_name='psd.txt'):
    # Сохраняем 1 массив созданый по 1 изображению
    f = open(path + '\\' + file_name, 'a')
    line = ''
    line += str(name) + '\t' + str([i for i in psd]) + '\n'
    f.write(line)
    f.close()


"""
 Читает файл c признаками и возвращает массив признаков и классов
    Вход: путь до файла чтения
    Выход: x - массив признаков, y - массив классов
"""


def psd_read(path, file_name='\\psd.txt'):
    file_path = path + file_name
    if not os.path.exists(file_path):
        f = open(file_path, 'a')
        f.close()

    f = open(file_path, 'r')
    line = '.'
    psd = []
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
        psd.append(tmp)
    f.close()
    return psd


"""
 Классифицирует полученные из файла массивы признаков по выборкам train и test
    Вход: путь до папки,
    Выход: точность для KN, SVM, DT 
"""


def classifier(x_train, y_train, x_test, y_test):
    y_train = [int(i[0]) for i in y_train]
    y_test = [int(i[0]) for i in y_test]

    # Классификация ближайших соседей
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)
    predicts_kn = neigh.predict(x_test)
    accuracy_kn = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_kn[i]:
            accuracy_kn += 1
    # cr = classification_report(y_test, predicts_kn)
    # print(cr)

    # Классификация векторов поддержки С радиальной базисной функции
    clf = SVC(kernel='rbf', gamma='auto')
    clf.fit(x_train, y_train)
    predicts_svm = clf.predict(x_test)
    accuracy_svm = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_svm[i]:
            accuracy_svm += 1
    # cr = classification_report(y_test, predicts_svm)
    # print(cr)

    # Классификатор дерева решений
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    predicts_dt = clf.predict(x_test)
    accuracy_dt = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_dt[i]:
            accuracy_dt += 1
    # cr = classification_report(y_test, predicts_dt)
    # print(cr)

    return accuracy_kn, accuracy_svm, accuracy_dt


"""
 Создает файла со статистикой точности
    Вход: путь до файла, точности для KN, SVM, DT.
    Выход: количество строк
"""


def accuracy_save(path, kn, svm, dt):
    # Формируем шапку таблицы в acc.txt
    f = open(path, 'w')
    form = '№\t KN(%)\t SVM(%)\t DT(%)\n'
    f.write(form)
    line = ''
    rows = 0
    one_procent = len(kn)/100
    for i in range(len(kn)):
        line += f'{i}\t {kn[i] / one_procent}\t {svm[i] / one_procent}\t {dt[i] / one_procent}\n'
        f.write(line)
        line = ''
        rows = i
    f.close()
    return rows


"""
 Читает файл со статистикой по всем 40 выборкам. Возвращает массив точностей по каждой выборке для каждого классификатор
    Вход: путь до файла чтения
    Выход: массивы точностей для KN, SVM, DT
"""


def read_acc(path):
    f = open(path, 'r')
    line = f.readline()  # Игнорируем шапку
    all_kn = []
    all_svm = []
    all_dt = []
    while line:
        line = f.readline()
        if len(line) == 0:
            break
        result = re.split(r'\t ', line)
        all_kn.append(float(result[1]))
        all_svm.append(float(result[2]))
        all_dt.append(float(result[3]))
    f.close()

    return all_kn, all_svm, all_dt


"""
 Читает файл со статистикой по всем 40 выборкам. Возвращает массив точностей по каждой выборке для каждого классификатора
    Вход: путь до файла чтения, колличество папок
    Выход: массивы точностей для KN, SVM, DT и интервал признаков (сколько признаков)
"""


def read_acc20(path, number_of_folders):
    all_kn = []
    all_svm = []
    all_dt = []
    intervals = []
    for i in range(number_of_folders):
        f = open(path + f'\\{i}\\acc20.txt', 'r')
        line = f.readline()  # Игнорируем шапку
        kn = []
        svm = []
        dt = []
        while line:
            line = f.readline()
            if len(line) == 0:
                break
            result = re.split(r'\t ', line)
            if i == 0:
                intervals.append(result[0])
            kn.append(float(result[1]))
            svm.append(float(result[2]))
            dt.append(float(result[3]))
        f.close()
        all_kn.append(kn)
        all_svm.append(svm)
        all_dt.append(dt)
        print(i)

    return all_kn, all_svm, all_dt, intervals


"""
 Строит тепловые карты для каждого массива точностей 
    Вход: массивы точностей для KN, SVM, DT и интервал признаков (сколько признаков), колличество папок
    Выход: отсутствует
"""


def show_temp(all_kn, all_svm, all_dt, intervals, number_of_folders):
    kn = []
    svm = []
    dt = []
    for j in range(len(all_kn[0])):
        mean_kn = []
        mean_svm = []
        mean_dt = []
        for i in range(number_of_folders):
            mean_kn.append(all_kn[i][j])
            mean_svm.append(all_svm[i][j])
            mean_dt.append(all_dt[i][j])
        kn.append(np.mean(mean_kn))
        svm.append(np.mean(mean_svm))
        dt.append(np.mean(mean_dt))

    a = [[] for i in range(71)]
    b = [[] for i in range(71)]
    c = [[] for i in range(71)]

    count = 0
    for i in range(72):
        count += i

    k = 0
    min_a = 100
    min_b = 100
    min_c = 100
    max_a = 0
    max_b = 0
    max_c = 0
    for j in range(71):
        for i in range(71):
            if i < j:
                a[j].append(0)
                b[j].append(0)
                c[j].append(0)
            else:
                break

        for i in range(j, 71):
            a[j].insert(i, int(kn[k]))
            b[j].insert(i, int(svm[k]))
            c[j].insert(i, int(dt[k]))

            if kn[k] < min_a:
                min_a = kn[k]
            if svm[k] < min_b:
                min_b = svm[k]
            if dt[k] < min_c:
                min_c = dt[k]

            if kn[k] > max_a:
                max_a = kn[k]
            if svm[k] > max_b:
                max_b = svm[k]
            if dt[k] > max_c:
                max_c = dt[k]
            i += 1
            k += 1
        print(j)

    fig = plt.figure(figsize=(15, 5))

    # KN
    fig.add_subplot(1, 3, 1)
    plt.title(f"Kn (min/max)\n{min_a}-{max_a}", fontsize=12)
    plt.matshow(a, 0)
    fig.colorbar(plt.matshow(a, 0), orientation='vertical', fraction=0.04)
    plt.clim(0, 100)

    # SVM
    fig.add_subplot(1, 3, 2)
    plt.title(f"SVM (min/max)\n{min_b}-{max_b}", fontsize=12)
    plt.matshow(b, 0)
    fig.colorbar(plt.matshow(b, 0), orientation='vertical', fraction=0.04)
    plt.clim(0, 100)

    # DT
    fig.add_subplot(1, 3, 3)
    plt.title(f"DT (min/max)\n{min_c}-{max_c}", fontsize=12)
    plt.matshow(c, 0)
    fig.colorbar(plt.matshow(c, 0), orientation='vertical', fraction=0.04)
    plt.clim(0, 100)

    plt.show()


"""
 Отображает график для общей статистики точности каждого классификатора
    Вход: колличество выборок, массивы точностей для KN, SVM, DT
    Выход: отсутствует
"""


def show_acc(num, all_kn, all_svm, all_dt, title):
    #  Задаем смещение равное половине ширины прямоугольника:
    x1 = np.arange(0, num) - 0.3
    x2 = np.arange(0, num) + 0
    x3 = np.arange(0, num) + 0.3

    mins = [round(min(all_kn), 1), round(min(all_svm), 1), round(min(all_dt), 1)]
    maxs = [round(max(all_kn), 1), round(max(all_svm), 1), round(max(all_dt), 1)]
    print(f'Наименьшее:{mins}')

    avg_str = ''
    if num > 1:
        avg = [round(np.mean(all_kn), 1), round(np.mean(all_svm), 1), round(np.mean(all_dt), 1)]
        avg_str = f'\navg kn:{avg[0]},    svm:{avg[1]},   dt:{avg[2]}'

    y1 = [all_kn[i] for i in range(len(all_kn))]
    y2 = [all_svm[i] for i in range(len(all_svm))]
    y3 = [all_dt[i] for i in range(len(all_dt))]

    # y_masked = np.ma.masked_where(int(y1) < 50, y1)

    fig, ax = plt.subplots()
    plt.ylim(min(mins)-((100-min(mins))/100)*10, 100)

    ax.bar(x1, y1, width=0.2, label='KN')
    ax.bar(x2, y2, width=0.2, label='SVM', color='orange')
    ax.bar(x3, y3, width=0.2, label='DT', color='green')

    ax.legend(loc="upper left")

    ax.set_title(f'Точность KN, SVM, DT {title}\n '
                 f'min/max kn:{mins[0]} / {maxs[0]},    svm:{mins[1]}/{maxs[1]},   dt:{mins[2]}/{maxs[2]}' + avg_str)
    ax.set_facecolor('seashell')
    fig.set_figwidth(12)  # ширина Figure
    fig.set_figheight(6)  # высота Figure
    fig.set_facecolor('floralwhite')

    plt.show()

    return 0


"""
 Сохраняет значения классификаторов для 10 или 20 признаков 
    Вход: путь до вайла сохранения, массивы точностей для KN, SVM, DT, интервалы признаков, режим 10 или 20
    Выход: сохраненные строки
"""


def save_in_1K(path, kn, svm, dt, intervals, mode):
    # Формируем шапку таблицы в acc[№].txt
    f = open(path, 'w')
    form = 'Interval\t KN(%)\t SVM(%)\t DT(%)\n'
    f.write(form)
    line = ''
    rows = 0
    j = 0
    one_procent = len(kn) / 100
    for i in range(len(kn)):
        if mode == 10:
            line += f'{intervals[j][0]}-{intervals[j][1]}\t {kn[i] / one_procent}\t {svm[i] / one_procent}\t {dt[i] / one_procent}\n'
        elif mode == 20:
            line += f'{intervals[j][0]}-{intervals[j][1]}:{intervals[j + 1][0]}-{intervals[j + 1][1]}\t {kn[i] / one_procent}\t {svm[i] / one_procent}\t {dt[i] / one_procent}\n'
        f.write(line)
        line = ''
        rows = i
        if mode == 10:
            j += 1
        elif mode == 20:
            j += 2
    f.close()
    return rows


"""
 Составляет массивы путей до настоящих и поддельных изображений. получаем массив путей до файлов картинок.
    Вход: 
        n - размер выборки, 
        path_true и path_false - пути до папок с настоящими и поддельными изображениями
    Выход: половины от массивов путей до настоящих и поддельных изображений.
"""


def get_data_list(path_true, path_false):
    true_datasets = [[os.path.join(path_true, dirpath)] for dirpath in os.listdir(path_true)]
    for j in true_datasets:
        true_items = []
        for dirpath, dirnames, filenames in os.walk(j[0]):
            if not (len(filenames) == 0):
                for i in filenames:
                    true_items.append(dirpath + '\\' + i)
        j.append(true_items)

    false_datasets = [[os.path.join(path_false, dirpath)] for dirpath in os.listdir(path_false)]
    for j in false_datasets:
        false_items = []
        for dirpath, dirnames, filenames in os.walk(j[0]):
            if not (len(filenames) == 0):
                for i in filenames:
                    false_items.append(dirpath + '\\' + i)
        j.append(false_items)

    return true_datasets, false_datasets


"""
 Примнимает список имен используемых изображений, вычисляет массивы признаков и записывает в файл
    Вход: 
        list_allK1 - массив объектов содержащих, имена путей до изображений и обозначения (0 или 1) 
        path_folder - путь до папки split (в которой будут лежать папки с выборками)
    Выход: x_train/x_test - вектора признаков, y_train/y_test - вектора классов
"""


def list2psd1_2(datasets, istrue, path):
    which = '\\false\\'
    if istrue:
        which = '\\true\\'

    psds = []
    for dataset in datasets:
        psds1 = [[] for i in range(len(dataset[1]))]

        # Путь до папки датасета
        path_folder = path + which + os.path.basename(os.path.normpath(dataset[0]))
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        bookmark = len(psd_read(path_folder))
        if len(dataset[1]) > bookmark:
            count = bookmark
            dataset_current = dataset[1][bookmark:]
            for i in dataset_current:
                psd1D = calculate_features(i)

                psd_save(path_folder, i, psd1D)

                psds1.append(psd1D)

                print(count, '/', len(dataset[1]))
                count += 1
        psds1 = psd_read(path_folder)
        psds.append(psds1)

    return psds


"""
 1. получаем массив путей до файлов картинок и оставляем только n/2 от каждого.
 2. помечаем 1 - true, 0 - false
 3. соединяем и перемешиваем.
 4. разбиваем массив на n/sample папок (20)
 5. Получаем массивы признаков и классов для изображений
    Вход: 
        n - размер выборки
        sample - размер 1ой выборки
        tf - train test соотношение
        path_true, path_false - пути до папок true false
        path - куда сохранить psd
    Выход: train.txt и test.txt файлы с путями до *.png файлов
"""


def data_to_psd(size_of_sample, number_of_samples, p, path_true, path_false, path):
    size_of_dataset = 10000
    # 1. создаем массив путей до каждого изображения, для каждого набора данных.
    true, false = get_data_list(path_true, path_false)

    # Приводим датасеты к одному размеру
    for i in range(len(true)):
        true[i][1] = true[i][1][:size_of_dataset]

    for i in range(len(false)):
        false[i][1] = false[i][1][:size_of_dataset]

    # 2. Вычисляем для всех изображений psd.
    psds_true = list2psd1_2(true, True, path + '\\datasets_psd')
    psds_false = list2psd1_2(false, False, path + '\\datasets_psd')

    true_datasets_names = [dirpath for dirpath in os.listdir(path + '\\datasets_psd' + '\\true\\')]
    false_datasets_names = [dirpath for dirpath in os.listdir(path + '\\datasets_psd' + '\\false\\')]

    # Выбираем какие наборы данных будут участвовать
    for i in true_datasets_names:
        if i == 'images1024x1024':
            psds_true.pop(true_datasets_names.index(i))
    for i in false_datasets_names:
        if i == '1m_faces_01':
            psds_false.pop(false_datasets_names.index(i))

    # 3. Создание выборок для классификации в количестве number_of_samples
    samples = []
    for j in range(number_of_samples):
        true_part_sample = []
        number_of_true_false = int(size_of_sample / 2)  # Колличество настоящих и поддельных в 1ой выборке
        a = number_of_true_false / len(psds_true)   # если число не целое
        int_a = int(a)
        if not (int_a % 2 == 0):
            b = int((a - int_a) * len(psds_true)) + 1
        else:
            b = 0
        for i in psds_true:
            if b - 1 < 0:
                true_part_sample += i[int_a * j: int_a * j + int_a]
            else:
                true_part_sample += i[(int_a + 1) * j:(int_a + 1) * j + (int_a + 1)]
                b -= 1

        false_part_sample = []
        number_of_true_false = int(size_of_sample / 2)  # Колличество настоящих и поддельных в 1ой выборке
        a = number_of_true_false / len(psds_false)  # если число не целое
        int_a = int(a)
        if not (int_a % 2 == 0):
            b = int((a - int_a) * len(psds_false)) + 1
        else:
            b = 0
        for i in psds_false:
            if b - 1 < 0:
                false_part_sample += i[int_a * j: int_a * j + int_a]
            else:
                false_part_sample += i[(int_a + 1) * j:(int_a + 1) * j + (int_a + 1)]
                b -= 1

        # соединяем и перемешиваем.
        x = true_part_sample + false_part_sample
        y = [[1] for i in range(number_of_true_false)] + [[0] for i in range(number_of_true_false)]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p, random_state=42)
        samples.append([])
        samples[j].append(x_train)
        samples[j].append(y_train)
        samples[j].append(x_test)
        samples[j].append(y_test)
        print(j)

    # 4. Сохранение выборок по отдельным папкам
    for i in samples:
        path_folder = f'\\split\\{samples.index(i)}'
        for j in i:
            file_name = ''
            if i.index(j) == 0:
                file_name = 'x_train.txt'
                f = open(path + path_folder + '\\' + file_name, 'w')    # стираем старые данные
            elif i.index(j) == 1:
                file_name = 'y_train.txt'
                f = open(path + path_folder + '\\' + file_name, 'w')    # стираем старые данные
            elif i.index(j) == 2:
                file_name = 'x_test.txt'
                f = open(path + path_folder + '\\' + file_name, 'w')    # стираем старые данные
            else:
                file_name = 'y_test.txt'
                f = open(path + path_folder + '\\' + file_name, 'w')    # стираем старые данные
            for c in j:
                psd_save(path + path_folder, j.index(c), c, file_name)
        print(samples.index(i))
    return 0


def check_for_comparison_psd(path, number_of_samples, p):
    # 1. получаем массив путей до файлов картинок.
    true = []
    true_datasets = [os.path.join(path + '\\true', dirpath) for dirpath in os.listdir(path + '\\true')]
    for i in true_datasets:
        true.append(psd_read(i))

    false = []
    false_datasets = [os.path.join(path + '\\false', dirpath) for dirpath in os.listdir(path + '\\false')]
    for i in false_datasets:
        false.append(psd_read(i))

    x = []
    y = []
    for i in true:
        x += i
        y += [1 for j in range(len(i))]
    for i in false:
        x += i
        y += [0 for j in range(len(i))]

    KN = []
    SVM = []
    DT = []
    for i in range(number_of_samples):
        sample = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p, random_state=42 + i)

        # 4. Классификация и расчёт точности
        kn, svm, dt = classifier_dct(x_train, y_train, x_test, y_test)
        KN.append(kn)
        SVM.append(svm)
        DT.append(dt)
        print(f'№{i + 1}')
        print(kn, svm, dt)

    # 5. Отрисовка граффиков точности классификации
    one_procent = len(y_test) / 100

    show_acc(number_of_samples, [i / one_procent for i in KN], [i / one_procent for i in SVM],
             [i / one_procent for i in DT], title=f' 5 выборок')



"""
[? зачем принт]
 Классифицирует все выборки, сохраняет точности по каждой выборке и выводит в консоль 
    Вход: 
        path - куда сохранить psd
        number_folders - количество выборок(папок)
        interval - колличество признаков которое будет использоваться при классификации
    Выход:
"""


def classification(path, number_folders, size_of_sample, interval):
    all_kn = []
    all_svm = []
    all_dt = []

    for i in range(number_folders):
        x_train = psd_read(path + f'\\{i}', '\\x_train.txt')
        y_train = psd_read(path + f'\\{i}', '\\y_train.txt')
        x_test = psd_read(path + f'\\{i}', '\\x_test.txt')
        y_test = psd_read(path + f'\\{i}', '\\y_test.txt')

        kn, svm, dt = classifier(x_train, y_train, x_test, y_test, interval)
        all_kn.append(kn)
        all_svm.append(svm)
        all_dt.append(dt)
        print('classified', i)
    accuracy_save(path + '\\acc.txt', all_kn, all_svm, all_dt, size_of_sample)


"""
 Классифицирует указанное количество выборок используя 10 признаков, сохраняет точности по каждой выборке
    Вход:
        path - куда сохранить psd
        number_folders - количество выборок(папок)
    Выход:
"""


def classification10(path, number_of_folders):
    for j in range(number_of_folders):
        all_kn = []
        all_svm = []
        all_dt = []
        intervals = []

        x_train = psd_read(path + f'\\{j}', '\\x_train.txt')
        y_train = psd_read(path + f'\\{j}', '\\y_train.txt')
        x_test = psd_read(path + f'\\{j}', '\\x_test.txt')
        y_test = psd_read(path + f'\\{j}', '\\y_test.txt')

        for i in range(0, 720, 10):
            interval = []
            if i == 710:
                interval.append([i, i + 14])
            else:
                interval.append([i, i + 10])

            start = interval[0]
            x_train_crop = [psd[start[0]:start[1]] for psd in x_train]
            x_test_crop = [psd[start[0]:start[1]] for psd in x_test]

            kn, svm, dt = classifier(x_train_crop, y_train, x_test_crop, y_test)
            all_kn.append(kn)
            all_svm.append(svm)
            all_dt.append(dt)
            intervals.append(interval[0])
        print(f'Выборка:{j}')
        save_in_1K(path + f'\\{j}\\acc10.txt', all_kn, all_svm, all_dt, intervals, mode=10)


"""
 Классифицирует указанное количество выборок используя 20 признаков, сохраняет точности по каждой выборке
    Вход:
        path - куда сохранить psd
        number_folders - количество выборок(папок)
    Выход:
"""


def classification20(path, number_folders):
    for j in range(number_folders):
        all_kn = []
        all_svm = []
        all_dt = []
        intervals = []

        x_train = psd_read(path + f'\\{j}', '\\x_train.txt')
        y_train = psd_read(path + f'\\{j}', '\\y_train.txt')
        x_test = psd_read(path + f'\\{j}', '\\x_test.txt')
        y_test = psd_read(path + f'\\{j}', '\\y_test.txt')

        for i in range(0, 720, 10):
            for k in range(10 + i, 720, 10):
                interval = [[i, i + 10]]
                if k == 710:
                    interval.append([k, k + 14])
                else:
                    interval.append([k, k + 10])

                start = interval[0]
                end = interval[1]
                x_train_crop = [psd[start[0]:start[1]] + psd[end[0]:end[1]] for psd in x_train]
                x_test_crop = [psd[start[0]:start[1]] + psd[end[0]:end[1]] for psd in x_test]

                kn, svm, dt = classifier(x_train_crop, y_train, x_test_crop, y_test)
                all_kn.append(kn)
                all_svm.append(svm)
                all_dt.append(dt)
                intervals.append(interval[0])
                intervals.append(interval[1])
            print(f'Выборка:{j},Интервал:{i}')
        save_in_1K(path + f'\\{j}\\acc20.txt', all_kn, all_svm, all_dt, intervals, mode=20)


""" ___________________________________________
                    Сценарий 3
    ___________________________________________
"""

"""
 Вычисление косинусного преобразования, μ – среднего по выборке и σ - среднеквадратического отклонения.
    Вход:
        img_nogrey - изображение
        is_zigzag - распологать ли элементы в зиг-заг
    Выход: 
        averages_beta - Вектор средних значений бетта коэффициентов 
"""


def cosinus_trans(img_nogrey, block_size=8):
    img = imread(img_nogrey)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_pic = np.dot(img[..., :3], rgb_weights)
    # img_grey = color.rgb2gray(img)  # Изображение в оттенках серого
    img_grey = grayscale_pic

    w, h = img_grey.shape
    count = int(w / block_size)
    blocks = []
    averages = [[] for i in range(block_size * block_size)]

    for i in range(count):
        for j in range(count):
            block = img_grey[i * block_size:(i * block_size + block_size), j * block_size:(j * block_size + block_size)]
            blocks.append(block)

            # Косинусное преобразование
            block_dct = dct(dct(block).T)

            block_dct = zigzag(block_dct)
            for k in range(len(block_dct)):
                averages[k].append(block_dct[k])

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


def cosinus_trans_show(img_nogrey):
    img = imread(img_nogrey)
    rgb_weights = [0.2989, 0.5870, 0.1140]
    grayscale_pic = np.dot(img[..., :3], rgb_weights)
    img_grey = color.rgb2gray(img)  # Изображение в оттенках серого
    img_grey = grayscale_pic

    w, h = img_grey.shape
    f = 8
    count = int(w / f)
    blocks = []
    blocks_dct = []
    averages = [[] for i in range(f * f)]

    for i in range(count):
        for j in range(count):
            block = img_grey[i * f:(i * f + f), j * f:(j * f + f)]
            blocks.append(block)

            # Косинусное преобразование
            block_dct = dct(dct(block).T)

            # модуль от косинусного перобразования надо ли ?
            # block_dct = np.abs(block_dct)
            blocks_dct.append(block_dct)

            block_dct = zigzag(block_dct)
            for k in range(len(block_dct)):
                averages[k].append(block_dct[k])

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

    dct2 = np.log(1 + np.abs(dct(dct(img_grey).T)))
    # Простотранство для отображения
    fig = plt.figure(figsize=(15, 5))

    fig.add_subplot(2, 3, 1)
    plt.title("Изображение до обработки", fontsize=12)
    imshow(img)

    # В оттенках серого. Значения от 0 до 1
    fig.add_subplot(2, 3, 2)
    plt.title("Изображение в отенках серого", fontsize=12)
    imshow(img_grey, cmap='gray')

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
 В каждой папке true и false, перебирает все датасеты и составляет массивы путей до изображений под каждый датасет 
 (сколько папок в path_... столько и датасетов) 
 Составляет массивы путей до настоящих и поддельных изображений в указанных папках.
    Вход:
        path_true и path_false - пути до папок с настоящими и поддельными изображениями
    Выход: 
        true_datasets - массивов путей до настоящих 
        false_datasets - массивов путей до поддельных изображений 
"""


def get_datasets_paths(path_true, path_false):
    true_datasets = [[os.path.join(path_true, dirpath)] for dirpath in os.listdir(path_true)]
    for j in true_datasets:
        true_items = []
        for dirpath, dirnames, filenames in os.walk(j[0]):
            if not (len(filenames) == 0):
                for i in filenames:
                    true_items.append(dirpath + '\\' + i)
        j.append(true_items)

    false_datasets = [[os.path.join(path_false, dirpath)] for dirpath in os.listdir(path_false)]
    for j in false_datasets:
        false_items = []
        for dirpath, dirnames, filenames in os.walk(j[0]):
            if not (len(filenames) == 0):
                for i in filenames:
                    false_items.append(dirpath + '\\' + i)
        j.append(false_items)

    return true_datasets, false_datasets


""" 
 Сохраняет по указанному пути массив бетта с путем к изображению  
    Вход: 
        path - путь до файла сохранения, 
        name - название сохраняемого, 
        averages_beta - массив бета
    Выход: нет
"""


def dct_save(path, name, averages_beta, file_name='dct.txt'):
    # Сохраняем 1 массив созданый по 1 изображению
    f = open(path + f'\\{file_name}', 'a')
    line = ''
    line += str(name) + '\t' + str([i for i in averages_beta]) + '\n'
    f.write(line)
    f.close()


"""
 Считывает из указанного пути к файлу массив бетта и пути до изображений  
    Вход: path - путь до файла чтения
    Выход: averages - массив матриц beta
"""


def dct_read(path, file_name='dct.txt'):
    file_path = path + f'\\{file_name}'
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


def beta_matrix_of_images(datasets, istrue, path, file_name, block_size):
    which = '\\false\\'
    if istrue:
        which = '\\true\\'

    beta_matrix = []
    matrices_images = []
    for dataset in datasets:
        averages = [[] for i in range(block_size*block_size)]

        # Путь до папки датасета
        path_folder = path + which + os.path.basename(os.path.normpath(dataset[0]))
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        bookmark = len(dct_read(path_folder, file_name=file_name))
        if len(dataset[1]) > bookmark:
            count = bookmark
            dataset_c = dataset[1][bookmark:]
            for i in dataset_c:

                averages_beta = cosinus_trans(i, block_size=block_size)
                dct_save(path_folder, i, averages_beta, file_name=file_name)
                for j in range(len(averages_beta)):
                    averages[j].append(averages_beta[j])
                print(count, '/', len(dataset[1]), block_size, dataset[0])
                count += 1

        averages = dct_read(path_folder, file_name=file_name)
        matrices_images.append(averages)
        transpose_averages = [*zip(*averages)]
        beta = [np.mean(j) for j in transpose_averages]
        beta_matrix.append(beta)

    return beta_matrix, matrices_images


# Чтение файлов dct.txt
def read_beta_matrix_of_images(istrue, path, file_name):
    which = '\\false\\'
    if istrue:
        which = '\\true\\'
    beta_matrix = []
    matrices_images = []
    datasets = [path + which + dirpath for dirpath in os.listdir(path + which)]
    for path_folder in datasets:
        # Путь до папки датасета
        averages = dct_read(path_folder, file_name=file_name)
        matrices_images.append(averages)
        # средняя матрица для всего датасета
        transpose_averages = [*zip(*averages)]
        beta = [np.mean(j) for j in transpose_averages]
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


def show_beta_statistic(beta_true, beta_false, true, false):
    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')

    for i in range(len(true)):
        x = beta_true[i][1:]
        y = [j for j in range(1, len(beta_true[i]))]
        ax.plot(y, x, label=os.path.basename(os.path.normpath(true[i])) + ' (Настоящие)')

    for i in range(len(false)):
        x = beta_false[i][1:]
        y = [j for j in range(1, len(beta_false[i]))]
        ax.plot(y, x, label=os.path.basename(os.path.normpath(false[i])) + ' (Сгенерированые)')

    ax.set_xlabel('Номер β коэффициента')
    ax.set_ylabel('Значение β коэффициента')
    ax.set_title("График значений β для каждого задействованного набора данных")
    ax.legend()
    show()


"""
 Вычисление значения хи квадрат () для двух масиво бетта из 2х датасетов
    Вход: 
        dataset1 - масиво бетта коэффициентов 1, 
        dataset2 - масиво бетта коэффициентов 2.
    Выход: с - значение хи квадрат (расстояние)
"""


def x_squer(dataset1, dataset2):
    c = [0 for i in range(len(dataset1[0]))]
    for i in range(len(c)):
        for j in range(len(dataset1)):
            c[i] += ((dataset1[j][i] - dataset2[j][i])**2)/dataset2[j][i]
    return c


"""
 Получает из массива максимальное значение, удаляет и ищет снова, повторяет count раз
    Вход: 
        c - массив, 
        count - сколько требуется наибольших значений.
    Выход: a - вектор count наибольших значений в массиве в порядке убывания.
"""


def multi_argmax(c, count):
    arr = c
    reduced_arr = []
    for i in range(count):
        arg = np.argmax(arr)
        maximum = round(np.max(arr))
        reduced_arr.append((arg, maximum))
        arr[arg] = 0
    return reduced_arr


"""
 Вычисляет массив номеров бетта коэффициентов, по которым можно классифицировать изображения 
    Вход: 
       path_true - путь до папки с датасетами настоящих изображений, 
       path_false - путь до папки с датасетами подельных изображений, 
       path - путь до папки, 
       size_of_dataset - количество изображений в датасете, которое используется.
    Выход: отсутствует
"""


def data_to_frequencies(path_true, path_false, path, size_of_dataset):
    if os.path.exists(path_true) and [dirpath for dirpath in os.listdir(path_true)]:
        # 1. получаем массив путей до файлов картинок.
        true, false = get_datasets_paths(path_true, path_false)

        # Приводим датасеты к одному размеру
        for i in range(len(true)):
            true[i][1] = true[i][1][:size_of_dataset]

        for i in range(len(false)):
            false[i][1] = false[i][1][:size_of_dataset]

        # 2. Высчитываем и сохраняем матрицу(64) для каждого изображения в каждом датасете
        beta_true, matrices_true = beta_matrix_of_images(true, True, path, block_size=8, file_name=f'dct{8}.txt')
        beta_false, matrices_false = beta_matrix_of_images(false, False, path, block_size=8, file_name=f'dct{8}.txt')


def data_to_frequencies1(path_true, path_false, path, size_of_dataset):
    sizes = [8 * (2 ** i) for i in range(1, 5)]

    for block_size in sizes:
        if os.path.exists(path_true) and [dirpath for dirpath in os.listdir(path_true)]:
            # 1. получаем массив путей до файлов картинок.
            true, false = get_datasets_paths(path_true, path_false)

            # Приводим датасеты к одному размеру
            for i in range(len(true)):
                true[i][1] = true[i][1][:size_of_dataset]

            for i in range(len(false)):
                false[i][1] = false[i][1][:size_of_dataset]

            # 2. Высчитываем и сохраняем матрицу(64) для каждого изображения в каждом датасете
            beta_true, matrices_true = beta_matrix_of_images(true, True, path, block_size=block_size,
                                                             file_name=f'dct{block_size}.txt')
            beta_false, matrices_false = beta_matrix_of_images(false, False, path, block_size=block_size,
                                                               file_name=f'dct{block_size}.txt')
        else:
            beta_true, matrices_true = read_beta_matrix_of_images(True, path, file_name=f'dct{block_size}.txt')
            beta_false, matrices_false = read_beta_matrix_of_images(False, path, file_name=f'dct{block_size}.txt')


""" ___________________________________________
                    Сценарий 4
    ___________________________________________
"""


# Оставляет только те бэтта коэффициенты, номера которых указаны в numbers_of_beta
#   Вход:
#       matrices - ,
#       numbers_of_beta - .
#   Выход:
#       matrices_tmp - массив аналогичный по структуре matrices, но в матрицах оставлены только те коэффициенты,
#       номера которых были указаны в numbers_of_beta
def beta_matrix_reduction(matrices, numbers_of_beta):
    matrices_tmp = []
    for dataset in matrices:
        tmp = []
        for j in numbers_of_beta:
            tmp.append(dataset[int(j)])
        matrices_tmp.append(tmp)
    return matrices_tmp


# Классифицирует массивы признаков по выборкам train и test
#   Вход:
#       x_train - массив признаокв тренировочной выборки,
#       y_train - массив классов тренировочной выборки,
#       x_test - массив признаокв тестовой выборки,
#       y_test - массив классов тестовой выборки.
#   Выход:
#         accuracy_KN - точночность для к ближайших соседей,
#         accuracy_SVM - точночность для метода опорных векторов,
#         accuracy_DT - точночность для дерева решений.
def classifier_dct(x_train, y_train, x_test, y_test):

    # Классификация ближайших соседей
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)

    predicts_kn = neigh.predict(x_test)
    accuracy_kn = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_kn[i]:
            accuracy_kn += 1
    # cr = classification_report(y_test, predicts_kn)
    # print(cr)

    # Классификация опорных векторов С радиальной базисной функции
    clf = SVC(kernel='rbf', gamma='auto')
    clf.fit(x_train, y_train)

    predicts_svm = clf.predict(x_test)
    accuracy_svm = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_svm[i]:
            accuracy_svm += 1
    # cr = classification_report(y_test, predicts_svm)
    # print(cr)

    # Классификатор дерева решений
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)

    predicts_dt = clf.predict(x_test)
    accuracy_dt = 0
    for i in range(len(y_test)):
        if y_test[i] == predicts_dt[i]:
            accuracy_dt += 1
    # cr = classification_report(y_test, predicts_dt)
    # print(cr)

    return accuracy_kn, accuracy_svm, accuracy_dt


"""
 Вычисляет массив номеров бетта коэффициентов, по которым можно классифицировать изображения 
    Вход: 
       path_true - путь до папки с датасетами настоящих изображений, 
       path_false - путь до папки с датасетами подельных изображений, 
       path - путь до папки, 
       size_of_dataset - колличество изображений в датасете, которое используется.
    Выход: отсутствует
"""


def classification_dct(path, number_of_samples, size_of_sample, p, file_read='dct.txt', file_special='special_beta.txt'):
    # 1. Считываем из файлов матрицу(64) для каждого изображения в каждом датасете
    beta_true, matrices_true = read_beta_matrix_of_images(True, path, file_name=file_read)
    beta_false, matrices_false = read_beta_matrix_of_images(False, path, file_name=file_read)

    # 2. Создание выборок для классификации в колличестве number_of_samples
    samples = [[] for i in range(number_of_samples)]
    for j in range(number_of_samples):
        print(j)
        true_part_sample = []
        number_of_true_false = int(size_of_sample / 2)  # Колличество настоящих и поддельных в 1ой выборке
        a = number_of_true_false / len(matrices_true)  # если число не целое
        int_a = int(a)
        if int_a % 2 == 0:
            b = int((a - int_a) * len(matrices_true)) + 1
        else:
            b = 0
        for i in matrices_true:
            if b - 1 < 0:
                true_part_sample += i[int_a * j: int_a * j + int_a]
            else:
                true_part_sample += i[(int_a + 1) * j:(int_a + 1) * j + (int_a + 1)]
                b -= 1

        false_part_sample = []
        number_of_true_false = int(size_of_sample / 2)  # Колличество настоящих и поддельных в 1ой выборке
        a = number_of_true_false / len(matrices_false)  # если число не целое
        int_a = int(a)
        if int_a % 2 == 0:
            b = int((a - int_a) * len(matrices_false)) + 1
        else:
            b = 0
        for i in matrices_false:
            if b - 1 < 0:
                false_part_sample += i[int_a * j: int_a * j + int_a]
            else:
                false_part_sample += i[(int_a + 1) * j:(int_a + 1) * j + (int_a + 1)]
                b -= 1

        x = true_part_sample + false_part_sample
        y = [1 for i in range(number_of_true_false)] + [0 for i in range(number_of_true_false)]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p, random_state=42)
        samples[j].append(x_train)
        samples[j].append(y_train)
        samples[j].append(x_test)
        samples[j].append(y_test)

    # 3. Сокращаем матрицу бетта значений, выбирая только те, которые есть в numbers_of_beta
    special_betas = dct_read(path, file_name=file_special)

    # 4. Классификация и расчёт точности для от 1 до 10 бетта коэффициентов
    all_kn = []
    all_svm = []
    all_dt = []
    for special_beta in special_betas:
        print(special_betas.index(special_beta) + 1)
        samples_tmp = copy.deepcopy(samples)
        kn1 = []
        svm1 = []
        dt1 = []
        for i in samples_tmp:
            a = beta_matrix_reduction(i[0], special_beta)
            b = beta_matrix_reduction(i[2], special_beta)

            kn, svm, dt = classifier_dct(a, i[1], b, i[3])
            kn1.append(kn)
            svm1.append(svm)
            dt1.append(dt)
            print(kn, svm, dt)
        all_kn.append(kn1)
        all_svm.append(svm1)
        all_dt.append(dt1)

    # 5. Отрисовка граффиков точности классификации для случаев 1-10 бета
    one_procent = size_of_sample*(1-p) / 100

    for i in range(len(special_betas)):
        show_acc(number_of_samples,
                 [j / one_procent for j in all_kn[i]],
                 [j / one_procent for j in all_svm[i]],
                 [j / one_procent for j in all_dt[i]],
                 title=f' при {str(i + 1)} β коэфициентах')


def check_for_comparison_dct(path, number_of_samples, p, count_of_features):
    # 1. Считываем из файлов матрицу(64) для каждого изображения в каждом датасете
    beta_true, matrices_true = read_beta_matrix_of_images(True, path)
    beta_false, matrices_false = read_beta_matrix_of_images(False, path)

    # 2. Сокращаем матрицу бета значений, выбирая только те, которые есть в numbers_of_beta
    numbers_of_beta = [35, 21, 20, 10, 9]
    matrices_true = beta_matrix_reduction(matrices_true, numbers_of_beta)
    matrices_false = beta_matrix_reduction(matrices_false, numbers_of_beta)

    # 3. Создание выборок для классификации в колличестве number_of_samples

    x = []
    y = []
    for i in matrices_true:
        x += i
        y += [1 for j in range(len(i))]
    for i in matrices_false:
        x += i
        y += [0 for j in range(len(i))]

    KN = []
    SVM = []
    DT = []

    for i in range(number_of_samples):
        sample = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=p, random_state=42 + i)
        sample.append(x_train)
        sample.append(y_train)
        sample.append(x_test)
        sample.append(y_test)

        # 4. Классификация и расчёт точности для от 1 до 5 бетта коэффициентов
        number_of_beta = 5
        KN1 = []
        SVM1 = []
        DT1 = []
        for count in range(number_of_beta):
            print(count+1)
            sample_tmp = copy.deepcopy(sample)

            for j in sample_tmp[0]:
                for k in range(number_of_beta - 1, count, -1):
                    j.pop(k)
            for j in sample_tmp[2]:
                for k in range(number_of_beta - 1, count, -1):
                    j.pop(k)
            kn, svm, dt = classifier_dct(sample_tmp[0], sample_tmp[1], sample_tmp[2], sample_tmp[3])
            KN1.append(kn)
            SVM1.append(svm)
            DT1.append(dt)
            print(kn, svm, dt)

        print(f'№{i+1}')
        KN.append(KN1)
        SVM.append(SVM1)
        DT.append(DT1)

    # 5. Отрисовка граффиков точности классификации для случаев 1-5 бетта
    one_procent = len(y_test) / 100

    KN = list(map(list, zip(*KN)))
    SVM = list(map(list, zip(*SVM)))
    DT = list(map(list, zip(*DT)))

    for i in range(number_of_beta):
        show_acc(number_of_samples, [i / one_procent for i in KN[i]], [i / one_procent for i in SVM[i]],
                 [i / one_procent for i in DT[i]], title=f' при {str(i + 1)} β коэфициентах')
