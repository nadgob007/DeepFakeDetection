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

    g = 0   # Отображение данных классификации для 10 признаков
    if g == 1:
        for i in range(40):
            kn_all, svm_all, dt_all = read_acc(path + f'\\split\\{i}\\acc.txt')
            show_acc(len(kn_all), kn_all, svm_all, dt_all, title=f'для 10 признаков изображение №{i}')

    e = 0  # Перещёт классификаторами для интервала в 20 признаков из участков по 10 из разных частей вектора признаков.
    if e == 1:
        classification20(path + '\\split', initial_params['number_of_samples'])

    f = 1  # Отображение тепловой карты
    if f == 1:
        all_kn, all_svm, all_dt, intervals = read_acc20(path + '\\split', initial_params['number_of_samples'])
        show_temp(all_kn, all_svm, all_dt, intervals, initial_params['number_of_samples'])


"""
     сценарий для обработки 1K изображений
"""


def scenario2(initial_params):
    path = initial_params['path']

    def read_saveold(path):
        file_path = path
        if not os.path.exists(file_path):
            f = open(file_path, 'r')
            f.close()

        f = open(file_path, 'r')
        line = '.'
        psd = []
        names = []

        line = f.readline()
        line = '1'
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
        return names, psd

    def sort_l(arr):
        first = True
        sort = True
        while first or sort:
            sort = True
            for i in arr[0]:
                if arr[0].index(i) == len(arr[0]) - 1:
                    break
                if i > arr[0][arr[0].index(i)+1]:
                    a, b = arr[0].index(i), arr[0].index(i)+1
                    arr[0][b], arr[0][a] = arr[0][a], arr[0][b]
                    arr[1][b], arr[1][a] = arr[1][a], arr[1][b]

                    sort = False
            if sort and first:
                first = False
                sort = False
            print(min(arr[0]), max(arr[0]))
            print(arr[0][0], arr[0][-1])
        return arr

    fake100 = [[], []]    # 100KFake_10K
    thispdne = [[], []]   # thispersondoesntexists_10K
    celeb = [[], []]      # celebA-HQ_10K
    flicker = [[], []]    # Flickr-Faces-HQ_10K

    for i in range(40):
        curent = path + '\\split' + f'\\{str(i)}'
        names_a, a = read_saveold(curent + '\\train_psd.txt')
        names_b, b = read_saveold(curent + '\\test_psd.txt')

        for j in names_a:
            data = names_a.index(j)
            pa, number = os.path.split(j)
            d, dataset = os.path.split(pa)
            if dataset == '100KFake_10K':
                fake100[0].append(int(number.split('.jpg')[0]))
                fake100[1].append(a[data])
            elif dataset == 'thispersondoesntexists_10K':
                thispdne[0].append(int(number.split('.jpg')[0]))
                thispdne[1].append(a[data])
            elif dataset == 'celebA-HQ_10K':
                celeb[0].append(int(number.split('.jpg')[0]))
                celeb[1].append(a[data])
            else:
                flicker[0].append(int(number.split('.jpg')[0]))
                flicker[1].append(a[data])

        for j in names_b:
            data = names_b.index(j)
            pa, number = os.path.split(j)
            d, dataset = os.path.split(pa)
            if dataset == '100KFake_10K':
                fake100[0].append(int(number.split('.jpg')[0]))
                fake100[1].append(b[data])
            elif dataset == 'thispersondoesntexists_10K':
                thispdne[0].append(int(number.split('.jpg')[0]))
                thispdne[1].append(b[data])
            elif dataset == 'celebA-HQ_10K':
                celeb[0].append(int(number.split('.jpg')[0]))
                celeb[1].append(b[data])
            else:
                flicker[0].append(int(number.split('.jpg')[0]))
                flicker[1].append(b[data])
        print(i)

    fake100 = sort_l(fake100)
    print(1)
    thispdne = sort_l(thispdne)
    print(3)
    flicker = sort_l(flicker)

    for i in range(len(fake100[0])):
        psd_save(path + '\\datasets_psd\\false\\100KFake_10K',
                 path + f'\\false\\100KFake_10K\\{fake100[0][i]}.jpg',
                 fake100[1][i])

    for i in range(len(thispdne[0])):
        psd_save(path + '\\datasets_psd\\false\\thispersondoesntexists_10K',
                 path + f'\\false\\thispersondoesntexists_10K\\{thispdne[0][i]}.jpg',
                 thispdne[1][i])

    for i in range(len(flicker[0])):
        psd_save(path + '\\datasets_psd\\true\\Flickr-Faces-HQ_10K',
                 path + f'\\false\\Flickr-Faces-HQ_10K\\{flicker[0][i]}.jpg',
                 flicker[1][i])


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
