
# *** МОДУЛЬ ПРОЕКТА ***

#######################################################################################
#######################################################################################

# РАЗДЕЛЫ МОДУЛЯ (в скобках номер строки для быстрого поиска):

#   (41)    1. Импорт необходимых бибилиотек 
#   (67)    2. Функциии для первого этапа (подготовка данных для кластеризации):
#   (69)          get_region_names_standart 
#  (109)          data_preparation_step_1 
#  (157)          get_trend 
#  (183)          data_preparation_step_2 
#  (279)          data_preparation_step_3 
#  (349)          feature_histogram 
#  (389)          data_features_rename 
#  (415)          output_problem_regions 
#  (432)          index_into_100k_population 
#  (465)    3. Функции для второго этапа (FE,EDA,FS):
#  (467)          normal_test
#  (501)          normal_test_df
#  (536)          corr_visual
#  (569)          target_corr_visual
#  (606)          target_matthews_corrcoef
#  (639)          get_redundant_pairs
#  (650)          get_top_abs_correlations
#  (669)          target_xi2_test
#  (689)    4. Функции для третьего этапа (кластеризация):
#  (691)          get_silhouette_score
#  (729)          get_calinski_harabasz_score
#  (776)          get_davies_bouldin_score
#  (813)          clustering_proba
#  (974)          feature_cluster_describe
# (1038)          plot_cluster_profile
# (1079)    5. Словарь стандартных названий регионов

#######################################################################################
#######################################################################################

# 1. Импорт необходимых библиотек

import pandas as pd
import numpy as np
import regex as re
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import spearmanr
from sklearn.feature_selection import chi2
from sklearn.metrics import matthews_corrcoef
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

random_state = 42
#######################################################################################
#######################################################################################

# 2. Функциии для первого этапа (сбор, обработка, промежуточный анализ и подготовка данных для проведения кластеризации.)

def get_region_names_standart(region, replacement):
    ''''
    Функция принимает на вход название региона из датасета и замену выражений в скобках,
    возвращает стандартизированное название региона. Стандартизированные названия извлекаются
    из словаря regions_names_standart, который содержится в модуле project_module.py
    и является немного модифицированнм словарем взятым отсюда:
        https://github.com/DKudryavtsev/RussiaRegions/tree/main
    Аргументы функции:
        region: названием региона из датасета
        replacement: замена слов в скобках в названии региона(в виде строки)
    Функция возвращает:
        стандартизированное название региона в зависимости от условий
    '''
    
    # очищаем аргумент функции от скобок
    region = re.sub('[\(].*?[\)]', replacement, region).lower()

    # если аргумент содержит название Архангельской или Тюменской областей
    if region.find('арханг') != -1 and region.find('без') != -1:
        return 'Архангельская область без Ненецкого АО'
    elif region.find('арханг') != -1:
        return 'Архангельская область'
    elif region.find('тюмен') != -1 and region.find('без') != -1:
        return 'Тюменская область без округов'
    elif region.find('тюмен') != -1:
        return 'Тюменская область'

    # если аргумент НЕ содержит название Архангельской или Тюменской областей
    else: 
        for key, value in regions_names_standart.items():
            # если стандартизированное название ПРИСУТСТВУЕТ в словаре
            if re.search(key, region):
                return value
        # если стандартизированное название ОТСУТСТВУЕТ в словаре 
        # (например, названия округов или различные словосочетания),
        # то возращаем NaN
        return np.nan

#######################################################################################

def data_preparation_step_1(df, region, replacement):
    ''''
    Функция принимает на вход исходный датасет, признак, содержащий названия регионов, замену выражения
    в скобках в названии региона (для функции get_region_names_standart) и 
    производит очистку данных, изменяет названия регионов на стандартизированные (используется функция 
    get_region_names_standart), выводит информацию об обновленном датасете и возращает
    обновленный датасет.
    Аргументы функции:
        df: исходный датасет
        region: названием региона из датасета
        replacement: замена слов в скобках в названии региона (в виде строки)
    Функция возвращает:
        df: обновленный датасет
    '''

    # сохраняем название таблицы
    df_name = df.name
    # заменяем нечисловые значения в данных на NaN, за исключением признака названий регионов
    df.loc[:,df.columns[1:]] = df.loc[:,df.columns[1:]].replace(regex=[r'\D+'], value=np.nan)

    # переводим данные в тип float кроме названий регионов
    df = pd.concat([df.iloc[:,0], df.iloc[:,1:].astype(float)], axis=1)

    # заменяем названия регионов в датасете на стандартизированные названия либо NaN
    df.loc[:,region] = df.loc[:,region].apply(lambda x: get_region_names_standart(x, replacement))
    # убираем строки с пропусками в признаке названия региона
    df = df.dropna(subset=['region']).reset_index().drop('index', axis=1)
    # заново присваиваем таблице ее название
    df.name = df_name
    
    # выводим информацию об обновленных данных
    print('*** Данные после первичной обработки ***')
    print(df.name)
    display(df.head())
    print(f'Размерность: {df.shape}')
    print(f'Количество строк, содержащих пропуски {df[df.isna().any(axis=1)].shape[0]}')
    print(f'Число найденных дубликатов: {df[df.duplicated()].shape[0]}\n')
    # выводим строки, содержащие пропуски в данных
    if df.isnull().any().any():
        print('Строки содержащие пропуски:')
        display(df[df.isna().any(axis=1)])
    # визуально разделяем выведенную информацию
    print('-'*100)
    
    return df

#######################################################################################

def get_trend(row):
    ''''
    Функция принимает на вход строку датасета с числовыми данными за представленный период 
    и возвращает тренд в виде числа. Для расчета тренда используется скользящее среднее.
    Аргументы функции:
        row: строка датасета с числовыми данными
    Функция возвращает:
        -1: если тренд на понижение
         0: если тренд без направления 
         1: если тренд на повышение
    '''
    
    # убираем NaN из представленных данных, если они присутствуют
    row_dropna = row[1:].dropna()

    # если количество значений больше 1, рассчитываем тренд
    if len(row_dropna) > 1:
        # расчитываем тренд, используя скользящую среднюю
        trend  = row_dropna.rolling(len(row_dropna)-1).mean().dropna().reset_index(drop=True)
        return -1 if trend[0] > trend[1] else 1 if trend[0] < trend[1] else 0
    # если количество значений 0 или 1, то тогда возвращаем 0
    else:
        return 0
    
#######################################################################################

def data_preparation_step_2(df, bad_trend, bad_per_of_rf, threshold_100):
    ''''
    Функция принимает на вход датасет, значение плохого тренда, плохой уровень показателя 
    по сравнению с уровнем по РФ в целом, и расчитывает тренд числовых данных
    за представленный период, добавляя соответствующий признак trend в таблицу
    (используется функция get_trend), очищает таблицу от числовых данных за 
    представленный период (за исключением последнего года), рассчитывает соотношение
    числовых данных за последний представленный год региона к данным по РФ, 
    создавая новый признак per_of_rf, выводит информацию по обновленному датасету
    и возвращает обновленный датасет.
    Аргументы функции:
        df: исходный датасет
        bad_ternd: указываем плохой тренд в виде числа
            -1: тренд на снижение
             0: тренд без направления
             1: тренд на повышение
        bad_per_of_rf: указываем плохой уровень показателя по сравнению с уровнем по РФ в целом
            'less': ниже уровня по РФ
            'more': выше уровня по РФ
        threshold_100: искать ли регионы с плохими показателями по сравнению с уровнем по РФ в целом
            True: искать
            False: не искать
    Функция возвращает:
        df: обновленный датасет
    '''

    # сохраняем название таблицы
    df_name = df.name

    # рассчитываем тренд в представленных числовых данных
    df_copy = copy.deepcopy(df)
    df_copy.loc[:,'trend'] = df_copy.loc[:,df.columns[1:]].apply(lambda row: get_trend(row), axis=1)
    df = df_copy
    # !Примечание: сложности с копированием датасета необходимы для предотвращения вывода предупреждений pandas

    # убираем данные за представленный период, за исключением последнего года
    columns_new = df.columns.to_list()[-2:]
    columns_new.insert(0,'region')
    df = df[columns_new]
    
    # рассчитываем соотношение числовых данных за последний представленный год региона к данным по РФ
    value_rf = df.iloc[0,1] # данные по РФ
    df_copy = copy.deepcopy(df)
    df_copy.loc[:, 'per_of_rf'] = df_copy.loc[:, df_copy.columns[1]].apply(lambda x: round(x/value_rf*100,2))
    df = df_copy
    # !Примечание: сложности с копированием датасета необходимы для предотвращения вывода предупреждений pandas

    # заново присваиваем таблице ее название
    df.name = df_name

    # выводим информацию об обновленных данных
    print('*** Данные после второго этапа обработки ***')
    print(df.name)
    display(df.head())
    print(f'Размерность: {df.shape}')
    print(f'Количество строк, содержащих пропуски {df[df.isna().any(axis=1)].shape[0]}')
    print(f'Число найденных дубликатов: {df[df.duplicated()].shape[0]}\n')
    # выводим строки, содержащие пропуски в данных
    if df.isnull().any().any():
        print('Строки содержащие пропуски:')
        display(df[df.isna().any(axis=1)])

    # выводим данные с ПЛОХИМ трендом
    print('* Данные с ПЛОХИМ трендом *')
    display(df[df['trend'] == bad_trend])
    
    # выводим кол-во регионов с ПЛОХИМИ показателями относительно уровня РФ
    if threshold_100 is True:
        # если плохой уровень - это уровень < 100%
        if bad_per_of_rf == 'less':    
            value_per_of_rf = df[df['per_of_rf'] < 100].shape[0]
            print(f'* Количество регионов с ПЛОХИМИ показателями относительно уровня РФ: {value_per_of_rf} *')
        # если плохой уровень - это уровень > 100%
        elif bad_per_of_rf == 'more':
            value_per_of_rf = df[df['per_of_rf'] > 100].shape[0]
            print(f'* Количество регионов с ПЛОХИМИ показателями относительно уровня РФ: {value_per_of_rf} *')

        # выводим данные с ПЛОХИМ трендом и ПЛОХИМИ показателями относительно уровня РФ
        print('* Данные с ПЛОХИМ трендом и ПЛОХИМИ показателями относительно уровня РФ:')
        mask_1 = df['trend'] == bad_trend
        # если плохой уровень - это уровень < 100%
        if bad_per_of_rf == 'less':
            mask_2 = df['per_of_rf'] < 100
        # если плохой уровень - это уровень > 100%
        if bad_per_of_rf == 'more':
            mask_2 = df['per_of_rf'] > 100
        display(df[mask_1 & mask_2])
        print(f'Всего таких регионов: {df[mask_1 & mask_2].shape[0]} *')

    # визуально разделяем выведенную информацию
    print('-'*100)

    return df

#######################################################################################

def data_preparation_step_3(df, bad_trend, bad_per_of_rf, and_threshold, add_threshold):
    ''''
    Функция принимает на вход датасет (df), направление плохого тренда (bad_trend), какой показатель по сравнению с уровнем РФ
    будет плохим (выше или ниже)(bad_per_of_rf), порог предыдущего показателя для дополнительного критерия
    плохого региона (and_threshold) и дополнительный порог значения показателя как еще один критерий плохого региона add_threshold. 
    На основании полученных аргументов
    функция создает признак неблагополучного региона (bad_region = 1), выводит 'плохие' регионы и возвращает
    обновленный датасет.
    Аргументы функции:
        df: исходный датасет
        bad_trend: направление плохого тренда представленных показателей
            -1: отрицательный тренд
             0: тренд без направления
             1: положительный тренд
        bad_per_of_rf: плохой уровень показателей региона по сравению с РФ
            'less': показатель региона ниже, чем по РФ (и это ПЛОХО)
            'more': показатель региона выше, чем по РФ (и это ПЛОХО)
        and_threshold: порог признака per_of_rf ниже/выше которого (зависит от предыдущего аргумента) регион
                также признается ПЛОХИМ (bad_region)
        add_threshold: искать ли еще ПЛОХИЕ регионы с плохими показателями по сравнению с уровнем по РФ в целом
            любое число: искать; введеное число - это уровень по которому будет доплолнительно выявляться плохие регионы
            False: не искать
    Функция возвращает:
        df: обновленный датасет
    '''

    # создаем признак bad_region и заполняем его нулями
    df['bad_region'] = 0
    
    # маска ПЛОХОГО тренда
    mask_trend = df['trend'] == bad_trend
    
    # если "чем ниже показатель, тем хуже"
    if bad_per_of_rf == 'less':
        # если ДОПОЛНИТЕЛЬНО помечаем ПЛОХИМИ регионы с показателем ниже уровня от РФ в целом, то создаем соответствующую маску
        if add_threshold is not False: mask_per_of_rf = df['per_of_rf'] < add_threshold
        # индексы ПЛОХИХ регионов - это пересечение индексов ПЛОХОГО тренда И показателя ниже указанного уровня and_threshold
        ixs_2 = df[mask_trend].index.intersection(df.loc[df['per_of_rf'] < and_threshold].index)
    # а если "чем выше показатель, тем хуже"
    elif bad_per_of_rf == 'more':
        # если ДОПОЛНИТЕЛЬНО помечаем ПЛОХИМИ регионы с показателем выше уровня РФ в целом, то создаем соответствующую маску
        if add_threshold is not False: mask_per_of_rf = df['per_of_rf'] > add_threshold
        # индексы ПЛОХИХ регионов - это пересечение индексов ПЛОХОГО тренда и показателя выше указанного уровня and_threshold
        ixs_2 = df[mask_trend].index.intersection(df.loc[df['per_of_rf'] > and_threshold].index)
    
    # если ДОПОЛНИТЕЛЬНО помечаем ПЛОХИМИ регионы с показателем ниже/выше уровня РФ в целом (см. подробно предыдущее условие)
    if add_threshold is not False:
        # индексы ПЛОХОГО региона - это маска составленная в предыдущем условии
        ixs_1 = df[mask_per_of_rf].index
    # а если дополнительно НЕ помечаем ПЛОХИМИ регионы с показателем ниже/выше уровня РФ в целом
    elif add_threshold is False:
        # тогда учитывваем только индексы по предыдущему условию (перечечение индексов плохого тренда и показателя выше/нижу уровня and_threshold)
        ixs_1 = ixs_2

    # помечаем плохие регионы, используя полученные индексы по предыдущим условиям
    df.loc[ixs_1, 'bad_region'] = 1
    df.loc[ixs_2, 'bad_region'] = 1

    # выводим плохие регионы
    print(df.name)
    print("* 'ПЛОХИЕ' регионы *")
    display(df[df['bad_region'] == 1])
    print(f"* Всего 'ПЛОХИХ' регионов: {df[df['bad_region'] == 1].shape[0]} *")
    # визуально разделяем выведенную информацию
    print('-'*100)

    return df

#######################################################################################

def feature_histogram(df, feature, bins=88, min_x='no limit', max_x='no limit'):
    ''''
    Функция принимает на вход датасет и его признак и выводи гистограмму
    распределения признака.
    Аргументы функции:
        df: датасет
        feature: признак в виде строки
        bins: количество столбцов (баров) (88 по умолчанию)
        min_x: ограничиваем минимальное значение по оси x в виде числа (по умолчанию 'no limit', т.е. без ограничений)
        max_x: ограничиваем максимальное значение по оси x в виде числа (по умолчанию 'no limit', т.е. без ограничений)
    '''
    # выводим название таблицы
    print(df.name)
    
    # если без ограничений значений по оси x
    if min_x == 'no limit' or max_x == 'no limit':
        df[feature].plot(
        kind='hist',
        figsize=(10, 2),
        title='Распределение ' + feature,
        grid = True,
        color = 'blue',
        bins=bins
        );
    # если c ограничениями значений по оси x
    else:
        fig = plt.figure()
        ax = plt.subplot(111)
        df[feature].plot(
        kind='hist',
        figsize=(10, 2),
        title='Распределение ' + feature,
        grid = True,
        color = 'blue',
        bins=bins
        )
        ax.set_xlim(min_x,max_x);

#######################################################################################

def data_features_rename(df, prefix):
    '''
    Функция принимает на вход таблицу и префикс, который
    добавляется к названиям признаков, начиная со второго.
    Нулевой признак (название региона) не меняется. Название
    первого признака (показатели за последний представленный год)
    меняется на префикс. Далее выводится обновленная таблица.
    Аргументы функции:
        df: исходный датасет
        prefix: префикс в виде строки
    '''
    # !!! переводим название колонок в строки (иначе может умирать kernel)
    df.columns = [str(x) for x in df.columns]

    # меняем название 1-го признака на префикс
    df.columns.values[1] = prefix
    # добавляем префикс к оставшимся признакам, начиная со 2-го
    df.columns.values[2:] = prefix + '_' + df.columns.values[2:]
    
    # выводим полученные данные
    print(df.name)
    print(f'Размерность данных: {df.shape}')
    display(df.head())

#######################################################################################

def output_problem_regions(df):
    '''
    Функция принимает на вход датафрейм с данными и выводит срез этих данных по проблемным регионам.
    Проблемные регионы - это Архангельская и Тюменская области с их округами и без.
    '''
    # маски по проблемным регионам
    mask_1 = (df['region'] == 'Архангельская область') | (df['region'] == 'Архангельская область без Ненецкого АО') | \
        (df['region'] == 'Ненецкий автономный округ')
    mask_2 = (df['region'] == 'Тюменская область') | (df['region'] == 'Тюменская область без округов') | \
        (df['region'] == 'Ханты-Мансийский автономный округ - Югра') | (df['region'] == 'Ямало-Ненецкий автономный округ')

    # вывод данных
    print(df.name)
    display(df[mask_1 | mask_2])

#######################################################################################

def index_into_100k_population(df, df_population, round, year_first, year_last):
    ''''
    Функция принимает на вход датафрейм с исходными данными выраженными в абсолютных числах, 
    датафрейм с численностью населения по регионам за период 1999-2022, 
    количество знаков после запятой для округления результатов, начальный и последний годы исходных данных.
    Аргументы функции:
        df: таблица с исходными данными в абсолютных числах
        df_population: таблица с численностью населения. 
            Примечание: учитываем период исходных данных и делаем соответствующий срез в данных по населению,
            например: df_population.iloc[:,1:3]
        round: количество цифр после запятой при округлении результата
        year_first: первый год периода исходных данных
        year_last: последний год исходных данных
    Функция возвращает:
        df: обновленный исходный датафрейм с данными переведенными на 100k населения
    
    '''
    # получаем матрицы с данными из исходных датафреймов
    df_values = df.iloc[:,1:].values
    df_population_values = df_population.values

    # расчитываем исходный абсолютный показатель на 100k населения с округлением, получив еще один массив
    result_values = np.round(df_values/df_population_values*100000, round)
    # переводим массив с результатами в датафрейм
    result_df = pd.DataFrame(data=result_values, columns=[i for i in range(year_first, year_last+1)])
    # cоединяем датафрейм с результатами с названиями регионов
    df = pd.concat([df.iloc[:,0], result_df], axis=1)

    return df

#######################################################################################
#######################################################################################

# 3. Функции для второго этапа (FE,EDA,FS):

def normal_test(data, alpha):
    '''
    Функция принимает на вход последовательность значений или объект Series (например, признак датафрейма)
    и уровень значимости (alpha), проводит тесты Шапиро и Д'Агостино на нормальность и 
    выводит результаты тестирования.
    Аргументы функции:
        data: объект Series/признак датафрейма
        alpha: уровень значимости
    '''

    # формулируем гипотезы
    H0 = 'Данные распределены нормально'
    Ha = 'Данные не распределены нормально (мы отвергаем H0)'

    # тест Шапиро-Уилка
    print('* тест Шапиро-Уилка *')
    stat, p = shapiro(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > alpha:
        print(H0)
    else:
        print(Ha)
		
    # тест Д'Агостино
    print("\n* тест Д'Агостино *")
    stat, p = normaltest(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print(H0)
    else:
        print(Ha)

#######################################################################################

def normal_test_df(df, alpha):
    '''
    Функция принимает на вход датафрейм (df) и уровень значимости (alpha) и проводит тесты
    Шапиро и Д'Агостино для распределений каждого признака. Функция выводит результаты
    тестирования в виде таблицы, где указаны те признаки, которые распределены нормально.
    Примечание:
        H0 = 'Данные распределены нормально'
        Ha = 'Данные не распределены нормально (мы отвергаем H0)'
    '''

    # списки с нормально распределенными признаками
    normal_features_shapiro = [] # тест Шапиро
    normal_features_normaltest = [] # тест Д'Агостино

    # проводим тесты для каждого признака
    for col in df.columns.to_list():
        # тест Шапиро-Уилка
        stat, p = shapiro(df[col])
        # если признак распределен нормально добавляем его в список
        if p > alpha: normal_features_shapiro.append(col)
        
        # тест Д'Агостино
        stat, p = normaltest(df[col])
        # если признак распределен нормально добавляем его в список
        if p > alpha: normal_features_normaltest.append(col)

    # сводим полученные результаты в таблицу и выводим ее
    print('Эти признаки распределены нормально:')
    result = pd.DataFrame(list(zip(normal_features_shapiro, normal_features_normaltest)),
               columns =['* тест Шапиро-Уилка *', '* тест ДАгостино *'])
    result.index += 1
    display(result)

#######################################################################################

def corr_visual(data, width, height, pearson=False, spearman=True, kendall=True):
    '''
    Функция принимает на вход датафрейм, размеры графика, методы корреляции для расчета и 
    выводит тепловую карту корреляции.
    Аргументы функции:
        data: датафрейм
        widh: ширина графика
        height: высота графика
        pearson: метод корреляции Пирсона(по умолчанию False)
        spearman: метод корреляции Спирмена (по умолчанию True)
        kedall: метод корреляции Кендалла (по умолчанию True)
    '''
    # задаем фигуру и ее размер 
    plt.figure(figsize=(width, height))
    
    # если выбрано True расчитываем и выводим корреляцию Спирмена
    if spearman is True:
        plt.subplot(3,1,1)
        sns.heatmap(data.corr(method='spearman'), annot=True, fmt='.2f', linewidth=.5, mask = np.triu(data.corr(method='spearman')))\
            .set(title = "* Корреляция Спирмена *");
    # если выбрано True расчитываем и выводим корреляцию Кендалла
    if kendall is True:
        plt.subplot(3,1,2)
        sns.heatmap(data.corr(method='kendall'), annot=True, fmt='.2f', linewidth=.5, mask = np.triu(data.corr(method='kendall')))\
            .set(title = "* Корреляция Кендалла *");
    # если выбрано True расчитываем и выводим корреляцию Пирсона
    if pearson is True:
        plt.subplot(3,1,3)
        sns.heatmap(data.corr(method='pearson'), annot=True, fmt='.2f', linewidth=.5, mask = np.triu(data.corr(method='pearson')))\
            .set(title = "* Корреляция Пирсона *");

#######################################################################################

def target_corr_visual(data, target, width, height, pearson=False, spearman=True, kendall=True, top_number=30):
    '''
    Функция принимает на вход датафрейм, целевой признак, размеры графика, методы корреляции для расчета,
    количество признаков для отображения на графике и 
    расчитывает корреляцию признаков датафрейма с целевым признаком, выводит результаты в виде графика, 
    сортируя признаки по убыванию значения модуля корреляции.
    Аргументы функции:
        data: датафрейм
        target: целевой признак
        widh: ширина графика
        height: высота графика
        pearson: метод корреляции Пирсона(по умолчанию False)
        spearman: метод корреляции Спирмена (по умолчанию True)
        kedall: метод корреляции Кендалла (по умолчанию True)
        top_number: количество признаков для вывода
    '''
    # задаем фигуру и ее размер
    plt.figure(figsize=(width, height))

    # если выбрано True расчитываем и выводим корреляцию Спирмена
    if spearman is True:
        plt.subplot(3,1,1)
        data.corr(method='spearman')[target].sort_values(key=abs)[-(top_number+1):].plot.barh().set_title(
            f'Correlation with * {target} * (Spearman)');
    # если выбрано True расчитываем и выводим корреляцию Кендалла
    if kendall is True:
        plt.subplot(3,1,2)
        data.corr(method='kendall')[target].sort_values(key=abs)[-(top_number+1):].plot.barh().set_title(
            f'Correlation with * {target} * (Kendall)');
    # если выбрано True расчитываем и выводим корреляцию Пирсона
    if pearson is True:
        plt.subplot(3,1,3)
        data.corr(method='pearson')[target].sort_values(key=abs)[-(top_number+1):].plot.barh().set_title(
            f'Correlation with * {target} * (Pearson)');

#######################################################################################

def target_matthews_corrcoef(data, binary_features, target, width, height, top_number, table=False):
    '''
    Функция принимает на вход датафрейм, бинарные признаки для сравнения, целевой признак, 
    размеры графика, количество признаков из топа по абсолютному значению корреляции для вывода,
    рассчитывает корреляцию Мэтьюса между внесенными признаками и таргетом и выводит результат в виде
    диаграммы и таблицы.
    Аргументы функции:
        data: датафрейм
        binary_features: перечень бинарных признаков в виде списка
        target: целевой признак
        widh: ширина графика
        height: высота графика
        top_number: сколько признаков из топа по абсолютным значениям корреляции выводить
        table: дополнительно вывести таблицу со значениями, если True (False по умолчанию)
    '''
    # список с результатами расчета корреляции
    result = []
    
    # рассчитываем корреляцию и вносим результат в список выше
    for feature in binary_features:
        result.append(matthews_corrcoef(data[feature], data[target]))

    # переводим результат в объект Series
    result = pd.Series(result, index=binary_features, name='corr')
    # сортируем результаты по абсолютному значению и задаем кол-во топ-результатов для вывода
    result = result.sort_values(key=abs)[-(top_number+1):-1]

    # выводим результаты
    result.plot(kind = 'barh', figsize=(width, height), title=f'Корреляция Мэтьюса между бинарными признаками и * {target} *');
    if table is True: display(result.to_frame())

#######################################################################################

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

#######################################################################################

def get_top_abs_correlations(df, method, top_number):
    '''
    Функция выводит топ коррелирующих пар признаков
    Параметры:
        df: датасет
        method: метод корреляции
        top_number: количество пар признаков для вывода на экран
    '''
    au_corr = df.corr(method = method).unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False, key=abs)
    au_corr.name='corr'
    print("Top Absolute Correlations")

    return au_corr[0:top_number].to_frame()

#######################################################################################


def target_xi2_test(data, target, width, height):
    '''
    Функция принимает на вход датафрейм, целевой признак, размеры графика и 
    проводит тест Xi-квадрат между целевым признаком и признаками датафрейма, 
    выводит результаты в виде графика, сортируя признаки по убыванию.
    Аргументы функции:
        data: датафрейм
        target: целевой признак
        widh: ширина графика
        height: высота графика
    '''
    # проводим тест Xi-квадрат
    result = pd.Series(chi2(data.drop(target, axis=1), data[target])[0], index=data.drop(target, axis=1).columns)
    result.sort_values(inplace = True)
    # выводим результат
    result.plot(kind = 'barh', figsize=(width, height), title=f'Тест Хi2 признака * {target} *');

#######################################################################################
#######################################################################################

# 4. Функции для третьего этапа (кластеризация):

def get_silhouette_score(cluster_num, X):
    '''
    Функция для рассчета коэффициента силуэта для трех алгоритмов:
    KMeans, AgglomerativeClustering, GaussianMixture.
    Параметры функции:
        cluster_num: количество кластеров
        X: данные в виде таблицы/двухмерного массива
    Функция возвращает:
        score km: коэффициент силуэта алгоритма KMeans
        score_agc: коэффициент силуэта алгоритма AgglomerativeClustering
        score_gm: коэффициент силуэта алгоритма GaussianMixture
    '''
    
    # алгоритм кластеризации KMeans
    km = KMeans(n_clusters=cluster_num, n_init='auto', random_state=random_state)
    # обучаем его на наших данных
    km.fit_predict(X)
    # вычисляем значение коэффициента силуэта
    score_km = silhouette_score(X, km.labels_, metric='euclidean')
   
    # алгоритм кластеризации AgglomerativeClustering
    agc = AgglomerativeClustering(n_clusters=cluster_num)
    # обучаем его на наших данных
    agc.fit_predict(X)
    # вычисляем значение коэффициента силуэта
    score_agc = silhouette_score(X, agc.labels_, metric='euclidean')
   
    # алгоритм кластеризации GaussianMixture
    gm = GaussianMixture(n_components=cluster_num, random_state=random_state)
    # обучаем его на наших данных
    gm_labels = gm.fit_predict(X)
    # вычисляем значение коэффициента силуэта
    score_gm = silhouette_score(X, gm_labels, metric='euclidean')

    return score_km, score_agc, score_gm

#######################################################################################

def get_calinski_harabasz_score(cluster_num, X):
    '''
    Функция для рассчета индекса Калински — Харабаса для алгоритма KMeans, AgglomerativeClustering, GaussianMixture.
    Также расчитывается количество объектов в полученных кластерах.
    Параметры функции:
        cluster_num: количество кластеров
        X: данные в виде таблицы/двухмерного массива
    Функция возвращает:
        score_km: индекс Калински — Харабаса для алгоритма KMeans 
        score_agc: индекс Калински — Харабаса для алгоритма AgglomerativeClustering
        score_gm: индекс Калински — Харабаса для алгоритма GaussianMixture
        obj_cluster_counts_km: количество объектов в полученных кластерах для алгоритма KMeans 
        obj_cluster_counts_agc: количество объектов в полученных кластерах для алгоритма AgglomerativeClustering
        obj_cluster_counts_gm: количество объектов в полученных кластерах для алгоритма GaussianMixture
    '''

    # алгоритм кластеризации KMeans
    km = KMeans(n_clusters=cluster_num, n_init='auto', random_state=random_state)
    # обучаем его на наших данных
    km.fit_predict(X)
    # вычисляем значение индекса
    score_km = calinski_harabasz_score(X, km.labels_)
    # вычисляем количество объектов в кластерах
    unique_km, obj_cluster_counts_km = np.unique(km.labels_, return_counts=True)

    # алгоритм кластеризации AgglomerativeClustering
    agc = AgglomerativeClustering(n_clusters=cluster_num)
    # обучаем его на наших данных
    agc.fit_predict(X)
    # вычисляем значение индекса
    score_agc = calinski_harabasz_score(X, agc.labels_)
    # вычисляем количество объектов в кластерах
    unique_agc, obj_cluster_counts_agc = np.unique(agc.labels_, return_counts=True)

    # алгоритм кластеризации GaussianMixture
    gm = GaussianMixture(n_components=cluster_num, random_state=random_state)
    # обучаем его на наших данных
    gm_labels = gm.fit_predict(X)
    # вычисляем значение индекса
    score_gm = calinski_harabasz_score(X, gm_labels)
    # вычисляем количество объектов в кластерах
    unique_gm, obj_cluster_counts_gm = np.unique(gm_labels, return_counts=True)

    return score_km, score_agc, score_gm, obj_cluster_counts_km, obj_cluster_counts_agc, obj_cluster_counts_gm

#######################################################################################

def get_davies_bouldin_score(cluster_num, X):
    '''
    Функция для рассчетв индекса Дэвиса — Болдина для алгоритмов KMeans, AgglomerativeClustering, GaussianMixture
    Параметры функции:
        cluster_num: количество кластеров
        X: данные в виде таблицы/двухмерного массива
    Функция возвращает:
        score_km: индекс Дэвиса — Болдина для алгоритма KMeans 
        score_agc: индекс Дэвиса — Болдина для алгоритма AgglomerativeClustering
        score_gm: индекс Дэвиса — Болдина для алгоритма GaussianMixture
    '''
    
    # алгоритм кластеризации KMeans
    km = KMeans(n_clusters=cluster_num, n_init='auto', random_state=random_state)
    # обучаем его на наших данных
    km.fit_predict(X)
    # вычисляем значение индекса
    score_km = davies_bouldin_score(X, km.labels_)

    # алгоритм кластеризации AgglomerativeClustering
    agc = AgglomerativeClustering(n_clusters=cluster_num)
    # обучаем его на наших данных
    agc.fit_predict(X)
    # вычисляем значение индекса
    score_agc = davies_bouldin_score(X, agc.labels_)

    # алгоритм кластеризации GaussianMixture
    gm = GaussianMixture(n_components=cluster_num, random_state=random_state)
    # обучаем его на наших данных
    gm_labels = gm.fit_predict(X)
    # вычисляем значение индекса
    score_gm = davies_bouldin_score(X, gm_labels)

    return score_km, score_agc, score_gm

#######################################################################################

def clustering_proba(data):
    '''
    Функция расчитывает индексы Калински-Харабаса (используется функция get_calinski_harabasz_score),
    индексы Дэвиса-Болдина (используется функция get_davies_bouldin_score),
    количество объектов в кластерах (так же используется функция get_calinski_harabasz_score)
    в зависимости от количества кластеров для алгоритмов KMeans, AgglomerativeClustering,
    GaussianMixture.
    Выводит рассчитанные данные в виде диаграмм и сводной таблицы.
    Параметры функции:
        data: массив данных со сниженной размерностью
    '''

    # создадим списки, где будут значения внутренних мер
    KMeans_silhouette_score = [] # коэффициент силуэта для алгоритма KMeans
    KMeans_calinski_harabasz_score = [] # индекс Калински-Харабаса для алгоритма KMeans
    KMeans_davies_bouldin_score = [] # индекс Дэвиса-Болдина для алгоритма KMeans
    AGC_silhouette_score = [] # коэффициент силуэта для алгоритма AgglomerativeClustering
    AGC_calinski_harabasz_score = [] # индекс Калински-Харабаса для алгоритма AgglomerativeClustering
    AGC_davies_bouldin_score = [] # индекс Дэвиса-Болдина для алгоритма AgglomerativeClustering
    GM_silhouette_score = [] # коэффициент силуэта для алгоритма GaussianMixture
    GM_calinski_harabasz_score = [] # индекс Калински-Харабаса для алгоритма GaussianMixture
    GM_davies_bouldin_score = [] # индекс Дэвиса-Болдина для алгоритма GaussianMixture
    # создадим списки с количеством объектов в полученных кластерах 
    KMeans_obj_cluster_counts = [] # количество объектов в кластерах для KMeans
    AGC_obj_cluster_counts = [] # количество объектов в кластерах для AgglomerativeClustering
    GM_obj_cluster_counts = [] # количество объектов в кластерах для GaussianMixture

    # расчитаем коэффициент силуэта для 2-10 кластеров
    for cluster_num in range(2,11):
        # рассчитываем индексы для алгоритмов и получаем количество объектов в кластерах
        score_km, score_agc, score_gm = get_silhouette_score(cluster_num, data)
        # добавляем полученные данные в соответствующие списки
        KMeans_silhouette_score.append(score_km)
        AGC_silhouette_score.append(score_agc)
        GM_silhouette_score.append(score_gm)

    # расчитаем индекс Калински — Харабаса для 2-10 кластеров
    for cluster_num in range(2,11):
        # рассчитываем индексы для обоих алгоритмов и получаем количество объектов в кластерах
        score_km, score_agc, score_gm, obj_cluster_counts_km, obj_cluster_counts_agc, obj_cluster_counts_gm = \
            get_calinski_harabasz_score(cluster_num, data)
        # добавляем полученные данные в соответствующие списки
        KMeans_calinski_harabasz_score.append(score_km)
        AGC_calinski_harabasz_score.append(score_agc)
        GM_calinski_harabasz_score.append(score_gm)

        KMeans_obj_cluster_counts.append(obj_cluster_counts_km)
        AGC_obj_cluster_counts.append(obj_cluster_counts_agc)
        GM_obj_cluster_counts.append(obj_cluster_counts_gm)

    # расчитаем индекс Дэвиса — Болдина для 2-10 кластеров
    for cluster_num in range(2,11):
        # рассчитываем индексы для обоих алгоритмов
        score_km, score_agc, score_gm = get_davies_bouldin_score(cluster_num, data)
        # добавляем полученные данные в соответствующие списки
        KMeans_davies_bouldin_score.append(score_km)
        AGC_davies_bouldin_score.append(score_agc)
        GM_davies_bouldin_score.append(score_gm)

    # визуализируем полученные данные (индексы)

    plt.subplots(nrows=3, ncols=3, figsize=(9, 9), tight_layout=True)
   
    # коэффициент силуэта для алгоритма KMeans
    plt.subplot(3, 3, 1)
    plt.title("Silhouette Score (KMeans)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("SS values", fontsize=10)
    plt.plot([i for i in range(2, 11)], KMeans_silhouette_score, color='r', label='KMeans')
    plt.legend()

    # индекс Калински-Харабаса для алгоритма KMeans
    plt.subplot(3, 3, 2)
    plt.title("Calinski Harabasz Index (KMeans)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("CH index", fontsize=10)
    plt.plot([i for i in range(2, 11)], KMeans_calinski_harabasz_score, color='b', label='KMeans')
    plt.legend()

    # индекс Дэвиса-Болдина для алгоритма KMeans
    plt.subplot(3, 3, 3)
    plt.title("Davies Bouldin Index (KMeans)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("DB index", fontsize=10)
    plt.plot([i for i in range(2, 11)], KMeans_davies_bouldin_score, color='g', label='KMeans')
    plt.legend()

    # коэффициент силуэта для алгоритма AgglomerativeClustering
    plt.subplot(3, 3, 4)
    plt.title("Silhouette Score (AGC)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("SS values", fontsize=10)
    plt.plot([i for i in range(2, 11)], AGC_silhouette_score, color='r', label='AgglomerativeClustering')
    plt.legend()

    # индекс Калински-Харабаса для алгоритма AgglomerativeClustering
    plt.subplot(3, 3, 5)
    plt.title("Calinski Harabasz Index (AGC)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("CH index", fontsize=10)
    plt.plot([i for i in range(2, 11)], AGC_calinski_harabasz_score, color='b', label='AgglomerativeClustering')
    plt.legend()

    # индекс Дэвиса-Болдина для алгоритма AgglomerativeClustering
    plt.subplot(3, 3, 6)
    plt.title("Davies Bouldin Index (AGC)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("DB index", fontsize=10)
    plt.plot([i for i in range(2, 11)], AGC_davies_bouldin_score, color='g', label='AgglomerativeClustering')
    plt.legend()

    # коэффициент силуэта для алгоритма GaussianMixture
    plt.subplot(3, 3, 7)
    plt.title("Silhouette Score (GM)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("SS values", fontsize=10)
    plt.plot([i for i in range(2, 11)], GM_silhouette_score, color='r', label='GaussianMixture')
    plt.legend()

    # индекс Калински-Харабаса для алгоритма GaussianMixture
    plt.subplot(3, 3, 8)
    plt.title("Calinski Harabasz Index (GM)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("CH index", fontsize=10)
    plt.plot([i for i in range(2, 11)], GM_calinski_harabasz_score, color='b', label='GaussianMixture')
    plt.legend()

    # индекс Дэвиса-Болдина для алгоритма GaussianMixture
    plt.subplot(3, 3, 9)
    plt.title("Davies Bouldin Index (GM)")
    plt.xlabel("No. of clusters", fontsize=10)
    plt.ylabel("DB index", fontsize=10)
    plt.plot([i for i in range(2, 11)], GM_davies_bouldin_score, color='g', label='GaussianMixture')
    plt.legend()

    plt.show();

    # выводим значения индексов и количество объектов в кластеров в виде таблицы
    cluster_metric = pd.DataFrame({
        'clust': [i for i in range(2, 11)],

        'KMeans_SS': KMeans_silhouette_score,
        'KMeans_CH': KMeans_calinski_harabasz_score,
        'KMeans_DB': KMeans_davies_bouldin_score,
        'KMeans_object_cluster_counts': KMeans_obj_cluster_counts,

        'AGC_SS': AGC_silhouette_score,
        'AGC_CH': AGC_calinski_harabasz_score,
        'AGC_DB': AGC_davies_bouldin_score,
        'AGC_object_cluster_counts':AGC_obj_cluster_counts,

        'GM_SS': GM_silhouette_score,
        'GM_CH': GM_calinski_harabasz_score,
        'GM_DB': GM_davies_bouldin_score,
        'GM_object_cluster_counts':GM_obj_cluster_counts
        
    })
    display(cluster_metric)

#######################################################################################

def feature_cluster_describe(data, feature, trend=True, bad_region = True):
    '''
    Функция принимает на вход датасет и признак, 
    выводит в разрезе кластеров:
        - распределение признака в виде гистограммы
        - распределение признака в виде коробчатой диаграммы
        - доля значений тренда признака
        - доля значений меток плохого региона по указанному признаку
    '''

    # настройка цвета для визуализации:
    # для долей значений тренда
    colors_trend = {
        0:'#f8ffe5', 
        1: '#ffc43d', 
       -1: '#06d6a0'
    }
    # для долей значений меток плохого региона
    colors_br = {
        0: '#8d99ae',
        1: '#e63946'
    }
    # для коробчатой диаграммы
    palette_box = ["#ffbe0b","#fb5607","#ff006e","#8338ec","#3a86ff"]

    print(f'  \033[1m Гистограмма распределения * {feature} * в разрезе кластеров \033[0m')
   
    # гистограмма распределения непрерывного признака в разрезе кластеров
    g = sns.FacetGrid(data=data, col="cluster")
    g.map(sns.histplot, feature);
    g.set_titles(col_template="cluster {col_name}", fontweight='bold')

    # задаем фигуру и ее размер 
    plt.subplots(nrows=1, ncols=1, figsize=(10, 3), tight_layout=True)
    plt.suptitle(
        f'Распределение * {feature} * и доля значений его производных в разрезе кластеров:',
        fontsize=12, weight='bold', x=0.4)
    
    # коробчатая диаграмма непрерывного признака в разрезе кластеров
    plt.subplot(1,1,1)        
    sns.boxplot(data=data, x=feature, y="cluster", orient="h", palette=palette_box, order=[4,3,2,1,0]).\
        set_title(f'  {feature}', fontsize=15, weight='bold', loc='left');
    plt.show();

    # задаем фигуру и ее размер 
    fig, ax = plt.subplots(1,2, figsize=(10,2), tight_layout=True) # создаем фигуру нужного размера
    
    if trend is True:
        # рассчитываем доли значений тренда признака в процентах в разрезе кластеров
        df_trend = data.groupby('cluster')[feature+'_trend'].value_counts(normalize=True).unstack() * 100
        # выводим диаграмму долей значений тренда признака в разрезе кластеров
        df_trend.plot(kind='barh', stacked='True', color=colors_trend, ax=ax[0]).legend(
            bbox_to_anchor=(1.0, 1.0), fontsize='small')
        ax[0].set_title(f"  {feature+'_trend'}, %", fontsize=12, weight='bold', loc='left')
    if bad_region is True:
        # рассчитываем доли значений меток плохого региона в процентах в разрезе кластеров
        df_br = data.groupby('cluster')[feature+'_bad_region'].value_counts(normalize=True).unstack() * 100
        # выводим диаграмму долей значений меток плохого региона признака в разрезе кластеров
        df_br.plot(kind='barh', stacked='True', color=colors_br, ax=ax[1]).legend(
            bbox_to_anchor=(-0.07, 1.0), fontsize='small')
        ax[1].set_title(f"  {feature+'_bad_region'}, %", fontsize=12, weight='bold', loc='left')

#######################################################################################

def plot_cluster_profile(grouped_data, n_clusters):
    """Функция для визуализации профиля кластеров в виде полярной диаграммы.

    Аргументы:
        grouped_data (DataFrame): таблица, сгруппированная по номерам кластеров с агрегированными характеристиками объектов.
        n_clusters (int): количество кластеров
    Функция возращает:
        fig: фигуру с полярной диаграммой
    """
    # Нормализуем сгруппированные данные, приводя их к масштабу 0-1.
    scaler = MinMaxScaler()
    grouped_data = pd.DataFrame(scaler.fit_transform(grouped_data), columns=grouped_data.columns)
    # Создаём список признаков
    features = grouped_data.columns
    # Создаём пустую фигуру
    fig = go.Figure()
    # В цикле визуализируем полярную диаграмму для каждого кластера
    for i in range(n_clusters):
        # Создаём полярную диаграмму и добавляем её на общий график
        fig.add_trace(go.Scatterpolar(
            r=grouped_data.iloc[i].values, # радиусы
            theta=features, # название засечек
            fill='toself', # заливка многоугольника цветом
            name=f'Cluster {i}', # название — номер кластера
        ))
    # Обновляем параметры фигуры
    fig.update_layout(
        showlegend=True, # отображение легенды
        autosize=False, # устаналиваем свои размеры графика
        width=1300, # ширина (в пикселях)
        height=1000, # высота (в пикселях)
    )
    # Отображаем фигуру
    fig.show()

    # возращаем фигуру
    return fig

#######################################################################################
#######################################################################################

# 5. Словарь стандартных названий регионов
#    взят отсюда и подпрвлен для собственного проекта:
#    https://github.com/DKudryavtsev/RussiaRegions/blob/main/

regions_names_standart = {
    'алтайск': 'Алтайский край',
    'амур': 'Амурская область',

    #'архангел': 'Архангельская область без Ненецкого АО',

    'астрахан':  'Астраханская область',
    'белгород':  'Белгородская область',
    'брянск': 'Брянская область',
    'владимир': 'Владимирская область',
    'волгоград':  'Волгоградская область',
    'вологод': 'Вологодская область',
    'воронеж': 'Воронежская область',
    'еврей':  'Еврейская автономная область',
    'забайкал': 'Забайкальский край',
    'иванов': 'Ивановская область',
    'иркутск':  'Иркутская область',
    'кабардин': 'Кабардино-Балкарская Республика',
    'калининград': 'Калининградская область',
    'калуж': 'Калужская область',
    'камчатск(?!о)': 'Камчатский край',
    'карач': 'Карачаево-Черкесская Республика',
    'кемеров': 'Кемеровская область',
    'киров': 'Кировская область',
    'костром': 'Костромская область',
    'краснодар': 'Краснодарский край',
    'краснояр': 'Красноярский край',
    'курган': 'Курганская область',
    'курск': 'Курская область',
    'ленин': 'Ленинградская область',
    'липецк': 'Липецкая область',
    'магадан': 'Магаданская область',
    'москов': 'Московская область',
    'мурман': 'Мурманская область',

    '(?<!-)ненец': 'Ненецкий автономный округ',

    'нижегород': 'Нижегородская область',
    'новгород': 'Новгородская область',
    'новосибир': 'Новосибирская область',
    '(?<!т)омск': 'Омская область',
    'оренбург': 'Оренбургская область',
    'орлов': 'Орловская область',
    'пенз': 'Пензенская область',
    'пермск(?!о)': 'Пермский край',
    'примор': 'Приморский край',
    'псков': 'Псковская область',
    'адыг': 'Республика Адыгея',
    'алтай(?!с)': 'Республика Алтай',
    'башк': 'Республика Башкортостан',
    'бурят(?!с)': 'Республика Бурятия',
    'дагестан': 'Республика Дагестан',
    'ингуш': 'Республика Ингушетия',
    'калмык': 'Республика Калмыкия',
    'карел': 'Республика Карелия',
    'коми(?![-н])': 'Республика Коми',
    'крым(?!с)': 'Республика Крым',
    'мари': 'Республика Марий Эл',
    'морд': 'Республика Мордовия',
    'саха(?!л)': 'Республика Саха (Якутия)',
    'якут': 'Республика Саха (Якутия)',
    'осет': 'Республика Северная Осетия-Алания',
    'татар': 'Республика Татарстан',
    'тыва': 'Республика Тыва',
    'хакас': 'Республика Хакасия',

    'российская': 'РФ', # добавил Российскую Федерацию
    
    'ростов': 'Ростовская область',
    'рязан': 'Рязанская область',
    'самар': 'Самарская область',
    'саратов': 'Саратовская область',
    'сахалин': 'Сахалинская область',
    'свердлов': 'Свердловская область',
    'смолен': 'Смоленская область',
    'ставрополь': 'Ставропольский край',
    'тамбов': 'Тамбовская область',
    'твер': 'Тверская область',
    'томск': 'Томская область',
    'туль': 'Тульская область',

    #'тюмен': 'Тюменская область без округов',

    'удмурт': 'Удмуртская Республика',
    'ульянов': 'Ульяновская область',
    'хабаровск': 'Хабаровский край',

    'хант': 'Ханты-Мансийский автономный округ - Югра',

    'челябинск': 'Челябинская область',
    'чечен': 'Чеченская Республика',
    'чуваш': 'Чувашская Республика',
    'чукот': 'Чукотский автономный округ',

    'ямал': 'Ямало-Ненецкий автономный округ',

    'ярослав': 'Ярославская область',
    'москва': 'Москва',
    'петербург': 'Санкт-Петербург',
    'севастополь': 'Севастополь'}