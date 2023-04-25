
#####################################################################################
# Модуль содержит функции, которые используются в проекте и имеет следующие разделы:
#   - импорт необходимых библиотек
#   - для работы с датасетом
#   - для EDA и Feature engineering
#   - для отбора, корреляции признаков


# импорт необходимых библиотек
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import cluster
from sklearn.preprocessing import OneHotEncoder


###############################
# для работы с датасетом
##############################

# добавяем в датасет погодные данные
def add_datetime_features(df):
    ''''
    Функция принимает на вход датафрейм и преобразует признаки даты/времени в 
    отдельные признаки даты, часа и дня недели, удаляет старые признаки даты.
    Функция принимает:
        df: датафрейм
    Функция возвращает новые признаки в датафрейм:
        pickup_date: дата включения счетчика
        pickup_hour: час включения счетчика
        pickup_day_of_week: день недели включения счетчика
    Функция удаляет признаки: 
        pickup_datetime: дата выключения счетчика
        dropoff_datetime: дата выключения счетчика
    '''

    # создаем новый признак - дата включения счетчика
    df['pickup_date'] = df['pickup_datetime'].dt.date
    # создаем новый признак - час дня включения счетчика
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    # создаем новый признак - день недели включения счетчика
    df['pickup_day_of_week'] = df['pickup_datetime'].dt.day_of_week

    # удаляем исходные признаки с датами
    df.drop(['pickup_datetime'], axis=1, inplace=True)

    return df


# добавляем в датасет признак праздника в день поездки
def add_holiday_features(df, holiday_df):
    ''''
    Функция принимает на вход 2 таблицы:
    - таблицу с данными о поездках
    - таблицу с данными о праздничных днях
    и возвращает обновленную таблицу с данными о поездках с добавленным в нее столбцом 
    pickup_holiday - бинарным признаком того, начата ли поездка в праздничный день или нет (1 - да, 0 - нет).
    Аргументы функции:
        df: исходная таблица с данными о поездках
        holiday_df: таблица с данными о праздничных днях
    Функция возвращает:
        df: обновленная исходная таблица
    '''
    # в датасете с праздничными днями переименовываем колонку date в pickup_date
    holiday_df.rename(columns={'date': 'pickup_date'}, inplace=True)
    # и переводим эту колонку в формат datetime
    holiday_df['pickup_date'] = pd.to_datetime(holiday_df['pickup_date'], format='%Y-%m-%d')
    # в основном датасете также переводим признак pickup_date в формат datetime
    df['pickup_date'] = pd.to_datetime(df['pickup_date'], format='%Y-%m-%d')

    # объединяем таблицы с основными данными и праздниками
    df = df.merge(
        holiday_df,
        on='pickup_date',
        how='left'
    )
    
    # создаем бинарный признак pickup_holiday: является ли день начала поездки праздником
    df['pickup_holiday'] = df['holiday'].apply(lambda x: 0 if x is np.nan else 1)
    
    # удаляем ненужные колонки в новой таблице
    df.drop(['day', 'holiday'], axis=1, inplace=True)
     
    return df


# добавляем в датасет OSRM-данные
def add_osrm_features(df, osrm_df):
    ''''
    Функция принимает на вход 2 таблицы:
    - с исходными данными о поездках
    - OSRM-данные
    Возвращает обновленый датасет с исходными данными.
    Аргументы функции:
        df: датасет с исходными данными о поездках
        osrm_df: OSRM-данные
    Функция возвращает:
        df: обновленый датасет с исходными данными
    '''

    # объединяем таблицы с основными и OSRM-данными
    df = df.merge(
        osrm_df,
        on='id',
        how='left'
    )
     
    return df


# функция вычисляет расстояние Хаверсина
def get_haversine_distance(lat1, lng1, lat2, lng2):
    '''
    Функция принимает на вход координаты точки (первая точка), где был включен счетчик и где он был
    выключен (вторая точка); возвращает растояние Хаверсина.
    Аргументы функции:
        lat1: широта первой точки
        lng1: долгота первой точки
        lat2: широта второй точки
        lng2: долгота второй точки
    Функция возвращает:
        h: расстояние Хаверсина
    '''
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # радиус земли в километрах
    EARTH_RADIUS = 6371 
    # считаем кратчайшее расстояние h по формуле Хаверсина
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# функция вычисляет направление движения
def get_angle_direction(lat1, lng1, lat2, lng2):
    '''
    Функция принимает на вход координаты точки (первая точка), где был включен счетчик и где он был
    выключен (вторая точка); возвращает направление движения.
    Аргументы функции:
        lat1: широта первой точки
        lng1: долгота первой точки
        lat2: широта второй точки
        lng2: долгота второй точки
    Функция возвращает:
        alpha: направление движения
    '''
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # считаем угол направления движения alpha по формуле угла пеленга
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha


# функция добавляет в датафрейм новые признаки: расстояние Хаверсина и направление движения
def add_geographical_features(df):
    '''
    Функция принимает на вход таблицу с координатами точки (первая точка), где был включен счетчик и где он был
    выключен (вторая точка); возвращает расстояние Хаверсина и направление движения, создавая соответствующие
    новые признаки в исходной таблице.
    Аргументы функции:
        df: исходная таблица
    Функция возвращает:
        df: исходная таблица с новыми признаками haversine_distance и direction
    '''
    # координаты стартовой точки
    lat1 = df['pickup_latitude']
    lng1 = df['pickup_longitude']
    # координаты финишной точки
    lat2 = df['dropoff_latitude']
    lng2 = df['dropoff_longitude']

    # вычисляем расстояние Хаверсина и направление движения, создав новые признаки
    df['haversine_distance'] = get_haversine_distance(lat1, lng1, lat2, lng2)
    df['direction'] = get_angle_direction(lat1, lng1, lat2, lng2)

    return df



# функция добавляет географические кластеры в данные
def add_cluster_features(df):
    '''
    Функция принимает на вход таблицу с данными о поездках.
    Функция проводит обучение алгоритмов кластеризации и возвращает обновленную таблицу 
    с новым признаками: географические кластеры поездок, начала поездки и окончания поездки,
    а также обученные алгоритмы кластеризации.
    Аргументы функции:
        df: таблица с данными о поездках
    Функция возвращает:
        df: обновленная таблица с новыми признаками:
            geo_cluster_trip - кластер поездки
            geo_cluster_pickup - кластер начала поездки
            geo_cluster_dropoff - кластер окончания поездки
    '''
    # создаем обучающую выборку из географических координат всех точек
    coords_trip = np.hstack((df[['pickup_latitude', 'pickup_longitude']],
                             df[['dropoff_latitude', 'dropoff_longitude']]))
    # обучаем алгоритм кластеризации
    kmeans_trip = cluster.KMeans(n_clusters=10, random_state=42)
    kmeans_trip.fit(coords_trip)
    # создаем выборку для предсказаний на основе координат
    coords_trip = np.hstack((df[['pickup_latitude', 'pickup_longitude']],
                             df[['dropoff_latitude', 'dropoff_longitude']]))
    # предсказываем географический кластер поездки
    cluster_predict = kmeans_trip.predict(coords_trip)
    # добавляем признак кластера в данные
    df['geo_cluster_trip'] = cluster_predict

    # создаем обучающую выборку из географических координат 
    coords_pickup = df[['pickup_latitude', 'pickup_longitude']]
    # обучаем алгоритм кластеризации
    kmeans_pickup = cluster.KMeans(n_clusters=10, random_state=42)
    kmeans_pickup.fit(coords_pickup)
    # создаем выборку для предсказаний на основе координат
    coords_pickup = df[['pickup_latitude', 'pickup_longitude']]
    # предсказываем географический кластер поездки
    cluster_predict = kmeans_pickup.predict(coords_pickup)
    # добавляем признак кластера в данные
    df['geo_cluster_pickup'] = cluster_predict

    # создаем обучающую выборку из географических координат 
    coords_dropoff = df[['dropoff_latitude', 'dropoff_longitude']]
    # обучаем алгоритм кластеризации
    kmeans_dropoff = cluster.KMeans(n_clusters=10, random_state=42)
    kmeans_dropoff.fit(coords_dropoff)
    # создаем выборку для предсказаний на основе координат
    coords_dropoff = df[['dropoff_latitude', 'dropoff_longitude']]
    # предсказываем географический кластер поездки
    cluster_predict = kmeans_dropoff.predict(coords_dropoff)
    # добавляем признак кластера в данные
    df['geo_cluster_dropoff'] = cluster_predict
    
    return df, kmeans_trip, kmeans_pickup, kmeans_dropoff


# функция добавляет признаки географических кластеров в тестовые данные
def add_cluster_features_for_test(df, kmeans_trip, kmeans_pickup, kmeans_dropoff):
    '''
    Функция принимает на вход таблицу с тестовыми данными о поездках и
    обученные алгоритмы кластеризации.
    Функция возвращает обновленную таблицу с новым признаками: 
    географические кластеры поездок, начала поездки и окончания поездки.
    Аргументы функции:
        df: таблица с тестовыми данными о поездках
        kmeans_trip: алгоритм кластеризации полных поездок
        kmeans_pickup: алгоритм кластеризации начала поездки
        kmeans_dropoff: алгоритм кластеризации окончания поездки
    Функция возвращает:
        df: обновленная таблица с новыми признаками:
            geo_cluster_trip - кластер поездки
            geo_cluster_pickup - кластер начала поездки
            geo_cluster_dropoff - кластер окончания поездки
    '''
    # создаем выборку для предсказаний на основе координат
    coords_trip = np.hstack((df[['pickup_latitude', 'pickup_longitude']],
                             df[['dropoff_latitude', 'dropoff_longitude']]))
    # предсказываем географический кластер поездки
    cluster_predict = kmeans_trip.predict(coords_trip)
    # добавляем признак кластера в данные
    df['geo_cluster_trip'] = cluster_predict

    # создаем выборку для предсказаний на основе координат
    coords_pickup = df[['pickup_latitude', 'pickup_longitude']]
    # предсказываем географический кластер поездки
    cluster_predict = kmeans_pickup.predict(coords_pickup)
    # добавляем признак кластера в данные
    df['geo_cluster_pickup'] = cluster_predict

    # создаем выборку для предсказаний на основе координат
    coords_dropoff = df[['dropoff_latitude', 'dropoff_longitude']]
    # предсказываем географический кластер поездки
    cluster_predict = kmeans_dropoff.predict(coords_dropoff)
    # добавляем признак кластера в данные
    df['geo_cluster_dropoff'] = cluster_predict
    
    return df


# функция добавляет признаки погоды начала поездки
def add_weather_features(df, weather_df):
    '''
    Функция принимает на вход две таблицы:
    - таблицу с данными о поездках
    - таблицу с данными о погодных условиях на каждый час
    Функция возвращает обновленную таблицу с данными о поездках с добавленными в нее 5 столбцами:
    - temperature - температура;
    - visibility - видимость;
    - wind speed - средняя скорость ветра;
    - precip - количество осадков;
    - events - погодные явления.
    Аргументы функции:
        df: таблица с данными о поездках
        weather_df: таблица с данными о погодных условиях на каждый час
    Функция возвращает:
        df: обновленный датафрейм
    '''
    # объединяем основные данные с погодными данными
    df = df.merge(
        weather_df,
        on=['pickup_date','pickup_hour'], # по двум колонкам: дате и часу начала поездки
        how='left'
    )
     
    return df


# функция заполняет пропуски в погодных данных
def fill_null_weather_data(df):
    '''
    Функция принимает на вход таблицу с данными о поездках и 
    заполняет пропущенные значения.
    Аргумент функции:
        df: таблица с данными о поездках
    Функция возвращает:
        df: обновленную таблицу с заполненнными пропусками
    '''
    # список погодных признаков с пропусками
    weather_features = ['temperature', 'visibility', 'wind speed', 'precip']
    # список OSRM признаков с пропусками
    OSRM_features = ['total_distance', 'total_travel_time', 'number_of_steps']

    # заполняем пропуски погодных признаков медианами с группировкой по дате начала поездки
    for feature in weather_features:
        df[feature] = df[feature].fillna(
        df.groupby('pickup_date')[feature].transform('median')
        )

    # заполняем пропуски OSRM признаков их медианами 
    for feature in OSRM_features:
        df = df.fillna({feature: df[feature].median()})

    # пропуски в признаке events заполняем 'None'
    df['events'] = df['events'].apply(lambda x: 'None' if x is np.nan else x)

    return df


################################
# для EDA и Feature engineering
################################

# для визуализации непрерывных числовых признаков
def num_features_visual(df, *features):
    ''' 
    Функция принимает датафрейм и непрерывный числовой признак/и и визуализирует его/их.
    Аргументы функции:
        df: датафрейм с данными
        *features:  признак или несколько признаков
    Функция возвращает:
        диаграмму в виде гистограммы распределения и коробки с усами;
        каждый признак - одна строка с гистограммой и коробкой;
    '''

    # получим количество указанных признаков
    n = len(features)
    # если указан один признак
    if n == 1:
        # выведем диаграмму признака
        fig, axes = plt.subplots(n, 2) # фигура + (n * 2) координатных плоскостей - 1 строка
        # первый график в строке - гистограмма
        histogram = sns.histplot(data=df, x=features[0], ax=axes[0]) # график
        histogram.set_title(f'*{features[0]}* histogram') # подпись графика
        # второй график в строке - ящик с усами
        boxplot = sns.boxplot(data=df, x=features[0], ax=axes[1]) # график
        boxplot.set_title(f'*{features[0]}* boxplot') # подпись графика
        # выравнивание графиков
        plt.tight_layout() 
    # если признаков больше одного
    else:
        # выведем диаграммы признаков
        fig, axes = plt.subplots(n, 2) # фигура + (n * 2) координатных плоскостей
        # создаем цикл для всех признаков 
        for i,feature in enumerate(features):
            # первый график в строке - гистограмма
            histogram = sns.histplot(data=df, x=feature, ax=axes[i][0]) # график
            histogram.set_title(f'*{feature}* histogram') # подпись графика
            # второй график в строке - ящик с усами
            boxplot = sns.boxplot(data=df, x=feature, ax=axes[i][1]) # график
            boxplot.set_title(f'*{feature}* boxplot') # подпись графика
            # выравнивание графиков
            plt.tight_layout() 



# для визуализации непрерывных числовых признаков
def num_features_visual_scatter(df, feature):
    '''
    Функция принимает на вход таблицу и признак, возвращает 
    диаграмму рассеяния признака.
    Аргументы функции:
        df: таблица/датафрейм
        feature: признак
    Функция возвращает:
        диаграмму рассеяния указанного признака
    '''
    sns.scatterplot(x=df[feature].index, y=(df[feature])).\
    set(
    title=f'Диаграмма рассеяния признака *{feature}*', 
    xlabel='поездки', 
    ylabel=feature
    );



# для визуализации связи категориального признака с таргетом
def cat_feature_target_visual(df, feature, target):
    ''' 
    Функция принимает датафрейм и категориальный признак с таргетом и 
    визуализирует таргет в виде гистограммы распределения и коробки с усами 
    в разрезе значений категориального признака.
    Аргументы функции:
        df: датафрейм с данными
        feature: категориальный признак
        target: таргет/целевая переменная
    Функция возвращает:
        диаграмму таргета в виде гистограммы распределения и коробки с усами 
        в разрезе значений указанного категориального признака;
    '''
    # получаем значения категориальнго признака в виде списка
    feature_values = df[feature].value_counts().index.tolist()

    # получим количество признаков
    n = len(feature_values)
    
    # выведем диаграмму
    fig, axes = plt.subplots(n, 2) # фигура + (n * 2) координатных плоскостей
    # создаем цикл для всех признаков 
    for i,feature_value in enumerate(feature_values):
        mask = df[feature] == feature_value # маска по значению категориального признака
        # первый график в строке - гистограмма
        histogram = sns.histplot(data=df[mask], x=target, ax=axes[i][0]) # график
        histogram.set_title(f'*{feature}* = *{feature_value}*    histogram') # подпись графика
        # второй график в строке - ящик с усами
        boxplot = sns.boxplot(data=df[mask], x=target, ax=axes[i][1]) # график
        boxplot.set_title(f'*{feature}* = *{feature_value}*    boxplot') # подпись графика
        # выравнивание графиков
        plt.tight_layout() 



# еще вариант визуализации связи категориального признака с таргетом
def cat_feature_target_visual_version_2(df, feature, h_size = 10, v_size = 5):
    ''' 
    Функция принимает категориальный признак и размеры диаграммы, а возвращает
    распределения значений таргета (trip_duration и trip_duration_log) в виде
    коробок с усами без выбросов в разрезе значений категориального признака.
    Аргументы функции:
        df: датасет
        feature: категориальный признак
        h_size: ширина диаграммы (по умолчанию = 10)
        v_size: высота диаграммы (по умолчанию = 5)
    Функция возвращает:
        диаграмму распределения таргета (trip_duration и trip_duration_log) в виде 
        коробки с усами без выбросов в разрезе значений указанного категориального признака;
    '''
    # визуализируем влияние признака feature на длительность поездки
    plt.figure(figsize=(h_size, v_size)) # размер диаграммы
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":10}) # размер шрифтов
    plt.suptitle(f"Зависимость от значений признака *{feature}*") # подпись диаграммы
    plt.subplot(1, 2, 1) # первый график в строке
    sns.boxplot(x = feature, y="trip_duration", data = df, showfliers = False, palette='husl') # график
    plt.title("длительности поездки") # подпись графика
    plt.subplot(1, 2, 2) # второй график в строке
    sns.boxplot(x = feature, y="trip_duration_log", data = df, showfliers = False, palette='husl') # график
    plt.title("длительности поездки (log)") # подпись графика
    plt.show();



# и еще вариант визуализации категориального признака через количество записей,
# медианного и среднего значения таргета в разрезе значений этого признака
def cat_feature_target_visual_version_3(df, feature, h_size = 15, v_size = 5):
    ''' 
    Функция принимает категориальный признак и размеры диаграммы, а возвращает
    в виде столбчатой диаграммы количество поездок, медианное и среднее значение 
    длительности поездок в разрезе значений категориального признака.
    Аргументы функции:
        df: датасет
        feature: категориальный признак
        h_size: ширина диаграммы (по умолчанию = 15)
        v_size: высота диаграммы (по умолчанию = 5)
    Функция возвращает:
        столбчатую диаграмму на которой представлены данные о количестве поездок, 
        медианном и среднем значениии длительности поездок в разрезе значений 
        категориального признака;
    '''
    # визуализируем зависимость количества и длительности поездок от значений признака
    plt.figure(figsize=(h_size, v_size)) # задаем размер диаграммы
    sns.set_context("paper", rc={"font.size":10,"axes.titlesize":10,"axes.labelsize":10}) # размер шрифтов
    plt.suptitle(f"Зависимость количества и длительности поездок от значений признака *{feature}*") # подпись диаграммы
    plt.subplot(1, 3, 1) # первый график в строке
    sns.countplot(df[feature], palette='husl'); # график
    plt.title("Количество поездок") # подпись графика
    plt.subplot(1, 3, 2) # второй график в строке
    # группируем данные по значениям категориального признака и считаем медиану длительности поездок
    grouped_median = df.groupby([feature]).agg({'trip_duration':'median'}).reset_index()
    sns.barplot(x=grouped_median[feature], y=grouped_median['trip_duration'], palette='husl'); # график
    plt.title("Медианная длительность поездки") # подпись графика
    plt.subplot(1, 3, 3) # третий график в строке
    # группируем данные по значениям категориального признака и считаем среднее длительности поездок
    grouped_mean = df.groupby([feature]).agg({'trip_duration':'mean'}).reset_index()
    sns.barplot(x=grouped_mean[feature], y=grouped_mean['trip_duration'], palette='husl'); # график
    plt.title("Средняя длительность поездки") # подпись графика
    plt.show();



# делает то же, что и функция выше, только возращает данные в виде таблицы
def feature_median_and_mean_target(df, feature):
    ''' 
    Функция принимает категориальный признак а возвращает
    таблицу с данными о количестве поездок и их медианной и средней
    длительности в разрезе значений категориального признака.
    Аргументы функции:
        df: датасет
        feature: категориальный признак
    Функция возвращает:
        display(grouped_feature): выводит таблицу с данными о количестве поездок 
        и их медианной и средней длительности в разрезе значений категориального признака;
    '''
    # группируем данные по значениям признака и считаем медианное значение длительности поездок
    grouped_feature_median = df.groupby([feature]).agg({'trip_duration':'median'}).\
        reset_index().rename({'trip_duration': 'trip_duration'+'_median'}, axis='columns')
    # группируем данные по значениям признака и считаем среднее значение длительности поездок
    grouped_feature_mean = df.groupby([feature]).agg({'trip_duration':'mean'}).\
        reset_index().rename({'trip_duration': 'trip_duration'+'_mean'}, axis='columns')
    # группируем данные по значениям признака и считаем количество поездок
    grouped_feature_count = df.groupby([feature]).agg({'trip_duration':'count'}).\
        reset_index().rename({'trip_duration': 'count'}, axis='columns')
    # собираем полученные данные в одну таблицу
    grouped_feature = pd.merge(grouped_feature_count, grouped_feature_median)
    grouped_feature = pd.merge(grouped_feature, grouped_feature_mean)
    
    return display(grouped_feature)



# для визуализации пар значений двух категориальных признаков
def heatmap_cat_features(df, feature_1, feature_2, h_size=20, v_size=5):
    ''' 
    Функция принимает 2 категориальных признака и размеры диаграммы, а возвращает
    в виде тепловых карт количество поездок, медианное и среднее значение 
    длительности поездок в разрезе значений категориальных признаков.
    Аргументы функции:
        df: датасет
        feature_1: первый категориальный признак
        feature_2: второй категориальный признак
        h_size: ширина диаграммы (по умолчанию = 20)
        v_size: высота диаграммы (по умолчанию = 5)
    Функция возвращает:
        тепловые карты с количеством поездок, медианным и средним значением 
        длительности поездок в разрезе значений категориальных признаков;
    '''
    # создаем таблицу с количеством поездок в разрезе значений категориальных признаков
    pivot_count = df.pivot_table(
        values='trip_duration',
        index=feature_1,
        columns=feature_2,
        aggfunc='count',
    )
    # создаем таблицу с медианнми значениями длительности поездок в разрезе значений категориальных признаков
    pivot_median = df.pivot_table(
        values='trip_duration',
        index=feature_1,
        columns=feature_2,
        aggfunc='median',
    )
    # создаем таблицу со средними значениями длительности поездок в разрезе значений категориальных признаков
    pivot_mean = df.pivot_table(
        values='trip_duration',
        index=feature_1,
        columns=feature_2,
        aggfunc='mean',
    )
    # визуализируем полученные данные
    plt.figure(figsize=(h_size, v_size)) # указываем размер диаграммы
    sns.set_context("paper", rc={"font.size":12,"axes.titlesize":12,"axes.labelsize":12}) # размер шрифтов
    plt.subplot(1,3,1) # первая тепловая карта в строке
    plt.title("Количество поездок") # подпись первой тепловой карты
    sns.heatmap(pivot_count, annot=True, fmt='g', cmap='coolwarm', annot_kws={"size":10}) # тепловая карта
    plt.subplot(1,3,2) # вторая тепловая карта в строке
    plt.title("Медианная длительность поездок") # подпись второй тепловой карты
    sns.heatmap(pivot_median, annot=True, fmt='g', cmap='coolwarm', annot_kws={"size":10}) # тепловая карта
    plt.subplot(1,3,3) # третья тепловая карта в строке
    plt.title("Средняя длительность поездок") # подпись третьей тепловой карты
    sns.heatmap(pivot_mean, annot=True, fmt='.0f', cmap='coolwarm', annot_kws={"size":10}) # тепловая карта
    plt.show();



# функция для создания пар признаков
def combinantorial(lst):
    """
    Функция для поиска всех возможных пар в списке, 
    где порядок не имеет значения, т.е. (a,b) = (b, a).
    Возвращает список кортежей с парами.
    """
    index = 1
    pairs = []
    for element1 in lst:
        for element2 in lst[index:]:
            pairs.append((element1, element2))
        index += 1

    return pairs


###################################################
# для отбора, корреляции признаков
###################################################

# две функции для определения сильно коррелирующих пар признаков
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, method='pearson', n=5):
    '''
    Функция выводит топ коррелирующих пар признаков
    Параметры:
        df: датасет
        method: метод (по умолчаню корреляция Пирсона)
        n: количество пар признаков для вывода на экран
    '''
    au_corr = df.corr(method = method).abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    print("Top Absolute Correlations")
    print('---')
    return au_corr[0:n]



