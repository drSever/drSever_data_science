# импорт библиотек
import numpy as np # для матричных вычислений
import pandas as pd # для анализа и предобработки данных
import matplotlib.pyplot as plt # для визуализации
import seaborn as sns # для визуализации
from sklearn.preprocessing import OneHotEncoder # для OHE
from sklearn import model_selection # для разделения выборки
from sklearn import metrics # метрики


# настройка цвета при визуализации
colors = {
    'yes':'#9ACD32', # открывшие депозит - зеленый цвет
    'no': '#CD5C5C' # неоткрвышие депозит - красный цвет
}

##########################
# функции для визуализации
##########################

def visual_cat_feature(data, feature, target='deposit', h_size=10, v_size=3):
    """
    Функция выводит 2 диаграммы категориального признака:
        №1: разделение признака по группам и долям классов;
        №2: количество значений в каждой группе признака в разрезе долей классов.
    Параметры:
        data: датафрейм;
        feature: название признака (столбца) в виде строки;
        target: целевой признак (столбец) в виде строки;
        h_size: ширина графика;
        v_size: высота графика;
    """
    # группируем признак по значениям и выделяем доли классов(%)
    types_norm_true = data.groupby(feature)[target].value_counts(normalize=True).unstack() * 100
    # группируем признак по значениям, выделяем классы и получаем количество значений в группах
    types_norm_false = data.groupby(feature)[target].value_counts(normalize=False).unstack()

    # визуализируем полученные данные
    fig, ax = plt.subplots(1,2, figsize=(h_size,v_size)) # создаем фигуру нужного размера
    # выводим первую диаграмму
    types_norm_true.plot(kind='bar', stacked='True', color=colors, legend=False, ax=ax[0])
    # выводим вторую диаграмму
    types_norm_false.plot(kind='bar', stacked='True', color=colors, ax=ax[1])
    # подписываем обе диаграммы
    ax[0].title.set_text('Доля классов в группах ' + '*' + feature + '*' + ', %')
    ax[1].title.set_text('Количество клиентов в группах ' + '*' + feature + '*')
    fig.show();
    
def visual_num_feature(data, feature, target='deposit', h_size=10, v_size=3):
    """
    Функция выводит 2 диаграммы непрерывного числового признака:
        №1: гистограмма распределения;
        №2: коробки с усами в разрезе целевого признака.
    Параметры:
        data: датафрейм;
        feature: название признака (столбца) в виде строки;
        target: целевой признак (столбец) в виде строки;
        h_size: ширина графика;
        v_size: высота графика;
    """
    # создаем фигуру нужного размера
    fig, axes = plt.subplots(1, 2, figsize=(h_size, v_size))
    #Строим гистограмму распределения признака 
    histplot = sns.histplot(data=data, x=feature, ax=axes[0])
    #Добавляем подпись графику
    histplot.set_title('Гистограмма распределения признака *{}*'.format(feature))
    #Строим диаграмму ящик с усами
    boxplot = sns.boxplot(data=data, x=feature, y=target, palette=colors, ax=axes[1])
    #Добавляем подпись графику
    boxplot.set_title('{} VS {}'.format(feature, target));
    plt.tight_layout() #выравнивание графиков
    
def heatmap_pair_features(data, feature_1, feature_2, bound, bal_left, bal_right, target='deposit', h_size=10, v_size=3):
    """
    Функция выводит 2 тепловые карты количества значений пары признаков в разрезе классов:
        №1: в разрезе целевого признака (класс 1);
        №2: в разрезе целевого признака (класс 0).
    Параметры:
        data: датафрейм;
        feature_1: название признака (столбца) в виде строки;
        feature_2: название признака (столбца) в виде строки;
        bound: нижняя граница количества значений класса 1 в паре признаков;
        bal_left: левая граница доли дисбаланса классов 
        bal_right: правая граница доли дисбаланса классов
        target: целевой признак (столбец) в виде строки;
        h_size: ширина графика;
        v_size: высота графика;
    """
    # маска для класса 1 ('yes')
    mask_yes = (data[target] == 1) | (data[target] == 'yes')
    # маска для класса 0 ('no')
    mask_no = (data[target] == 0) | (data[target] == 'no')

    # группируем оба признака по маске класса 1
    df_target_yes = data[mask_yes].groupby(feature_1)[feature_2].value_counts(normalize=False).unstack()
    # группируем оба признака по маске класса 0
    df_target_no = data[mask_no].groupby(feature_1)[feature_2].value_counts(normalize=False).unstack()
    
    # складываем сгруппированные таблицы и вычисляем доли их значений (определяем дисбаланс классов)
    df_new_1 = (df_target_yes+1) / (df_target_no+df_target_yes+1)
    df_new_2 = (df_target_no+1) / (df_target_no+df_target_yes+1)
    
    # если доли классов хотя бы одного признака находятся в промежутке указанных границ дисбаланса
    if ((bal_left < df_new_1.values).any() and (df_new_1.values < bal_right).any()) or \
        ((bal_left < df_new_2.values).any() and (df_new_2.values < bal_right).any()):
    
        # если в сгруппированной таблице по маске класса 1 есть хотя бы одно значение > bound
        if (df_target_yes.values > bound).any():
            # визуализируем данные
            fig, axes = plt.subplots(1, 2, figsize=(h_size, v_size)) # создаем фигуру нужного размера
            # первая тепловая карта в разрезе класса 1
            heatmap_1 = sns.heatmap(df_target_yes, annot=True, fmt='g', ax=axes[0]) 
            heatmap_1.set_title(f'Heatmap *{feature_1}* / *{feature_2}*\n(target = yes = 1)', fontdict={'fontsize':12}, pad=12);
            # вторая тепловая карта в разрезе класса 0
            heatmap_2 = sns.heatmap(df_target_no, annot=True, fmt='g', ax=axes[1]) 
            heatmap_2.set_title(f'Heatmap *{feature_1}* / *{feature_2}*\n (target = no = 0)', fontdict={'fontsize':12}, pad=12);
            fig.show();
        else:
            print('there are no features that satisfy the conditions')
    else:
        print('there are no features that satisfy the conditions')

def plot_learning_curve(model, X, y, cv, scoring="f1", ax=None, title=""):
    ''''
    Функция строит кривую обучения модели.
    Параметры:
        model: название модели
        X: матрица наблюдений X
        y: вектор ответов y
        cv: кросс-валидатор
        scoring: целевая метрика (f1-score по умолчанию)
        ax: для координатной плоскости (если необходимо)
        title: подпись для графика
    '''
    # Вычисляем координаты для построения кривой обучения
    train_sizes, train_scores, valid_scores = model_selection.learning_curve(
        estimator=model,  # модель
        X=X,  # матрица наблюдений X
        y=y,  # вектор ответов y
        cv=cv,  # кросс-валидатор
        scoring=scoring,  # метрика
    )
    # Вычисляем среднее значение по фолдам для каждого набора данных
    train_scores_mean = np.mean(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    # Если координатной плоскости не было передано, создаём новую
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))  # фигура + координатная плоскость
    # Строим кривую обучения по метрикам на тренировочных фолдах
    ax.plot(train_sizes, train_scores_mean, label="Train")
    # Строим кривую обучения по метрикам на валидационных фолдах
    ax.plot(train_sizes, valid_scores_mean, label="Valid")
    # Даём название графику и подписи осям
    ax.set_title("Learning curve: {}".format(title))
    ax.set_xlabel("Train data size")
    ax.set_ylabel("Score")
    # Устанавливаем отметки по оси абсцисс
    ax.xaxis.set_ticks(train_sizes)
    # Устанавливаем диапазон оси ординат
    ax.set_ylim(0, 1)
    # Отображаем легенду
    ax.legend()

def dependence_metrics_threshold(model, X_test, y_test):
    '''
    Функция строит график зависимости метрик от порога верятности и 
    возвращает вектор вероятности принадлежности к классу 1 в виде pandas Series
    Параметры:
        model:  название модели
        X_test: тестовая выборка
        y_test: таргет тестовой выборки
    '''
    # получаем массив вероятности принадлежности классу 1
    y_test_proba_pred = model.predict_proba(X_test)[:, 1]
    # Для удобства завернем numpy-массив в pandas Series
    y_test_proba_pred = pd.Series(y_test_proba_pred)
    # Создадим списки, в которых будем хранить значения метрик 
    recall_scores = []
    precision_scores = []
    f1_scores = []
    # Сгенерируем набор вероятностных порогов в диапазоне от 0.1 до 1
    thresholds = np.arange(0.1, 1, 0.05)
    # В цикле будем перебирать сгенерированные пороги
    for threshold in thresholds:
        # клиентов, для которых вероятность открытия депозита > threshold относим к классу 1
        # в противном случае - к классу 0
        y_test_pred = y_test_proba_pred.apply(lambda x: 1 if x>threshold else 0)
        # Считаем метрики и добавляем их в списки
        recall_scores.append(metrics.recall_score(y_test, y_test_pred))
        precision_scores.append(metrics.precision_score(y_test, y_test_pred))
        f1_scores.append(metrics.f1_score(y_test, y_test_pred))

    # Визуализируем метрики при различных threshold
    fig, ax = plt.subplots(figsize=(10, 4)) # фигура + координатная плоскость
    # Строим линейный график зависимости recall от threshold
    ax.plot(thresholds, recall_scores, label='Recall')
    # Строим линейный график зависимости precision от threshold
    ax.plot(thresholds, precision_scores, label='Precision')

    # Строим линейный график зависимости F1 от threshold
    ax.plot(thresholds, f1_scores, label='F1-score')
    # Даем графику название и подписи осям
    ax.set_title('Recall/Precision dependence on the threshold')
    ax.set_xlabel('Probability threshold')
    ax.set_ylabel('Score')
    ax.set_xticks(thresholds)
    ax.legend();
    
    return y_test_proba_pred

def plot_pr_curve(model, X_train, y_train, cv, model_name):
    '''
    Функция принимает на вход модель, тренировочную выборку с таргетом,
    кросс-валидатор и название модели, а возвращает график PR-кривой обучения,
    лучший порог вероятнояти с F1-score и PR AUC
    Параметры:
        model: модель для которой строим кривую обучения
        X_train: тренировочная выборка
        y_tarin: таргет тренировочной выборки
        cv: кросс-валидатор
        model_name: название модели для подписи графика в виде строки
    '''
    #Делаем предсказание вероятностей на кросс-валидации
    y_cv_proba_pred = model_selection.cross_val_predict(
        estimator=model, 
        X=X_train, 
        y=y_train, 
        cv=cv, 
        method='predict_proba'
    )

    #Выделяем столбец с вероятностями для класса 1 (True)
    y_cv_proba_pred = y_cv_proba_pred[:, 1]

    #Вычисляем координаты PR-кривой
    precision, recall, thresholds = metrics.precision_recall_curve(y_train, y_cv_proba_pred)

    #Вычисляем F1-score при различных threshold
    f1_scores = (2 * precision * recall) / (precision + recall)
    #Определяем индекс максимума
    idx = np.argmax(f1_scores)
    print('Best threshold = {:.2f}, F1-Score = {:.2f}'.format(thresholds[idx], f1_scores[idx]))
    print('PR AUC: {:.2f}'.format(metrics.auc(recall, precision)))
 
    #Строим PR-кривую
    fig, ax = plt.subplots(figsize=(10, 5)) #фигура + координатная плоскость
    #Строим линейный график зависимости precision от recall
    ax.plot(precision, recall, label=f'{model_name} PR')
    #Отмечаем точку максимума F1
    ax.scatter(precision[idx], recall[idx], marker='o', color='black', label='Best F1 score')
    #Даём графику название и подписываем оси
    ax.set_title('Precision-recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    #Отображаем легенду
    ax.legend();


#############################
# функции для поиска выбросов
#############################
                
def outliers_iqr_mod(data, feature, left=1.5, right = 1.5):
    """
    Функция находит выбросы по методу Тьюки в признаке (столбце) датафрейма. 
    Возвращает новый датафрейм со строками, содержащими выбросы.
    Параметры:
        data: датафрейм;
        feature: признак (столбец) в виде строки;
        left: число IQR влево;
        right: число IQR вправо;
    """
    # расчитываем границы за которыми будут выбросы
    x = data[feature] 
    quartile_1, quartile_3 = x.quantile(0.25), x.quantile(0.75), 
    iqr = quartile_3 - quartile_1 
    lower_bound = quartile_1 - (iqr * left) 
    upper_bound = quartile_3 + (iqr * right) 
    
    # выводим нижнюю и верхнюю границу
    print(f'lower bound = {round(lower_bound)}')
    print(f'upper bound = {round(upper_bound)}')
    
    # создаем новый датафрейм с выбросами
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
    
    # выводим количество найденных выбросов
    print('total outliers in ' + '*' + feature + '*: {}'.format(outliers.shape[0]))
    
    return outliers

def outliers_z_score(data, feature, log_scale=False, y=1):
    '''
    Функция принимает на вход признак, при необходимости логарифмирует его,
    ищет выбросы по методу Z-отклонений и возвращает датасет с выбросами вместе с
    границами этих выбросов
    Параметры:
        data: датасет
        feature: признак датасета
        log_scale: True или False - необходимость логарифмировать признак
        y: прибавить к значениям признака необходимое чесло (если значение признака <=0)
    '''
    # проверяем необходимость логарифмизации признака
    if log_scale:
        x = np.log(data[feature]+y)
    else:
        x = data[feature]
    
    # расчитываем границы выбросов методом Z-отклонений
    mu = x.mean()
    sigma = x.std()
    lower_bound = mu - 3 * sigma
    upper_bound = mu + 3 * sigma
    
    # выводим нижнюю и верхнюю границу
    print(f'lower bound = {lower_bound}')
    print(f'upper bound = {upper_bound}')
    
    # создаем новый датафрейм с выбросами
    outliers = data[(x < lower_bound) | (x > upper_bound)]
    
    # выводим количество найденных выбросов
    print('total outliers in ' + '*' + feature + '*: {}'.format(outliers.shape[0]))
    
    return outliers, lower_bound, upper_bound

################
# прочие функции
################

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

def encode_cat_features(columns_to_change, X_train, X_test, y_train):
    '''
    Функция выполняет OneHotEncoding категориальных признаков тренировочной и
    тестовой выборок по отдельности (!)
    Параметры:
        columns_to_change: признаки для кодирования в виде списка
        X_train: тренировочная выборка 
        X_test:  тестовая выборка
        y_train: таргет тренировочной выборки
    '''
    # Обучаем энкодер на тренировочной выборке и применяем преобразование. 
    # Результат переводим в массив
    one_hot_encoder = OneHotEncoder()
    X_train_onehot = one_hot_encoder.fit_transform(X_train[columns_to_change]).toarray()
    X_test_onehot = one_hot_encoder.transform(X_test[columns_to_change]).toarray()
    
    # сохраним полученные названия новых колонок в отдельную переменную
    columns = one_hot_encoder.get_feature_names(columns_to_change)
    
    # Теперь у нас есть массив закодированных признаков и наша изначальная таблица. 
    # Чтобы соединить эти данные, переведём массив в формат DataFrame.
    X_train_onehot_df = pd.DataFrame(X_train_onehot, columns=columns)
    X_test_onehot_df = pd.DataFrame(X_test_onehot, columns=columns)
    
    # Переустановим индексацию в таблицах, применив подряд сразу два метода: 
    # reset_index() — для изменения индексов с рандомных на последовательные от 0 до n и 
    # drop(['index'], axis = 1) — для удаления образовавшегося столбца 'index'.
    X_train = X_train.reset_index().drop(['index'], axis = 1)
    X_test = X_test.reset_index().drop(['index'], axis = 1)
    y_train = y_train.reset_index().drop(['index'], axis = 1)
    
    # Объединяем таблицы и удаляем старые категориальные признаки
    X_train_new = pd.concat([X_train, X_train_onehot_df], axis=1)
    X_test_new = pd.concat([X_test, X_test_onehot_df], axis=1)
  
    X_train_new = X_train_new.drop(columns=columns_to_change)
    X_test_new = X_test_new.drop(columns=columns_to_change)

    return X_train_new, X_test_new

