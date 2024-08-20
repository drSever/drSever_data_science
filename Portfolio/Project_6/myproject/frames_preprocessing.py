# импорт необходимых библиотек

import numpy as np
import math

import os
from pathlib import Path
from tqdm import tqdm

import cv2

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_keypoints
import torchvision.transforms.functional as F

# задаем ключевые точки
coco_keypoints = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# задаем скелет
connect_skeleton = [

        [coco_keypoints.index('right_eye'), coco_keypoints.index('nose')],
        [coco_keypoints.index('right_eye'), coco_keypoints.index('right_ear')],
        [coco_keypoints.index('left_eye'), coco_keypoints.index('nose')],
        [coco_keypoints.index('left_eye'), coco_keypoints.index('left_ear')],
        [coco_keypoints.index('right_shoulder'), coco_keypoints.index('right_elbow')],
        [coco_keypoints.index('right_elbow'), coco_keypoints.index('right_wrist')],
        [coco_keypoints.index('left_shoulder'), coco_keypoints.index('left_elbow')],
        [coco_keypoints.index('left_elbow'), coco_keypoints.index('left_wrist')],
        [coco_keypoints.index('right_hip'), coco_keypoints.index('right_knee')],
        [coco_keypoints.index('right_knee'), coco_keypoints.index('right_ankle')],
        [coco_keypoints.index('left_hip'), coco_keypoints.index('left_knee')],
        [coco_keypoints.index('left_knee'), coco_keypoints.index('left_ankle')],
        [coco_keypoints.index('right_shoulder'), coco_keypoints.index('left_shoulder')],
        [coco_keypoints.index('right_hip'), coco_keypoints.index('left_hip')],
        [coco_keypoints.index('right_shoulder'), coco_keypoints.index('right_hip')],
        [coco_keypoints.index('left_shoulder'), coco_keypoints.index('left_hip')]

]

# зададим вспомогательные функции

def get_keypoints(keypoints):
  """
  Функция принисает на вход тензор с координатами ключевых точек и
  возращает пары координат [x,y] в виде массива [[x1,y1], [x2,y2], ...]
  """
  keypoints_list = []
  for i in range(len(keypoints[0])):
      x = round(float(keypoints[0][i][0].data),4)
      y = round(float(keypoints[0][i][1].data),4)
      keypoints_list.append([x,y])

  return np.asarray(keypoints_list)


def weight_distance(pose1, pose2, conf1):
    """
    Функция получает на вход в виде массивов ключевые точки входа и модели,
    а также score каждой точки входной позы в виде списка.
    Функция возвращает значения взвешенного совпадения по осям, т.е. ошибки по
    каждой оси.
    """
    # D(U,V) = (1 / sum(conf1)) * sum(conf1 * ||pose1 - pose2||) = sum1 * sum2

    sum1 = 1 / np.sum(conf1)
    sum2 = 0

    for i in range(len(pose1)):
        # каждый индекс i имеет x и y, у которых одинаковая оценка достоверности
        conf_ind = math.floor(i / 2)
        sum2 = conf1[conf_ind] * abs(pose1[i] - pose2[i])

    weighted_dist = sum1 * sum2

    return weighted_dist

# зададим основную функцию для обработки исходных фреймов
def frames_preprocessing(PATH_TO_FRAMES, PATH_TO_OUTPUT_FRAMES, model, transforms, text_color, frames_count):
    """
    Функция производит обработку фреймов, полученных из исходного видео и сохраняет их. Расчитывает метрики. 
    Возвращает разрешение фреймов (ширина и высота)

    Args:
        PATH_TO_FRAMES : путь к исходным фреймам из исходного видео
        PATH_TO_OUTPUT_FRAMES : путь для сохранения обработанных фреймов 
        model : модель для обработки фреймов
        transforms : трансформации для исходных фреймов перед подачей в модель
        text_color : цвет текста метрик на итоговом видео
        frames_count : количество исходных фреймов из исходного видео

    Returns:
        width : ширина фрейма в пикселях
        height : высота фрейма в пикселях
    """

    # модель в режим инференса
    model.eval()

    # обрабатываем последовательно все исходные фреймы
    for i in tqdm(range(frames_count)):

        # загружаем i-й фрейм
        frame_path = os.path.join(PATH_TO_FRAMES, f'frame_{i}.jpg')
        person_int = read_image(Path(frame_path), mode=ImageReadMode.RGB)

        # трансформация и прогон через сеть
        person_float = transforms(person_int)
        outputs = model([person_float])

        # извлечем все keypoints
        kpts = outputs[0]['keypoints']
        # извлечем scores
        scores = outputs[0]['scores']

        # выделим реальные keypoints 2-х танцоров
        detect_threshold = 0.9 # порог вероятности
        idx = torch.where(scores > detect_threshold) # получаем индексы с вероятностью выше порога
        keypoints = kpts[idx] # выделяем ключевые точки обоих танцоров

        # выделим отдельно keypoints каждой девушки
        
        # определим правильно фигуры коуча и обучаемого (помним, что коуч слева от зрителя)
        if keypoints[0][0][0] < keypoints[1][0][0]: # если фигура левее
            idx_model = 0 # индекс model в представленных данных
            idx_input = 1 # индекс input в представленных данных
        else:
            idx_model = 1
            idx_input = 0

        # получаем отдельно ключевые точки для каждого танцора
        keypoints_input = keypoints[idx_input].unsqueeze(0) # коуч слева
        keypoints_model = keypoints[idx_model].unsqueeze(0) # обучаемый справа

        # наносим ключевые точки и скелет
        res_1 = draw_keypoints(person_int, keypoints_input, connectivity=connect_skeleton, colors="yellow", radius=6, width=5)
        res_2 = draw_keypoints(res_1, keypoints_model, connectivity=connect_skeleton, colors="red", radius=6, width=5)

        # получим score каждой точки входной позы в виде списка
        keypoints_scores_input = outputs[0]['keypoints_scores'][0].tolist()
        # получим keypoints модели и входа в виде массивов
        keypoints_model = get_keypoints(keypoints_model)
        keypoints_input = get_keypoints(keypoints_input)

        # Аффинное преобразование

        # С помощью расширенной матрицы можно осуществить умножение вектора x на матрицу A
        # и добавление вектора b за счёт единственного матричного умножения.
        # Расширенная матрица создаётся путём дополнения векторов "1" в конце.
        pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
        unpad = lambda x: x[:, :-1]

        # Расширим наборы ключевых точек до [[ x y 1] , [x y 1]]
        Y = pad(keypoints_model)
        X = pad(keypoints_input)

        # Решим задачу наименьших квадратов X * A = Y
        # и найдём матрицу аффинного преобразования A.
        A, res, rank, s = np.linalg.lstsq(X, Y, rcond=None) # добавлен rcond=None, чтобы не было предупреждений
        A[np.abs(A) < 1e-10] = 0  # превратим в "0" слишком маленькие значения

        # Теперь, когда мы нашли расширенную матрицу A,
        # мы можем преобразовать входной набор ключевых точек
        transform = lambda x: unpad(np.dot(pad(x), A))
        keypoints_input_transform = transform(keypoints_input)

        # вычисление косинусного расстояния и взвешенного совпадения

        # косинусное расстояние по оси 0
        cos_sim0 = torch.nn.CosineSimilarity(dim=0)
        output_0 = cos_sim0(torch.from_numpy(keypoints_model), torch.from_numpy(keypoints_input_transform))
        # косинусное расстояние по оси 1
        cos_sim1 = torch.nn.CosineSimilarity(dim=1)
        output_1 = cos_sim1(torch.from_numpy(keypoints_model), torch.from_numpy(keypoints_input_transform))
        # взвешенное совпадение
        weighted_dist = weight_distance(keypoints_input_transform, keypoints_model, keypoints_scores_input)

        # пробразование полученного тензора в итоговое изображение

        # tensor в массив numpy
        numpy_image = res_2.numpy()
        # меняем прядок осей
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        # конвертируем в rgb
        rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)

        # выводим метрики в виде текста

        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 0.75
        fontColor              = text_color
        thickness              = 2
        lineType               = 1

        # выведем среднее значение косинусного сходства по оси 0
        cv2.putText(rgb_image,f'CosSim_0_mean: {float(torch.mean(output_0)):.4f}',
            (10, numpy_image.shape[0] - 30),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

        # выведем среднее значение косинусного сходства по оси 1
        cv2.putText(rgb_image,f'CosSim_1_mean: {float(torch.mean(output_1)):.4f}',
            (10, numpy_image.shape[0] - 5),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

        # выведем взвешенное совпадение по оси 0
        cv2.putText(rgb_image,f'weighted_dist_0: {weighted_dist[0]:.4f}',
            (350, numpy_image.shape[0] - 30),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

        # выведем взвешенное совпадение по оси 1
        cv2.putText(rgb_image,f'weighted_dist_1: {weighted_dist[1]:.4f}',
            (350, numpy_image.shape[0] - 5),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType)

        # сохраняем итоговый фрейм
        cv2.imwrite(PATH_TO_OUTPUT_FRAMES + f'frame_{i}.jpg', rgb_image)

    # получаем разрешение фреймов
    width = rgb_image.shape[1]
    height = rgb_image.shape[0]

    return width, height