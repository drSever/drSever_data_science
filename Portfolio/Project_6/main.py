# импорт необходимых библиотек

import os

# импорт из модулей пакета проекта
from myproject.model import get_model
from myproject.video_preprocessing import video_to_frames
from myproject.frames_preprocessing import frames_preprocessing
from myproject.video_output import frames_to_video

# задаем параметры (пути к файлам, цвет текста метрик на изображении)
# ! ПАРАМЕТРЫ МОЖНО ИЗМЕНЯТЬ !

if not os.path.exists('output_data'):
    os.makedirs('output_data')

# путь к исходному видеофайлу
PATH_TO_INPUT_VIDEO = "./video.mp4"
# путь к итоговому видеофайлу
PATH_TO_OUTPUT_VIDEO = "./output_data/"
# путь к исходным фреймам
PATH_TO_FRAMES = "./output_data/frames/"
# путь к обработанным фреймам
PATH_TO_OUTPUT_FRAMES = "./output_data/frames_output/"

# цвет текста метрик на итоговом видео
#text_color = (255,255,255) # белый 
text_color = (0,0,0) # черный

if not os.path.exists(PATH_TO_FRAMES):
    os.makedirs(PATH_TO_FRAMES)
if not os.path.exists(PATH_TO_OUTPUT_FRAMES):
    os.makedirs(PATH_TO_OUTPUT_FRAMES)

# запуск кода приложения
def main():
    
    print(f'\n*** Приложаение запущено ***')

    # получаем модель и трансформации для исходных фреймов (кадров)
    model, transforms  = get_model()
    print('\nМодель получена')

    # преобразуем исходное видео в набор фреймов, получаем кол-во фреймов, указываем путь к фреймам
    print(f'\nПреобразование исходного видео в фреймы...')
    frames_count = video_to_frames(PATH_TO_INPUT_VIDEO, PATH_TO_FRAMES)
    print(f'Из исходного видео {frames_count} фреймов сохранено в ' + PATH_TO_FRAMES)  

    # выполняем обработку исходных фреймов, сохраняем их, указываем путь сохранения и получаем разрешения фреймов (ширина, высота)
    print(f'\nОбработка исходных фреймов...')
    width, height = frames_preprocessing(PATH_TO_FRAMES, PATH_TO_OUTPUT_FRAMES, model, transforms, text_color, frames_count)
    print('Обработанные фреймы сохраненв в ' + PATH_TO_OUTPUT_FRAMES)

    # собираем итоговый видеофайл из обработанных фреймов, сохраняем и указываем путь к нему
    print(f'\nСобираем итоговый видеофайл...')
    frames_to_video(PATH_TO_OUTPUT_FRAMES, PATH_TO_OUTPUT_VIDEO, width, height)
    print('Итоговый видеофайл сохранен в ' + PATH_TO_OUTPUT_VIDEO)

    print(f'\n*** Приложение остановлено ***\n')

if __name__=='__main__':
    main()