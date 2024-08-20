# импорт необходимых библиотек
import cv2
from tqdm import tqdm
import glob
import os


def frames_to_video(PATH_TO_OUTPUT_FRAMES, PATH_TO_OUTPUT_VIDEO, width, height):
    """
    Функция собирает итоговое видео из обработанных фреймов исходного видео.
    Сохраняет видеофайл.

    Args:
        PATH_TO_OUTPUT_FRAMES : путь к обработанным фреймам
        PATH_TO_OUTPUT_VIDEO : путь к итоговому видео
        width : ширина фрейма в пикселях
        height : высота фрейма в пикселях
    """

    # папка с фреймами
    image_folder = PATH_TO_OUTPUT_FRAMES + '*'
    # место сохранения итогового видео
    video_name = PATH_TO_OUTPUT_VIDEO + 'output.avi'

    # настройки
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video=cv2.VideoWriter(video_name,fourcc, 2.0, (width,height))

    # сборка итогового видео
    for i in tqdm((sorted(glob.glob(image_folder),key=os.path.getmtime))):
        x=cv2.imread(i)
        video.write(x)

    cv2.destroyAllWindows()
    video.release()