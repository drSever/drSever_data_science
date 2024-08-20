# импорт необходимых библиотек
import os
from pathlib import Path
import cv2

def video_to_frames(PATH_TO_INPUT_VIDEO, PATH_TO_FRAMES):
    """
    Функция разбивает исходное видео на фреймы (кадры) и сохранет их.
    Возвращает количество полученных фреймов из исходного видео.

    Args:
        PATH_TO_INPUT_VIDEO : путь к исходному видео
        PATH_TO_FRAMES : путь полученным исходным фреймам

    Returns:
        frames_count : число полученных фреймов
    """
    
    # счетчик фреймов
    frames_count = 0

    # определяем видеофайл для иззвлечения кадров
    capture = cv2.VideoCapture(PATH_TO_INPUT_VIDEO)

    frameNr = 0 # номер текущего кадра (с нуля)

    # прогоняем видео, пока не закончатся кадры
    while (True):

        success, frame = capture.read() # получаем кадр

        # если кадр успешно получен
        if success:
            cv2.imwrite(PATH_TO_FRAMES + f'frame_{frameNr}.jpg', frame)

        else:
            break

        frameNr = frameNr+1
        frames_count += 1

    capture.release()

    return frames_count