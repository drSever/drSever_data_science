"""Игра угадай число
Компьютер сам загадывает и сам угадывает число
"""

import numpy as np

def random_predict(number:int=1) -> int:
    """Рандомно угадываем число

    Args:
        number (int, optional): Загаданное число. Defaults to 1.

    Returns:
        int: Число попыток
    """
    
    number = np.random.randint(1,101) # компьютер загадал число
    count = 0 # счетчик попыток изначально равен 0
    min_predict_number = 1 # начало интервала для угадывания числа
    max_predict_number = 101 # конец интервала для угадывания числа
    
    while True:
        """
        С каждой итерацией сужаем интервал угадывания числа 
        """
        count += 1
        predict_number = np.random.randint(min_predict_number,max_predict_number) # предполагаемое число
        
        if number == predict_number:
            break # выход из цикла, если угадали
        
        elif number > predict_number:
            min_predict_number = predict_number
                            
        elif number < predict_number:
            max_predict_number = predict_number
                               
    return(count)

print(f'Количество попыток: {random_predict()}')

def score_game(random_predict) -> int:
    """За какое количество попыток в среднем из 1000 подходов угадывает наш алгоритм

    Args:
        random_predict ([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """

    count_ls = [] # список для сохранения количества попыток
    np.random.seed(1) # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(1000)) # загадали список чисел

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls)) # находим среднее количество попыток

    print(f'Ваш алгоритм угадывает число в среднем за: {score} попыток')
    return(score)

# RUN
if __name__ == '__main__':
    score_game(random_predict)