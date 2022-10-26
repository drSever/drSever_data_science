import numpy as np

def random_predict(number:int=1) -> int:
    """Р Р°РЅРґРѕРјРЅРѕ СѓРіР°РґС‹РІР°РµРј С‡РёСЃР»Рѕ

    Args:
        number (int, optional): Р—Р°РіР°РґР°РЅРЅРѕРµ С‡РёСЃР»Рѕ. Defaults to 1.

    Returns:
        int: Р§РёСЃР»Рѕ РїРѕРїС‹С‚РѕРє
    """

    count = 0

    while True:
        count += 1
        predict_number = np.random.randint(1, 101) # РїСЂРµРґРїРѕР»Р°РіР°РµРјРѕРµ С‡РёСЃР»Рѕ
        if number == predict_number:
            break # РІС‹С…РѕРґ РёР· С†РёРєР»Р°, РµСЃР»Рё СѓРіР°РґР°Р»Рё
    return(count)

print(f'РљРѕР»РёС‡РµСЃС‚РІРѕ РїРѕРїС‹С‚РѕРє: {random_predict()}')

def score_game(random_predict) -> int:
    """Р—Р° РєР°РєРѕРµ РєРѕР»РёС‡РµСЃС‚РІРѕ РїРѕРїС‹С‚РѕРє РІ СЃСЂРµРґРЅРµРј РёР· 1000 РїРѕРґС…РѕРґРѕРІ СѓРіР°РґС‹РІР°РµС‚ РЅР°С€ Р°Р»РіРѕСЂРёС‚Рј

    Args:
        random_predict ([type]): С„СѓРЅРєС†РёСЏ СѓРіР°РґС‹РІР°РЅРёСЏ

    Returns:
        int: СЃСЂРµРґРЅРµРµ РєРѕР»РёС‡РµСЃС‚РІРѕ РїРѕРїС‹С‚РѕРє
    """

    count_ls = [] # СЃРїРёСЃРѕРє РґР»СЏ СЃРѕС…СЂР°РЅРµРЅРёСЏ РєРѕР»РёС‡РµСЃС‚РІР° РїРѕРїС‹С‚РѕРє
    np.random.seed(1) # С„РёРєСЃРёСЂСѓРµРј СЃРёРґ РґР»СЏ РІРѕСЃРїСЂРѕРёР·РІРѕРґРёРјРѕСЃС‚Рё
    random_array = np.random.randint(1, 101, size=(1000)) # Р·Р°РіР°РґР°Р»Рё СЃРїРёСЃРѕРє С‡РёСЃРµР»

    for number in random_array:
        count_ls.append(random_predict(number))

    score = int(np.mean(count_ls)) # РЅР°С…РѕРґРёРј СЃСЂРµРґРЅРµРµ РєРѕР»РёС‡РµСЃС‚РІРѕ РїРѕРїС‹С‚РѕРє

    print(f'Р’Р°С€ Р°Р»РіРѕСЂРёС‚Рј СѓРіР°РґС‹РІР°РµС‚ С‡РёСЃР»Рѕ РІ СЃСЂРµРґРЅРµРј Р·Р°: {score} РїРѕРїС‹С‚РѕРє')
    return(score)

# RUN
if __name__ == '__main__':
    score_game(random_predict)