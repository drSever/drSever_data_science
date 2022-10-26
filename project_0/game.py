""" This is proba50"""

import numpy as np

number = np.random.randint(1, 101) # Р·Р°РіР°РґС‹РІР°РµРј С‡РёСЃР»Рѕ
count = 0

while True:
    count += 1
    predict_number = int(input("РЈРіР°РґР°Р№ С‡РёСЃР»Рѕ РѕС‚ 1 РґРѕ 100"))

    if predict_number > number:
        print("Р§РёСЃР»Рѕ РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ РјРµРЅСЊС€Рµ!")

    elif predict_number < number:
        print("Р§РёСЃР»Рѕ РґРѕР»Р¶РЅРѕ Р±С‹С‚СЊ Р±РѕР»СЊС€Рµ!")

    else:
        print(f"Р’С‹ СѓРіР°РґР°Р»Рё С‡РёСЃР»Рѕ! Р­С‚Рѕ С‡РёСЃР»Рѕ = {number}, Р·Р° {count} РїРѕРїС‹С‚РѕРє")
        break # РєРѕРЅРµС† РёРіСЂС‹, РІС‹С…РѕРґ РёР· С†РёРєР»Р°