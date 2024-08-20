
# импорт необходимых библиотек
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

# получаем модель и трансформации для обработки фреймов исходного видео
def get_model():
    """
    Функция получает веса предобученной модели KeypointRCNN_ResNet50_FPN,
    возвращает саму модель и трансформации для входящих фреймов.
    """

    # используем веса предобученной keypointrcnn_resnet50_fpn сети
    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    # получаем трансформации для входящих фреймов
    transforms = weights.transforms()

    # используем предобученную keypointrcnn_resnet50_fpn сеть
    model = keypointrcnn_resnet50_fpn(weights=weights, progress=False)

    return model, transforms
