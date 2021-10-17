"""Данный модель содержит реализацию функционала качества модели."""
import typing as tp

import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


def validate(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    device: torch.device
) -> tp.Tuple[float, float]:
    """
    Произвести валидацию модели на валидационном множестве.

    Валидация проводится отдельно для задачи классификации цвета и отдельно для
    задачи классификации длины хвоста.

    Параметры:
        model: Нейронная сеть;
        dataloader: Загрузчик валидационных данных;
        device: Аппаратный ускоритель.

    Возвращает:
        Валидационные метрики модели.
    """
    model.eval()
    color_true = []
    color_pred = []
    tail_true = []
    tail_pred = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].float().to(device)
            colors = batch[1].long().to(device)
            tails = batch[2].float().to(device)
            color_outputs, tail_outputs = model(images)
            tail_true.extend(tails.cpu().byte().tolist())
            tail_predictions = (
                torch.sigmoid(tail_outputs) > 0.5
            ).byte().cpu().squeeze(dim=1).tolist()
            tail_pred.extend(tail_predictions)
            color_true.extend(colors.cpu().tolist())
            color_predictions = torch.argmax(
                color_outputs, dim=1
            ).cpu().tolist()
            color_pred.extend(color_predictions)
    f1_color = f1_score(color_true, color_pred, average='micro')
    f1_tail = f1_score(tail_true, tail_pred)
    return f1_color, f1_tail


def fitness_function(
    color_metric: float, tail_metric: float, alpha: float
) -> float:
    """
    Посчитать функцию подгонки как взвешенную сумму метрик по цвету и по длине
    хвоста.

    Параметры:
        color_metric: Метрика для задачи классификации цвета;
        tail_metric: Метрика для задачи классификации длины хвоста;
        alpha: Коэффициент смешивания.

    Возвращает:
        Единую метрику оценки "хорошести" обучения модели.
    """
    return alpha * color_metric + (1 - alpha) * tail_metric
