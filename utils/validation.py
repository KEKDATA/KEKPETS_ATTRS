"""Данный модель содержит реализацию функционала качества модели."""
import typing as tp

import torch
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader


def validate(
    model: torch.nn.Module,
    criterion,
    dataloader: DataLoader, 
    device: torch.device
) -> float:
    """
    Произвести валидацию модели на валидационном множестве.

    Параметры:
        model: Нейронная сеть;
        dataloader: Загрузчик валидационных данных;
        device: Аппаратный ускоритель.

    Возвращает:
        Валидационные метрики модели.
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].float().to(device)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                labels = batch[1].float().unsqueeze(dim=1)
            else:
                labels = batch[1].tolist()
            outputs = model(images)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                outputs = torch.sigmoid(outputs).squeeze(dim=1).cpu().tolist()
            else:
                outputs = torch.argmax(outputs, dim=1).cpu()
            y_true.extend(labels)
            y_pred.extend(outputs)
    if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
        metric = roc_auc_score(y_true, y_pred)
    else:
        metric = f1_score(y_true, y_pred, average='macro')    
    return metric
