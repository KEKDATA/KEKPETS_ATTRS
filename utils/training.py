"""
Данный модуль содержит в себе необходимые функции для процедуры 
тренировки.
"""
import typing as tp

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from clearml import Logger
from pydantic import BaseModel
from torch.utils.data import DataLoader

from utils.validation import validate, fitness_function
from utils.ema import EMA
from model import AnimalModel
from utils.augmentation import get_augmentations
from utils.config import Config
from utils.general import get_cpu_state_dict, object_from_pydantic
from utils.data import build_dataloaders, build_data


class TrainingResult(BaseModel):
    """Содержит в себе результаты процедуры тренировки."""

    val_score_value: float
    weights: tp.Optional[tp.Dict]

    class Config:
        arbitrary_types_allowed = True


def add_weight_decay(
    model: torch.nn.Module, weight_decay: float,
) -> tp.List[tp.Dict[str, float]]:
    """
    Добавить L2-регуляризации ко всем параметрам модели, кроме 
    параметров смещения.
    
    Параметры:
        model: Нейронная сеть;
        weight_decay: Коэффициент L2-регуляризации.

    Возвращает:
        Список групп параметров модели с установленным значением параметра L2-
        регуляризации.
    """
    to_decay = []
    not_to_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias'):
            not_to_decay.append(param) 
        else:
            to_decay.append(param)
    return [
        {'params': not_to_decay, 'weight_decay': 0.},
        {'params': to_decay, 'weight_decay': weight_decay}
    ]


def create_model(config: Config) -> AnimalModel:
    """
    Создать объект модели нейронной сети.

    Параметры:
        config: Конфигурация проекта.

    Возвращает:
        Объект модели нейронной сети.
    """
    model = AnimalModel(config).float().to(config.training.device)
    return model


def choose_ema_or_normal(
    normal_val_score: float,
    normal_model: AnimalModel, 
    ema_val_score: tp.Optional[float],
    ema_model: tp.Optional[EMA]
):
    """
    Выбрать лучшую модель между EMA-моделью и обычной моделью.

    Параметры:
        normal_val_score: Метрика обычной модели;
        normal_model: Обычная модель;
        ema_val_score: Метрика EMA-модели;
        ema_model: EMA-модель.
    
    Возвращает:
        Наибольшую метрику между сравниваемыми моделями и соответствующий
        словарь состояний модели.
    """
    if ema_val_score is None or normal_val_score > ema_val_score:
        return normal_val_score, get_cpu_state_dict(normal_model)
    return ema_val_score, get_cpu_state_dict(ema_model.module)


def train_one_epoch(
    model: torch.nn.Module,
    criterion: tp.Tuple[torch.nn.CrossEntropyLoss, torch.nn.BCEWithLogitsLoss],
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    model_ema: tp.Optional[EMA] = None,
    scheduler: tp.Optional[torch.optim.lr_scheduler.StepLR] = None,
) -> tp.Tuple[float, float]:
    """
    Провести тренировочную процедуру на одной эпохе.

    Параметры:
        model: Нейронная сеть;
        criterion: Функции потерь для задач классификации;
        optimizer: Алгоритм оптимизации;
        dataloader: Загрузчик тренировочных данных;
        device: Аппаратный ускоритель;
        model_ema: EMA-модель;
        scheduler: Планировщик изменения шага обучения;
        logger: ClearML-логгер.

    Возвращает:
        Значения функций потерь для текущий эпохи.
    """
    color_criterion, tail_criterion = criterion
    model.train()
    epoch_color_loss = 0.
    epoch_tail_loss = 0.
    for batch in dataloader:
        images = batch[0].float().to(device)
        colors = batch[1].long().to(device)
        tails = batch[2].unsqueeze(dim=1).float().to(device)
        optimizer.zero_grad()
        color_outputs, tail_outputs = model(images) 
        color_loss = color_criterion(color_outputs, colors)
        tail_loss = tail_criterion(tail_outputs, tails)

        # Необходимо восстанавливать граф вычислений, чтобы обратное 
        # распространение ошибки сработало во второй раз.
        color_loss.backward(retain_graph=True)
        tail_loss.backward(retain_graph=True)

        optimizer.step()
        epoch_color_loss += color_loss.item()
        epoch_tail_loss += tail_loss.item()
        if model_ema is not None:
            model_ema.update(model)
        if scheduler is not None:
            scheduler.step()
    return epoch_color_loss / len(dataloader), epoch_tail_loss / len(dataloader)


def train(df: pd.DataFrame, config: Config, logger: Logger):
    """
    Провести цикл тренировки.

    Parameters:
        df: Исходный дата-фрейм с путями до изображений и метками;
        config: Конфигурация проекта;
        logger: ClearML логгер.

    Возвращает:
        Результаты процедуры тренировки.
    """

    # Строим данные и их загрузчики.
    train_df, val_df = build_data(df)
    train_dataloader, val_dataloader = build_dataloaders(
        (train_df, val_df), get_augmentations(config), config
    )

    # Строим модель.
    model = create_model(config)
    model_ema = None
    if config.training.ema != 0:
        model_ema = EMA(
            model,  device=config.training.device,  decay=config.training.ema
        )

    # Создаем объекты функций потерь, алгоритма оптимизации, планировщика и т.д.
    optimizable_params = add_weight_decay(model, config.optimizer.weight_decay)
    config.optimizer.weight_decay = 0.
    optimizer = object_from_pydantic(
        config.optimizer, params=optimizable_params
    )
    color_criterion = object_from_pydantic(config.color_criterion)
    tail_criterion = object_from_pydantic(config.tail_criterion)
    scheduler = object_from_pydantic(config.scheduler, optimizer=optimizer)
    
    training_result = TrainingResult(val_score_value=0, weights=None)
    pbar = tqdm(
        range(config.training.epochs), 
        desc='Fold #1 training procedure'
    )
    for epoch in pbar:
        np.random.seed(config.training.seed + epoch)
        train_color_loss, train_tail_loss = train_one_epoch(
            model, 
            (color_criterion, tail_criterion), 
            optimizer, 
            train_dataloader, 
            config.training.device,
            model_ema,
            None,
        )
        if scheduler is not None:
            logger.report_scalar('LR', 'LR', scheduler.get_last_lr()[0], epoch)
            scheduler.step()
        compose_loss = fitness_function(
            train_color_loss, 
            train_tail_loss, 
            config.training.compose_loss_coeff
        )
        logger.report_scalar(
            'Color losses', 'Color train loss', train_color_loss, epoch
        )
        logger.report_scalar(
            'Tail losses', 'Tail train loss', train_tail_loss, epoch
        )
        logger.report_scalar(
            'Compose Losses', 
            'Compose loss', 
            compose_loss,
            epoch
        )
        normal_color, normal_tail = validate(
            model, val_dataloader, config.training.device
        )
        normal_compose = fitness_function(
            normal_color, 
            normal_tail, 
            config.training.compose_loss_coeff
        )
        logger.report_scalar(
            'Color normal metric', 'F1', normal_color, epoch
        )
        logger.report_scalar(
            'Tail normal metric', 'F1', normal_tail, epoch
        )
        logger.report_scalar(
            'Normal compose metric', 'Compose F1', normal_compose, epoch
        )
        ema_compose = None
        if model_ema is not None:
            ema_color, ema_tail = validate(
                model_ema.module, val_dataloader, config.training.device
            )
            ema_compose = fitness_function(
                ema_color, 
                ema_tail, 
                config.training.compose_loss_coeff
            )
            logger.report_scalar(
                'EMA color metric', 'F1', ema_color, epoch
            )
            logger.report_scalar(
                'EMA tail metric', 'F1', normal_tail, epoch
            )
            logger.report_scalar(
                'EMA compose metric', 'Compose F1', ema_compose, epoch
            )
        val_score, weights = choose_ema_or_normal(
            normal_compose, model, ema_compose, model_ema
        )
        pbar.set_postfix_str(s=f'Validation score: {val_score: .2f}')
        if val_score >= training_result.val_score_value:
            training_result.val_score_value = val_score
            training_result.weights = weights
    return training_result
