"""Данный модуль содержит утилити-функции для работы с данными."""
import random
import typing as tp

import torch
import numpy as np
import pandas as pd
from albumentations import Compose
from torch.utils.data import DataLoader

from dataset import AnimalDataset
from utils.config import Config


def worker_init_fn(worker_id) -> None:
    """
    Зафиксировать зерно ГСЧ в текущем потоке.

    Параметры:
        worker_id: ID текущего потока в загрузчике данных.
    """
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def build_data(dataframe: pd.DataFrame) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Построить тренировочную и валидационную подвыборки.
    
    Параметры:
        dataframe: Исходный дата-фрейм с путями до изображений и метками.
        
    Возвращает:
        Датафейрмы для тренировочной и валидационной подвыборок.
    """
    train_df = dataframe.query('subset == "train"')
    val_df = dataframe.query('subset == "val"')
    return train_df, val_df


def build_dataloaders(
    dataframes: tp.Tuple[pd.DataFrame, pd.DataFrame],
    transforms: tp.Dict[str, Compose], 
    config: Config,
) -> tp.Tuple[DataLoader, DataLoader]:
    """
    Построить загрузчики данных для процедуры тренировки.

    Параметры:
        dataframes: Дата-фреймы для тренировочной и валидационной подвыборок;
        transforms: Трансформации для аугментации;
        config: Конфигурация проекта.

    Returns:
        Загрузчики данных для тренировочной и валидационной процедур.
    """
    train_df, val_df = dataframes
    train_dataset = AnimalDataset(
        train_df, config.training.target_column_name, transforms['train']
    )
    sampler = None
    shuffle = True
    train_dataloader = DataLoader(
        train_dataset, 
        config.training.batch_size,
        sampler=sampler,
        shuffle=shuffle, 
        pin_memory=True, 
        num_workers=config.training.num_workers,
        worker_init_fn=worker_init_fn
    )
    val_dataset = AnimalDataset(
        val_df, config.training.target_column_name, transforms['val']
    )
    val_dataloader = DataLoader(
        val_dataset, 
        2 * config.training.batch_size,
        pin_memory=True, 
        num_workers=config.training.num_workers,
    )
    return train_dataloader, val_dataloader
