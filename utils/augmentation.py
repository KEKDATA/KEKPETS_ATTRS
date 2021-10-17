"""Данный модуль содержит функции построения трансформаций для аугментации."""
import typing as tp

import albumentations as album
from albumentations.core import serialization

from utils.config import Config


def get_augmentations(config: Config) -> tp.Dict[str, album.Compose]:
    """
    Построить конвеер из трансформаций для аугментации.

    Parameters:
        config: Конфигурация проекта.

    Returns:
        Конвееры транформаций для аугментации.
    """
    train_augs = serialization.from_dict(config.augmentations.train)
    validation_augs = serialization.from_dict(config.augmentations.val)
    return {'train': train_augs, 'val': validation_augs}
