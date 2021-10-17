"""Данный модуль содержит общие вспомогательные функции."""
import os
import pydoc
import random
import typing as tp

import numpy as np
import torch
from pydantic import BaseModel

from utils.config import Config


def object_from_pydantic(
    pydantic_model: BaseModel,
    parent: tp.Optional[BaseModel] = None,
    **additional_kwargs: tp.Dict[str, tp.Union[float, str, int]],
) -> tp.Any:
    """
    Распарсить pydantic-модель и построить инстансы указанных типов.

    Параметры:
        pydantic_model: Pydantic-модель;
        parent: Родительская модель;
        additional_kwargs: Дополнительные параметры для процедуры 
            инстансцирования.

    Возвращает:
        Инстанс указанного типа.
    """
    kwargs = pydantic_model.dict().copy()
    object_type = kwargs.pop('algo')
    for param_name, param_value in additional_kwargs.items():
        kwargs.setdefault(param_name, param_value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)

    return pydoc.locate(object_type)(**kwargs)


def seed_everything(config: Config) -> None:
    """
    Зафиксировать зерно ГСЧ во всех фреймворках 
    и на всех аппаратных ускорителях.
    
    Параметры:
        config: Конфигурация проекта.
    """
    random.seed(config.training.seed)
    np.random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    os.environ['PYTHONHASHSEED'] = str(config.training.seed)
    torch.cuda.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_cpu_state_dict(model: torch.nn.Module) -> tp.Dict:
    """
    Получить веса модели на ЦПУ.

    Параметры:
        model: Модель нейронной сети..

    Возвращает:
        Словарь состояний модели на ЦПУ.
    """
    return {k: v.cpu() for k, v in model.state_dict().items()}
