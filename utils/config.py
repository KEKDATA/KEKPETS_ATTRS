"""Данный модуль содержит в себе определения объектов с настройками проекта."""
import typing as tp

import yaml
from pydantic import BaseModel, create_model, validator


class PathConfig(BaseModel):
    """Хранит в себе настройки путей для всего проекта."""

    csv: str
    weights_folder: str


class ClearMLConfig(BaseModel):
    """Хранит в себе настройки логгирование в ClearML."""
    project_name: str
    experiment_number: int


class TrainingConfig(BaseModel):
    """Хранит в себе настройки для процедуры тренировки."""

    seed: int
    epochs: int
    device: str
    num_workers: int
    batch_size: int
    ema: float
    target_column_name: str


class ModelConfig(BaseModel):
    """Хранит в себе настройки нейронной сети."""

    num_classes: int
    name: str
    pretrained: bool
    freeze: bool
    file_name: tp.Optional[str]
    dropout_rate: tp.Optional[float]
    num_features: tp.Optional[int]


class Config(BaseModel):
    """Хранит в себе настройки всего проекта."""

    training: TrainingConfig
    path: PathConfig
    model: ModelConfig
    clearml: ClearMLConfig
    optimizer: tp.Any
    scheduler: tp.Any
    criterion: tp.Any
    augmentations: tp.Any

    @validator(
        'optimizer', 
        'scheduler', 
        'criterion', 
        'augmentations',
    )
    def build(
        cls,                    # noqa: N805
        model_parameters,
        values,                 # noqa: WPS110
        config,
        field
    ) -> BaseModel:
        """
        Построить pydantic-модели для конфигурации объектов.
        
        Параметры:
            model_parameters: Параметры для построения pydantic-модели;

        Returns:
            Pydantic-модель.
        """
        model_name = ''.join((field.name.capitalize(), 'Config'))
        return create_model(model_name, **model_parameters)()


def get_config(path_to_cfg: str) -> Config:
    """
    Распарсить .YANL файл с конфигурацией проекта и построить объект
    конфигурации.
    
    Параметры:
        path_to_cfg: Путь до .YAML файла.

    Returns:
        Объект с конфигурациями проекта.
    """
    with open(path_to_cfg, 'r') as yf:
        yml_file = yaml.safe_load(yf)
        config = Config.parse_obj(yml_file)
    return config
