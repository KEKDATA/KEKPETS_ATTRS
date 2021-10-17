"""
Данный модуль содержит определение модели нейронной сети, решающей задачи
классификации цвета шерсти собак и классификации длины хвоста собак.
"""
import typing as tp

import timm
import torch

from utils.config import Config


class AnimalModel(torch.nn.Module):
    """Нейронная сеть для классификации цвета окраса шерсти и длины хвоста у 
    собак."""
    
    def __init__(self, config: Config) -> None:
        """
        Параметры:
            config: Конфигурация проекта.
        """
        super().__init__()
        
        # Создаем каркас сети из предобученной на ImageNet'e модели.
        self.backbone = timm.create_model(
            config.model.name,
            pretrained=config.model.pretrained
        )
        
        # Избавляемся от старой классификаторной головы.
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = torch.nn.Identity()

        # Замораживаем веса каркаса.
        if config.model.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Добавляем классификаторную голову на класс окраса шерсти.
        if config.model.color_num_features is not None:
            if config.model.color_dropout_rate:
                dropout = torch.nn.Dropout(config.model.color_dropout_rate)
            else:
                dropout = torch.nn.Identity()
            self.color_head = torch.nn.Sequential(
                torch.nn.Linear(in_features, config.model.color_num_features),
                torch.nn.ReLU(inplace=True),
                dropout,
                torch.nn.Linear(config.model.color_num_features, 4)
            )
        else:
            self.color_head = torch.nn.Linear(in_features, 4)

        # Добавляем классификаторную голову на длину хвоста.
        if config.model.tail_num_features is not None:
            if config.model.tail_dropout_rate:
                dropout = torch.nn.Dropout(config.model.tail_dropout_rate)
            else:
                dropout = torch.nn.Identity()
            self.tail_head = torch.nn.Sequential(
                torch.nn.Linear(in_features, config.model.tail_num_features),
                torch.nn.ReLU(inplace=True),
                dropout,
                torch.nn.Linear(config.model.tail_num_features, 1)
            )
        else:
            self.tail_head = torch.nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход сети по входной картинке.
        
        Параметры:
            x: Преобразованная в тензор картинка.
            
        Возвращает:
            Сырые логиты для классификации цвета и длины хвоста.
        """
        features = self.backbone(x)
        color = self.color_head(features)
        tail = self.tail_head(features)
        return color, tail
