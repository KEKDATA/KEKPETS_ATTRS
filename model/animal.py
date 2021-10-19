"""
Данный модуль содержит определение модели нейронной сети, решающей задачи
классификации цвета шерсти собак и классификации длины хвоста собак.
"""
import typing as tp

import timm
import torch

from utils.config import Config


class AnimalModel(torch.nn.Module):
    """
    Нейронная сеть для классификации цвета окраса шерсти и длины хвоста у 
    собак.
    """
    
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
        
        # Добавляем классификаторную голову шерсти.
        if config.model.num_features is not None:
            if config.model.dropout_rate:
                dropout = torch.nn.Dropout(config.model.dropout_rate)
            else:
                dropout = torch.nn.Identity()
            self.head = torch.nn.Sequential(
                torch.nn.Linear(in_features, config.model.num_features),
                torch.nn.ReLU(inplace=True),
                dropout,
                torch.nn.Linear(
                    config.model.num_features, config.model.num_classes
                )
            )
        else:
            self.head = torch.nn.Linear(in_features, config.model.num_classes)

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход сети по входной картинке.
        
        Параметры:
            x: Преобразованная в тензор картинка.
            
        Возвращает:
            Сырые логиты для классификации.
        """
        features = self.backbone(x)
        return self.head(features)
