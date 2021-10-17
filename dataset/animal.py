"""Модуль содержит определение класса для работы с данными."""
import typing as tp

import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as album
from torch.utils.data import Dataset


# Тип точки данных, возвращаемый набором данных.
DataPoint = tp.Tuple[tp.Union[torch.Tensor, np.ndarray], int, int]


class AnimalDataset(Dataset):
    """Имплементирует логику подготовки данных для подачи в нейронную сеть."""
    
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        transform: tp.Optional[album.Compose] = None,
    ) -> None:
        """
        Параметры:
            dataframe: Исходный дата-фрейм с метками и путями до картинок;
            transform: Трансформации для аугментации изображений.
        """
        self.data = dataframe.copy()
        self.transform = transform
        
    def __len__(self) -> int:
        """Посчитать длину набора данных."""
        return len(self.data)
        
    def __getitem__(self, index: int) -> DataPoint:
        """
        Взять одну точку данных (изображение, класс цвета, класс длины хвоста).
        
        Параметры:
            index: Индекс точки данных.
            
        Возвращает:
            Точку данных: (изображение, класс цвета, класс длины хвоста).
        """
        data_point = self.data.iloc[index]
        image = cv2.imread(data_point['image'])
        color = data_point['color']
        tail_class = data_point['long_tail']
        if self.transform:
            image = self.transform(image=image)['image']
        return image, color, tail_class
