"""
Данный модуль содержит в себе реализацию алгоритма экспоненциально скользящего
среднего для весов модели.
"""
import typing as tp
from copy import deepcopy

import torch


class EMA(torch.nn.Module):
    """Э
    кспоненциально скользящее среднего для всего, что есть в словаре 
    состояния модели.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.9999,
        device: tp.Optional[torch.device] = None
    ) -> None:
        """
        Параметры:
            model: Начальное состояние нейронной сети;
            decay: Коэффициент, определяющий силу скольжения;
            device: Аппаратный ускоритель, на котором будет хранится результат.
        """
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(self.device)

    def _update(self, model: torch.nn.Module, update_fn: tp.Callable) -> None:
        """
        Произвести скользящее среднее по весам модели.

        Параметры:
            model: Новое состояние модели;
            update_fn: Функция, реализующая логику обновления.
        """
        with torch.no_grad():
            _iterator = zip(
                self.module.state_dict().values(), model.state_dict().values()
                )
            for ema_v, model_v in _iterator:
                if self.device is not None:
                    model_v = model_v.to(self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model: torch.nn.Module) -> None:
        """
        Произвести обновление весов модели.

        Параметры:
            model: Новое состояние нейронной сети.
        """
        self._update(
            model, 
            update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m
        )
