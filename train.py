"""Данный модуль является точкой запуска процедуры тренировки сети."""
import os

import torch
import pandas as pd
from clearml import Task

from utils.general import seed_everything
from utils.config import get_config
from utils.training import train


if __name__ == '__main__':
    config = get_config('config.yml')

    # Инициализируем ClearML логгирование.
    task = Task.init(
        project_name=config.clearml.project_name,
        task_name='{project}-experiment {exp_num}'.format(
            project=config.clearml.project_name,
            exp_num=str(config.clearml.experiment_number)
        )
    )
    task.connect(config.dict())
    logger = task.get_logger()

    # Фиксируем зерна ГСЧ на всех девайсах и во всех используемых фреймворках
    # для воспроизводимости результатов тренировки.
    seed_everything(config)

    # Запускаем процесс тренировки.
    df = pd.read_csv(config.path.csv)
    training_result = train(df, config, logger)
    print(f'Best result: {training_result.val_score_value}.')

    # Сохраняем веса модели.
    if not os.path.exists(config.path.weights_folder):
        os.mkdir(config.path.weights_folder)
    weight_file_name = '_'.join(
        (f'exp{config.clearml.experiment_number}', config.model.file_name)
    )
    path_to_weights = os.path.join(
        config.path.weights_folder, weight_file_name
    )
    torch.save(training_result.weights, path_to_weights)
