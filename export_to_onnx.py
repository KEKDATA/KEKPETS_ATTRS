"""
Данный модуль содержит в себе процедуры перевода модели из фреймворка 
PyTorch в фреймворк ONNX.
"""
import os
import argparse as ap

import onnx
import torch
import onnxsim
import numpy as np
import onnxoptimizer

from model import AnimalModel
from utils.config import get_config


if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--input_model_path', 
        '-i', 
        required=True, 
        type=str, 
        help='Path to PyTorch checkpoint.'
    )
    parser.add_argument(
        '--output_model_path',
        '-o',
        required=True,
        type=str,
        help='Path to output .ONNX file.'
    )
    parser.add_argument(
        '--input_image_size',
        '-s',
        required=False,
        default=(224, 224),
        help='Input image size for onnx model.'
    )
    args = parser.parse_args()
    try:
        height, width = map(int, args.input_image_size.split(','))
    except AttributeError:
        height, width = args.input_image_size
    
    # Создаем объект модели и загружаем в него натренированные веса.
    config = get_config('config.yml')
    config.model.pretrained = False
    model = AnimalModel(config).float().cpu()
    model.load_state_dict(torch.load(args.input_model_path))

    # Переключаем модель в режим инференса.
    model.eval()

    # Экспортируем модель в ONNX.
    base_path, onnx_name = os.path.split(args.output_model_path)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    x = torch.randn(1, 3, height, width, requires_grad=True)
    torch_out = model(x)
    torch.onnx.export(
        model,
        x,
        args.output_model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    onnx_model = onnx.load(args.output_model_path)
    onnx.checker.check_model(onnx_model)
    print('Exporting successfully done.')
    print('Optimizing model . . .')
    model_simp, check = onnxsim.simplify(onnx_model)
    optimized_model = onnxoptimizer.optimize(onnx_model)
    if not check:
        print('Can not simplify.')
    else:
        print('Simplification was successfully done.')
        onnx.save(optimized_model, args.output_model_path)

