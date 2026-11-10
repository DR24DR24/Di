import numpy as np
import torch
import os
import shutil

from projection_correlation import save_experiment_results

# Удаляем старую тестовую папку, если есть
if os.path.exists('results_test'):
    shutil.rmtree('results_test')

# Тест 1: numpy arrays 2D
print('=== Тест 1: numpy 2D arrays ===')
paths = save_experiment_results('test_exp', np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0], [6.0]]), 0.8421, output_dir='results_test')
print('Paths:', paths)
for key, p in paths.items():
    with open(p, 'r', encoding='utf-8') as f:
        print(f'--- {key}: {p} ---')
        print(f.read())

# Тест 2: torch tensors (включая 1D Y)
print('=== Тест 2: torch tensors ===')
paths2 = save_experiment_results('torch_test', torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([7.0, 8.0]), 0.5, output_dir='results_test')
print('Paths:', paths2)
for key, p in paths2.items():
    with open(p, 'r', encoding='utf-8') as f:
        print(f'--- {key}: {p} ---')
        print(f.read())

# Тест 3: 1D arrays
print('=== Тест 3: 1D arrays ===')
paths3 = save_experiment_results('oned_test', np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), 0.99, output_dir='results_test')
print('Paths:', paths3)
for key, p in paths3.items():
    with open(p, 'r', encoding='utf-8') as f:
        print(f'--- {key}: {p} ---')
        print(f.read())

# Тест 4: кастомный суффикс для PC
print('=== Тест 4: кастомный суффикс PC ===')
paths4 = save_experiment_results('suffix_test', np.array([[1.0]]), np.array([[2.0]]), 0.123, output_dir='results_test', pc_filename_suffix='proj_corr')
print('Paths:', paths4)

print('\nВсе тесты пройдены!')

# Очистка
shutil.rmtree('results_test')
print('Тестовая папка удалена.')