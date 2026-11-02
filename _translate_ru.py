import os
d = os.path.dirname(os.path.abspath(__file__))

# Read build_notebooks.py
path1 = os.path.join(d, 'build_notebooks.py')
with open(path1, encoding='utf-8') as f:
    content = f.read()

# All replacements - only Russian text fragments, never touching backslash sequences
repls = []

# Header
repls.append(('Демонстрация алгоритма Di et al. (2025) в Google Colab', 'Demonstration of the Di et al. (2025) Algorithm in Google Colab'))
repls.append(('Авторы статьи:', 'Article Authors:'))
repls.append(('Цель ноутбука:', 'Notebook Objective:'))
repls.append(('Показать феномен **фазового перехода (Phase Transition)** между универсальными (general) и специфическими (specific) слоями глубокой нейросети.', 'Demonstrate the **Phase Transition** phenomenon between universal (general) and specific layers of a deep neural network.'))
repls.append(('Продемонстрировать вычисление меры общности', 'Compute the generality measure'))
repls.append(('с помощью **проекционной корреляции (Projection Correlation)**', 'using **Projection Correlation**'))
repls.append(('без переобучения признаков (за 1 прямой проход).', 'without feature overfitting (in a single forward pass).'))
repls.append(('Оценить теоретическую точку среза:', 'Estimate the theoretical cut-point:'))
repls.append(('Запустить **Алгоритм 1 (Forward-Adding Layer Selection)** с переносом весов (warm-start) и сравнить его с полным перебором моделей', 'Run **Algorithm 1 (Forward-Adding Layer Selection)** with weight transfer (warm-start) and compare it with exhaustive model search'))

# Step 1
repls.append(('Шаг 1. Настройка окружения Google Colab и проверка GPU', 'Step 1. Google Colab Environment Setup and GPU Check'))
repls.append(('Проверка окружения Google Colab', 'Check Google Colab Environment'))
repls.append(('Обнаружена среда Google Colab', 'Google Colab environment detected'))
repls.append(('Список возможных путей к репозиторию на Google Диске', 'List of possible paths to the repository on Google Disk'))
repls.append(('Автоматический поиск папки по характерным файлам проекта', 'Automatic search for the folder by project characteristic files'))
repls.append(('Поиск папки проекта в MyDrive', 'Searching for project folder in MyDrive'))
repls.append(('Рабочая директория установлена:', 'Working directory set:'))
repls.append(('Внимание: папка проекта не найдена в MyDrive, используется текущая директория.', 'Warning: project folder not found in MyDrive, using current directory.'))
repls.append(('Локальный запуск. Директория:', 'Local execution. Directory:'))
repls.append(('Проверка доступности GPU (CUDA)', 'GPU (CUDA) Availability Check'))
repls.append(('Устройство для вычислений:', 'Computing device:'))
repls.append(('Название GPU:', 'GPU name:'))
repls.append(('Объем памяти:', 'Memory size:'))

# Step 2
repls.append(('Шаг 2. Импорт модулей проекта и проверка свойств проекционной корреляции (PC)', 'Step 2. Import Project Modules and Projection Correlation (PC) Properties Check'))
repls.append(('Проекционная корреляция', 'Projection Correlation'))
repls.append(('обладает ключевым свойством:', 'has a key property:'))
repls.append(('Она способна обнаруживать любые нелинейные зависимости (в отличие от линейной корреляции Пирсона', "It can detect any nonlinear dependencies (unlike Pearson's linear correlation)"))
repls.append(('не видит зависимость)', 'does not see the dependency)'))
repls.append(('четко детектирует зависимость)', 'clearly detects the dependency)'))
repls.append(('Проверка детекции нелинейной зависимости (окружность X^2 + Y^2 = 1)', 'Nonlinear Dependency Detection Check (circle X^2 + Y^2 = 1)'))
repls.append(('Нелинейная зависимость (окружность):', 'Nonlinear dependency (circle):'))
repls.append(('Линейный коэффициент Пирсона:', 'Pearson coefficient:'))

# Step 3
repls.append(('Шаг 3. Синтетический бенчмарк переноса и 6-слойная CNN (Figure 3)', 'Step 3. Synthetic Transfer Benchmark and 6-Layer CNN (Figure 3)'))
repls.append(('Создаем синтетическую задачу с фазовым переходом:', 'Create a synthetic task with a phase transition:'))
repls.append(('Исходная задача (Source)**', 'Source Task**'))
repls.append(('объектов, 5 классов.', 'samples, 5 classes.'))
repls.append(('Целевая задача (Target)**', 'Target Task**'))
repls.append(('объектов, сдвиг домена на глубоких слоях', 'samples, domain shift at deep layers'))
repls.append(('Архитектура**:', 'Architecture:**'))
repls.append(('сверточных блоков (Di et al., Figure 3)', 'convolutional blocks (Di et al., Figure 3)'))
repls.append(('Генерация выборок с фазовым переходом на 4-м слое', 'Generation of samples with phase transition at layer 4'))
repls.append(('Построение 6-слойной CNN', 'Building a 6-layer CNN'))
repls.append(('Архитектура модели:', 'Model architecture:'))

# Step 4
repls.append(('Шаг 4. Предобучение модели на исходной задаче (Source Pretraining)', 'Step 4. Model Pretraining on Source Task (Source Pretraining)'))
repls.append(('Предобучение на Source выборке (5 эпох)...', 'Pretraining on Source set (5 epochs)...'))

# Step 5
repls.append(('Шаг 5. Профилирование общности слоев через Проекционную Корреляцию (PC Profile)', 'Step 5. Profiling Layer Generality via Projection Correlation (PC Profile)'))
repls.append(('Вычисляем меру общности:', 'Computing the generality measure:'))
repls.append(('для каждого слоя', 'for each layer'))
repls.append((' Вычисление проводится **за один прямой проход (forward pass)** без градиентного спуска!', ' Computed **in a single forward pass** without gradient descent!'))
repls.append(('ТЕОРЕТИЧЕСКАЯ ТОЧКА ФАЗОВОГО ПЕРЕХОДА (Теорема 2):', 'THEORETICAL PHASE TRANSITION POINT (Theorem 2):'))
repls.append(('Срез j*', 'Cut point j*'))
repls.append(('(слой:', '(layer:'))
repls.append(('Универсальные слои (переносим):', 'General layers (to transfer):'))
repls.append(('Специфические слои (отбрасываем):', 'Specific layers (to discard):'))

# Step 6
repls.append(('Шаг 6. Визуализация фазового перехода (Phase Transition Plot)', 'Step 6. Phase Transition Visualization (Phase Transition Plot)'))
repls.append(('Выделение точки фазового перехода', 'Highlighting the phase transition point'))
repls.append(('Теоретический срез', 'Theoretical cut-point'))
repls.append(('Заливка областей', 'Shading areas'))
repls.append(('General Layers (универсальные)', 'General Layers'))
repls.append(('Specific Layers (переобучение)', 'Specific Layers (overfitting)'))
repls.append(('Сверточные слои сети', 'Network Convolutional Layers'))
repls.append(('Мера общности', 'Generality Measure'))
repls.append(('Профиль общности слоев и точка фазового перехода (Di et al., 2025)', 'Layer Generality Profile and Phase Transition Point (Di et al., 2025)'))

# Step 7
repls.append(('Шаг 7. Запуск Алгоритма 1: Forward-Adding Layer Selection с Warm-Start', 'Step 7. Running Algorithm 1: Forward-Adding Layer Selection with Warm-Start'))

# Step 8
repls.append(('Шаг 8. Сравнение с полным перебором (Exhaustive Search', 'Step 8. Comparison with Exhaustive Search'))
repls.append(('Время работы Алгоритма 1:', 'Algorithm 1 runtime:'))
repls.append(('Обучение вложенных моделей M_0 ... M_6 полным перебором...', 'Training nested models M_0 ... M_6 exhaustively...'))
repls.append(('перенесено {k} слоев', '{k} transferred layers'))
repls.append(('Сводная таблица', 'Summary table'))
repls.append(('\'Модель\':<10', '\'Model\':<10'))
repls.append(('\'Перенесено слоев\':<18', '\'Transferred Layers\':<18'))
repls.append(('<<< ЛУЧШАЯ (min loss)', '<<< BEST (min loss)'))
repls.append(('Выбор Алгоритма 1:', 'Algorithm 1 choice:'))
repls.append(('Выбор полного перебора:', 'Exhaustive search choice:'))
repls.append(('Время:', 'Time:'))
repls.append(('Ускорение (Speedup):', 'Speedup:'))

# Step 9
repls.append(('Шаг 9. График валидационной ошибки по числу перенесенных слоев', 'Step 9. Validation Error vs. Number of Transferred Layers'))
repls.append(('Число перенесенных слоев $ (Модель $)', 'Number of Transferred Layers $ (Model $)'))
repls.append(('Число перенесенных слоев $', 'Number of Transferred Layers $'))
repls.append(('Отметка фазового перехода', 'Phase transition marker'))
repls.append(('Валидационная ошибка и точность вложенных моделей', 'Validation Error and Accuracy of Nested Models'))
repls.append(('## Заключение', '## Conclusion'))
repls.append(('Теорема 2 доказана на практике:', 'Theorem 2 is validated in practice:'))
repls.append(('**Алгоритм 1 с warm-start** оценивает значительно меньше моделей и останавливается сразу после точки насыщения, экономя вычислительные ресурсы в разы.', '**Algorithm 1 with warm-start** evaluates significantly fewer models and stops immediately after the saturation point, saving computational resources by orders of magnitude.'))

# CIFAR10 notebook header
repls.append(('Репликация Эксперимента CIFAR-10 из статьи Di et al. (2025)', 'CIFAR-10 Experiment Replication from Di et al. (2025)'))
repls.append(('Раздел статьи:', 'Paper section:'))
repls.append(('Воспроизводимые эффекты:', 'Reproducible effects:'))
repls.append(('Два режима переноса:', 'Two transfer modes:'))
repls.append(('Непересекающиеся классы (Источник = 5 классов животных, Цель = 5 классов транспорта).', 'Non-overlapping classes (Source = 5 animal classes, Target = 5 vehicle classes).'))
repls.append(('Разделение 10 классов (50% объектов источник, 50% цель).', 'Split of 10 classes (50% source objects, 50% target).'))
repls.append(('Воспроизведение графиков Figure 4 (A, B)', 'Reproduction of Figure 4 (A, B)'))
repls.append((' Профиль меры общности', ' Generality measure profile'))
repls.append(('Воспроизведение графиков Figure 4 (C, D)', 'Reproduction of Figure 4 (C, D)'))
repls.append(('Сравнение вычислительной сложности:', 'Computational complexity comparison:'))
repls.append(('Оценка ускорения Алгоритма 1 по сравнению с классическим перебором', 'Algorithm 1 speedup estimate compared with classical exhaustive search'))

# CIFAR10 step 1
repls.append(('## 1. Настройка окружения Google Colab и подключение GPU', '## 1. Google Colab Environment Setup and GPU Connection'))
repls.append(('# Подключение Google Диска при работе в Colab', '# Mounting Google Drive when running in Colab'))
repls.append(('[Colab] Рабочая директория:', '[Colab] Working directory:'))
repls.append(('[Local] Директория:', '[Local] Directory:'))
repls.append(('# Проверка GPU', '# GPU Check'))
repls.append(('Устройство: {device}', 'Device: {device}'))

# CIFAR10 step 2
repls.append(('Параметры эксперимента и выбор режима', 'Experiment Parameters and Mode Selection'))
repls.append(('Интерактивные параметры эксперимента', 'Interactive experiment parameters'))
repls.append(('\'imbalance\' (животные -> транспорт) или \'balance\' (50% / 50%)', '\'imbalance\' (animals -> vehicles) or \'balance\' (50% / 50%)'))
repls.append(('Размеры выборок для профилирования PC', 'Sample sizes for PC profiling'))
repls.append(('Количество повторов для расчета среднего и погрешности (std)', 'Number of repetitions for mean and std calculation'))
repls.append((' Эпохи предобучения базовой CNN на источнике', ' Pretraining epochs of the base CNN on source'))
repls.append((' Эпохи адаптации на целевой задаче', ' Adaptation epochs on the target task'))
repls.append((' использовать HuggingFace datasets как сетевой источник', ' use HuggingFace datasets as a network source'))
repls.append(('# True: загрузить официальный CIFAR-10; False: синтетический быстрый режим', '# True: load official CIFAR-10; False: fast synthetic mode'))
repls.append(('Режим эксперимента:', 'Experiment mode:'))
repls.append(('Размеры выборок PC:', 'PC sample sizes:'))
repls.append(('Использование настоящего CIFAR-10:', 'Using real CIFAR-10:'))

# CIFAR10 step 3
repls.append(('Шаг 3. Загрузка данных и создание разбиений Источник / Цель', '3. Data Loading and Source / Target Split Creation'))
repls.append(('Загрузка полного датасета CIFAR-10 (60,000 изображений)...', 'Loading full CIFAR-10 dataset (60,000 images)...'))
repls.append(('Генерация быстрого синтетического бенчмарка для демонстрации в Colab...', 'Generating fast synthetic benchmark for Colab demonstration...'))
repls.append(('Источник: обучающих=', 'Source: training='))
repls.append(('тестовых=', 'test='))
repls.append(('Цель:     обучающих=', 'Target:   training='))

# CIFAR10 step 4
repls.append(('Шаг 4. Предобучение 6-слойной CNN на базовом датасете (Source)', '4. Pretraining 6-Layer CNN on Base Dataset (Source)'))
repls.append(('Предобучение на Source ({PRETRAIN_EPOCHS} эпох)...', 'Pretraining on Source ({PRETRAIN_EPOCHS} epochs)...'))

# CIFAR10 step 5
repls.append(('Шаг 5. Вычисление профилей общности', '5. Computing Generality Profiles'))
repls.append(('Воспроизведение Figure 4A / 4B)', 'Reproduction of Figure 4A / 4B)'))
repls.append(('В статье Di et al. (2025) профиль', 'In the Di et al. (2025) paper, the profile'))
repls.append(('Оценка PC для выборки n = {n_s} ({NUM_REPLICATIONS} повторов)...', 'PC estimation for sample size n = {n_s} ({NUM_REPLICATIONS} repetitions)...'))
repls.append(('Теоретический срез (argmax omega_j):', 'Theoretical cut-point (argmax omega_j):'))

# CIFAR10 step 6
repls.append(('Шаг 6. График Figure 4 (A/B): Мера общности слоев с погрешностями', '6. Figure 4 Plot (A/B): Layer Generality Measure with Errors'))
repls.append(('Линия теоретической границы', 'Theoretical boundary line'))
repls.append(('Сверточные слои', 'Convolutional Layers'))
repls.append(('Профиль общности слоев', 'Layer Generality Profile'))

# CIFAR10 step 7
repls.append(('Шаг 7. Запуск Алгоритма 1: Forward-Adding с warm-start', '7. Running Algorithm 1: Forward-Adding with Warm-Start'))
repls.append(('Время выполнения Алгоритма 1:', 'Algorithm 1 execution time:'))

# CIFAR10 step 8
repls.append(('Шаг 8. Полный перебор моделей', '8. Exhaustive Search of Models'))
repls.append(('Воспроизведение Figure 4C / 4D)', 'Reproduction of Figure 4C / 4D)'))
repls.append(('Обучение полного набора вложенных моделей M_0 ... M_6...', 'Training complete set of nested models M_0 ... M_6...'))
repls.append(('Обучение модели M_{k} (перенесено {k} слоев)...', 'Training model M_{k} ({k} transferred layers)...'))

# CIFAR10 step 9
repls.append(('Шаг 9. График Figure 4 (C/D) и Сравнительный Анализ', '9. Figure 4 Plot (C/D) and Comparative Analysis'))
repls.append(('График точности (Figure 4C)', 'Accuracy plot (Figure 4C)'))
repls.append(('Оптимум ^*=', 'Optimum ^*='))
repls.append(('График функции потерь (Figure 4D)', 'Loss function plot (Figure 4D)'))
repls.append(('Точность валидации моделей', 'Validation Accuracy of Models'))
repls.append(('Ошибка валидации моделей', 'Validation Error of Models'))
repls.append(('Сводный отчет', 'Summary report'))
repls.append(('ИТОГОВОЕ СРАВНЕНИЕ', 'FINAL COMPARISON'))
repls.append(('Теоретический расчет (argmax omega_j):', 'Theoretical calculation (argmax omega_j):'))
repls.append(('Алгоритм 1 (Forward-Adding):', 'Algorithm 1 (Forward-Adding):'))
repls.append(('Полный перебор (Exhaustive Search):', 'Exhaustive Search:'))
repls.append(('Фактическое ускорение алгоритма:', 'Actual algorithm speedup:'))
repls.append(('быстрее!', 'faster!'))
repls.append(('Шаг 10. Сохранение результатов в папку', '10. Saving Results to'))
repls.append(('Результаты эксперимента успешно сохранены в:', 'Experiment results successfully saved to:'))

for old, new in repls:
    content = content.replace(old, new)

with open(path1, 'w', encoding='utf-8') as f:
    f.write(content)

print('Translated build_notebooks.py:', len(repls), 'replacements')

# cifar10_experiment.py
path2 = os.path.join(d, 'cifar10_experiment.py')
with open(path2, encoding='utf-8') as f:
    content2 = f.read()

for old, new in [
    ('(ограничить число изображений CIFAR-10)', '(limit the number of CIFAR-10 images)'),
    ('(папка с cifar-10-python.tar.gz или кэшем)', '(folder with cifar-10-python.tar.gz or cache)'),
    ('(использовать HuggingFace datasets как сетевой источник)', '(use HuggingFace datasets as a network source)'),
]:
    content2 = content2.replace(old, new)

with open(path2, 'w', encoding='utf-8') as f:
    f.write(content2)

print('Translated cifar10_experiment.py')
