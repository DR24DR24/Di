import os, re

d = os.path.dirname(os.path.abspath(__file__))

# --- Fix build_notebooks.py ---
path = os.path.join(d, "build_notebooks.py")
with open(path, encoding="utf-8") as f:
    content = f.read()

# Comprehensive list of remaining Russian fragments -> English
# Ordered: longest/most specific first to avoid partial-match issues
fixes = [
    # "сек." abbreviation (appears in multiple f-strings)
    ("сек.", "sec."),

    # Line 365: Conclusion text
    ("пик проекционной корреляции", "the peak of projection correlation"),

    # Line 365 (continued, with backslash escaping in file)
    ("в точности указывает на оптимальную точку переноса, где ошибка валидации минимальна.",
     "precisely identifies the optimal transfer point where validation error is minimal."),

    # Line 385
    ("с доверительными интервалами для объемов выборки",
     "with confidence intervals for sample sizes"),

    # Line 386
    ("Зависимость точности и ошибки от количества перенесенных слоев",
     "Dependence of accuracy and error on the number of transferred layers"),

    # Line 455
    ("в статье:", "in the paper:"),

    # Line 467: ## 3. ...
    ("Загрузка данных и создание разбиений Источник / Цель",
     "Data Loading and Source / Target Split Creation"),

    # Line 511: ## 4. ...
    ("Предобучение 6-слойной CNN на базовом датасете",
     "Pretraining 6-Layer CNN on Base Dataset"),

    # Line 538: ## 5. ...
    ("Вычисление профилей общности", "Computing Generality Profiles"),

    # Line 540
    ("оценивается для разных объемов выборки", "is evaluated for different sample sizes"),
    ("с несколькими повторениями.", "with multiple repetitions."),

    # Line 564
    ("Определение теоретического оптимума k*", "Determining the theoretical optimum k*"),

    # Line 573: ## 6. ...
    ("График Figure 4 (A/B): Generality Measure слоев с погрешностями",
     "Figure 4 Plot (A/B): Layer Generality Measure with Errors"),

    # Line 607: ## 7. ...
    ("Запуск Алгоритма 1: Forward-Adding с warm-start",
     "Running Algorithm 1: Forward-Adding with Warm-Start"),

    # Line 629: ## 8. ...
    ("Полный перебор моделей", "Exhaustive Search of Models"),

    # Line 638: Обучение модели -> Training model
    ("Обучение модели M_{k}", "Training model M_{k}"),

    # Line 656: ## 9. ...
    ("График Figure 4 (C/D) и Сравнительный Анализ",
     "Figure 4 Plot (C/D) and Comparative Analysis"),

    # Line 668, 677: Оптимум -> Optimum (in f-string labels)
    ("Оптимум $k^*=", "Optimum $k^*="),

    # Line 699: ## 10. ...
    ("Сохранение результатов в папку", "Saving Results to"),

    # Line 718
    ("Результаты эксперимента успешно сохранены в:",
     "Experiment results successfully saved to:"),

    # General: "Алгоритма" (genitive case) -> "Algorithm"
    ("Алгоритма", "Algorithm"),

    # "Модель" -> "Model" (in plot labels)
    ("(Модель $M_k$)", "(Model $M_k$)"),

    # "Теоретический оптимум" -> "Theoretical optimum" (line 351)
    ("Теоретический оптимум", "Theoretical optimum"),

    # "График" -> "Plot" (general for any remaining plot label headers)
    ("График ", "Plot "),

    # Any remaining "Запуск" -> "Running"
    ("Запуск ", "Running "),

    # Any remaining "Обучение" -> "Training"
    ("Обучение", "Training"),
]

for old, new in fixes:
    content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

# Verify
cyr = re.compile(r"[а-яА-ЯЁё]")
remaining = [(i + 1, l.rstrip()) for i, l in enumerate(content.splitlines(keepends=True)) if cyr.search(l)]
print(f"Remaining Russian lines in build_notebooks.py: {len(remaining)}")
for ln, text in remaining:
    print(f"  L{ln}: {text[:120]}")

# --- Fix cifar10_experiment.py (already clean, but verify) ---
path2 = os.path.join(d, "cifar10_experiment.py")
with open(path2, encoding="utf-8") as f:
    content2 = f.read()
remaining2 = [l for l in content2.splitlines() if cyr.search(l)]
print(f"Remaining Russian lines in cifar10_experiment.py: {len(remaining2)}")
