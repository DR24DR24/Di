# Replication of Di et al. (2025) Algorithm: Computationally Efficient General Convolutional Layer Selection for Transfer Learning

Implementation of the algorithm for selecting general convolutional layers for transfer learning (Transfer Learning) based on the paper:

> **Xiao Di, Bingnan Li, Wenzhan Song, Yuan Ke (2025)**  
> *Computationally Efficient General Convolutional Layer Selection for Transfer Learning*.  
> **Stat**, 14:e70037. DOI: [10.1002/sta4.70037](https://doi.org/10.1002/sta4.70037)  
> Appendix: `appendix.pdf` (proofs of Theorems 1 and 2, Lemmas 1.1 and 1.2).

The package source code is located in:  
`I:\Мой диск\user\goit\Diplom\var1_Di\code\Di`

---

## 1. Theoretical Foundation and Algorithm

### 1.1 Phase Transition Phenomenon
When pre-training deep convolutional networks (CNNs) on a base dataset (ImageNet, CIFAR, etc.):
- **Early layers ($1 \le j \le j^*$)** extract universal (general) features (edges, textures, color patches) transferable to target tasks without risk of overfitting.
- **Deep layers ($j > j^*$)** specialize (specific) to source classes and distribution. Transferring them to small target datasets leads to **overfitting** and degradation of generalization ability.

### 1.2 Layer Generality Measure: Projection Correlation (PC)
For the $j$-th layer network with feature representation $\Phi^j(X^t) \in \mathbb{R}^p$ and target responses $Y^t \in \mathbb{R}^q$ the generality measure is defined as:
$$\omega_j = \mathrm{PC}(\Phi^j(X^t), Y^t) \in [0, 1]$$

The sample estimate $\widehat{\mathrm{PC}}(X, Y)$ is constructed according to Zhu et al. (2017) and Di et al. (2025, formulas 1–4):
$$a_{klr} = \arccos\left(\frac{(X_k - X_r)^T(X_l - X_r)}{\|X_k - X_r\| \|X_l - X_r\|}\right), \quad a_{klr} = 0 \text{ при } k=r \text{ или } l=r$$
$$b_{klr} = \arccos\left(\frac{(Y_k - Y_r)^T(Y_l - Y_r)}{\|Y_k - Y_r\| \|Y_l - Y_r\|}\right), \quad b_{klr} = 0 \text{ при } k=r \text{ или } l=r$$

After double-centering the matrices by indices $k$ and $l$ for each reference object $r \in \{1, \dots, n\}$:
$$A_{klr} = a_{klr} - \bar{a}_{k\cdot r} - \bar{a}_{\cdot lr} + \bar{a}_{\cdot\cdot r}, \quad B_{klr} = b_{klr} - \bar{b}_{k\cdot r} - \bar{b}_{\cdot lr} + \bar{b}_{\cdot\cdot r}$$

Squared projection covariance and projection correlation:
$$\widehat{\mathrm{Pcov}}(X, Y)^2 = \frac{1}{n^3} \sum_{k,l,r=1}^n A_{klr} B_{klr}$$
$$\widehat{\mathrm{PC}}(X, Y) = \frac{\widehat{\mathrm{Pcov}}(X, Y)}{\sqrt{\widehat{\mathrm{Pcov}}(X, X) \widehat{\mathrm{Pcov}}(Y, Y)}}$$

**PC metric properties:**
1. $\mathrm{PC}(X, Y) = 0 \iff X \perp Y$ (equals zero if and only if the variables are statistically independent).
2. No moment conditions required (works with heavy-tailed distributions).
3. Invariant to orthogonal rotations and scale shifts.
4. Computed **in a single forward pass** without training parameters.

### 1.3 Non-Asymptotic Bounds (Theorems 1, 2 and Appendix)
Proven in `appendix.pdf` using Serfling's U-statistics apparatus (Serfling 2009):
$$\Pr\left(\left|\widehat{\mathrm{PC}}(X, Y)^2 - \mathrm{PC}(X, Y)^2\right| > \varepsilon\right) \le 5 c_1 \exp\left(-c_2 n \varepsilon^2 \gamma_0\right)$$
где $c_1 = 6$, $c_2 = \frac{1}{320 \pi^4}$, $M = (2\pi)^2$, $\gamma_0 = \min\left\{\frac{\gamma_x^3 \gamma_y^3}{4M^4}, \frac{\gamma_x^2 \gamma_y^2}{4M^4}, \frac{\gamma_x \gamma_y}{4}\right\}$.

From this, the phase transition point estimate:
$$\hat{j}^* = \arg\max_{1 \le j \le J} \hat{\omega}_j$$
is consistent with exponential convergence rate:
$$\Pr(\hat{j}^* = j^*) > 1 - c_1' \exp\left(-c_2' n^{1-2\tau} \gamma\right)$$
under minimum signal strength condition $\min \Delta_k > c_3 n^{-\tau}$ ($\tau < 1/2$).

### 1.4 Алгоритм 1: Forward Adding Selection of General Layers
```text
Input: Pre-trained model Φ with J layers, target dataset (X^t, Y^t), loss function e
1: Initialize:
2:   Compute projection correlation \hat{\omega}_j = \widehat{PC}(Φ^j(X^t), Y^t) for j = 1, ..., J
3:   Set initial cut-off k_0 = argmax_j \hat{\omega}_j (Warm Start)
4:   Train initial candidate model M_{k_0} on target dataset using first k_0 layers; compute loss e_0
5: Iterative Update:
6: while True do
7:   Increment layer count: k = k_0 + 1
8:   if k > J then break
9:   Train updated model M_k on target dataset using first k layers; compute loss e_1
10:  if e_1 >= e_0 then
11:    Stop and set final number of layers k = k_0
12:    Break
13:  else
14:    Update k_0 = k and e_0 = e_1
15:  end if
16: end while
Output: Optimized model for target task with selected layers k transferred.
```

---

## 2. Структура репозитория `code/Di`

```
code/Di/
├── __init__.py               # Экспорт API пакета
├── projection_correlation.py # Эффективный расчет PC (O(n^2) RAM chunking, PyTorch/NumPy)
├── models.py                 # Архитектура 6-layer CNN (Figure 3) и трансфер-модели M_k
├── layer_extractor.py        # Forward-hooks для извлечения признаков Phi^j(X) (GAP / Flatten)
├── di_algorithm.py           # Реализация Алгоритма 1 (Warm Start + Forward-Adding)
├── dataset_splits.py         # Сплиты CIFAR-10 (Imbalance / Balance) и синтетический бенчмарк
├── cifar10_experiment.py     # Скрипт репликации экспериментов CIFAR-10 (Figure 4)
├── demo_quickstart.py        # Экспресс-демонстрация полного цикла (15 секунд)
├── test_di.py                # Модульные и интеграционные тесты (9 тестов)
└── README.md                 # Данная документация
```

---

## 3. Ключевые инженерные улучшения над исходниками

1. **Решение проблемы памяти $O(n^3) \to O(\text{chunk} \cdot n^2)$:**
   Оригинальный скрипт `PC_screen.py` аллоцировал тензор обратных косинусов размером $(n, n, n)$, что при $n \ge 500$ требовало гигабайты памяти и вызывало Out-Of-Memory.
   В модуле `projection_correlation.py` вычисление векторизовано по чанкам индекса $r$ (`chunk_size=64`). Расход памяти составляет всего $O(\text{chunk} \cdot n^2)$ (несколько мегабайт), что позволяет точно вычислять PC для $n=1000, 2000$ на CPU и GPU.
2. **Быстрый одномерный путь (`1D target`):**
   Для скалярных целевых меток (классы, регрессия) применено аналитическое упрощение ядра без матричных умножений по $Y$, ускоряющее расчет в разы.
3. **Полная независимость от устаревших зависимостей:**
   Код работает на стандартных `torch` и `numpy`, без устаревших вызовов Python 2.7 и без обязательного требования scipy/torchvision.
4. **Реализация теплого старта Алгоритма 1:**
   В отличие от полного перебора 7 вложенных моделей $\mathcal{M}_0, \dots, \mathcal{M}_6$, алгоритм Di et al. стартует с $k_0 = \arg\max \hat{\omega}_j$ и обучает всего 1-2 модели до критерия останова.

---

## 4. Быстрый запуск

### 4.1 Запуск экспресс-демонстрации
Выполняет полный пайплайн (генерация задачи с фазовым переходом, предобучение 6-layer CNN, расчет профиля PC, Алгоритм 1, отчет):
```powershell
python "i:\Мой диск\user\goit\Diplom\var1_Di\code\Di\demo_quickstart.py"
```
*Время выполнения: ~15 секунд на CPU.*

### 4.2 Запуск тестового набора
Проверяет математические свойства PC, модель CNN, заморозку слоев и Алгоритм 1:
```powershell
python "i:\Мой диск\user\goit\Diplom\var1_Di\code\Di\test_di.py"
```
*Результат: 9/9 тестов пройдены успешно (OK).*

### 4.3 Воспроизведение экспериментов Section 4.1 (CIFAR-10)
- **Imbalance case** (5 классов source против 5 non-overlapping классов target):
  ```powershell
  python "i:\Мой диск\user\goit\Diplom\var1_Di\code\Di\cifar10_experiment.py" --case imbalance --sample_size 400 --epochs 8
  ```
  *Результат:* фазовый переход на 4-м слое ($k^* = 4$), модель $\mathcal{M}_4$ достигает максимума точности.

- **Balance case** (все 10 классов распределены поровну):
  ```powershell
  python "i:\Мой диск\user\goit\Diplom\var1_Di\code\Di\cifar10_experiment.py" --case balance --sample_size 400 --epochs 8
  ```
  *Результат:* монотонный рост $\omega_j$ до 6-го слоя ($k^* = 6$, отсутствие фазового перехода).

---

## 5. Использование в собственном коде (API)

```python
import torch
from Di import (
    CIFAR6CNN,
    compute_pc_profile,
    estimate_phase_transition,
    forward_adding_layer_selection,
    projection_correlation
)

# 1. Загрузка предобученной модели и целевого DataLoader
model = CIFAR6CNN(num_classes=10)
# ... загрузка весов ...

# 2. Расчет профиля проекционной корреляции (Zero-Training фазовый переход)
profile = compute_pc_profile(
    model=model,
    dataloader=target_dataloader,
    max_samples=500,
    pooling='gap'
)

# 3. Теоретическая точка перехода без обучения целевой сети (Теорема 2)
pt = estimate_phase_transition(profile)
print(f"Теоретический cut-off j* = {pt['j_star_index']} ({pt['j_star_name']})")
print(f"Общие переносимые слои: {pt['general_layers']}")
print(f"Специфические слои:     {pt['specific_layers']}")

# 4. Запуск Алгоритма 1 (оптимальный перенос с теплым стартом)
result = forward_adding_layer_selection(
    pretrained_model=model,
    target_train_loader=target_train_loader,
    target_val_loader=target_val_loader,
    num_target_classes=5,
    train_epochs=15,
    freeze_transferred=True
)

print(f"Выбранное число слоев для переноса: {result['selected_k']}")
best_target_model = result['best_model']
```
