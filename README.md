# Hyperplane-Forge — виявлення діпфейків через осідання фізичного маніфолду

> Дослідницький інструмент і бібліотека для виявлення діпфейків: відновлення глибинної поверхні обличчя, її осідання під фізично-вмотивованим енергетичним функціоналом і пошук місць зриву як ознаки синтетичного втручання.
>
> Розгорнутий технічний опис англійською мовою — у файлі [`README.dev.md`](README.dev.md). Повний математичний виклад і експериментальна частина — у [`paper/main.typ`](paper/main.typ).

---

## Автор

- **ПІБ**: Сірик Микита Артурович
- **Група**: ФеП-42с
- **Керівник**: Парубочий Віталій, асистент, викладач
- **Дата виконання**: 01.06.2026

---

## Загальна інформація

- **Тип проєкту**: дослідницька бібліотека з інтерфейсом командного рядка та окремим веб-демонстратором
- **Мова програмування**: Rust (математичне ядро) + Python (оркестратор, навчання, аналіз) + TypeScript/React (веб-демо)
- **Фреймворки / Бібліотеки**: PyTorch (CUDA-бекенд), PyO3 + `ndarray` + `rayon` (Rust-бекенд), EfficientNet-B0, scikit-learn (Gradient Boosting), MTCNN (виділення облич), Typst (текст роботи)
- **Цільові датасети**: FaceForensics++ c23, Celeb-DF v2

---

## Опис функціоналу

- Відновлення глибинної поверхні `z_forged` із вхідного зображення обличчя
- Осідання поверхні розв'язанням рівняння Ейлера–Лагранжа для енергетичного функціоналу `E(z)`
- Пошук розривів потоку `R = z* − z_ideal` і геометричних тріщин `L = Δz*` як ознак втручання
- Багатоканальний вхідний тензор: RGB + три фізичні мапи (`W_cnn`, `z*`, `R`) + три частотні мапи (DCT-енергія, відношення високих частот, FFT log-magnitude)
- Навчання шестиканальної та дев'ятиканальної EfficientNet-B0 зі збереженням попередньо навчених ваг ImageNet у RGB-каналах
- Покрокова перевірка на FaceForensics++ c23 і перенесення на Celeb-DF v2
- Веб-демонстратор (директорія `web/`) для одиничної перевірки відеофайлу обличчя
- Усі тривалі етапи відновлювані після збою — повторний запуск тієї самої команди продовжує роботу

---

## Експериментальні результати

Усі числа отримано на FaceForensics++ c23 з офіційним поділом 720/140/140 за відео без перетину осіб, 10 кадрів на відео, EfficientNet-B0, 20 епох, AdamW + cosine annealing, lr 2·10⁻⁴, WeightedRandomSampler з горизонтальним відображенням. Кросдатасетне перенесення — на Celeb-DF v2 (518-відео тестовий список).

| Тестова вибірка | Метрика | `baseline_3ch` | `physics_6ch` | `physics_9ch` |
|---|---|---:|---:|---:|
| **FF++ c23** (об'єднано) | Frame AUROC | **0.7309** | 0.7197 | 0.7028 |
| **FF++ c23** (об'єднано) | Video AUROC, середнє по кадрах | **0.8179** | 0.8176 | 0.7851 |
| **FF++ c23** (об'єднано) | Video AUROC, максимум по кадрах | 0.9130 | **0.9273** | 0.8996 |
| **FF++ c23 — Deepfakes** | Video AUROC, середнє | 0.8713 | **0.8787** | 0.8583 |
| **FF++ c23 — Face2Face** | Video AUROC, середнє | 0.8029 | 0.7731 | **0.8282** |
| **FF++ c23 — FaceSwap** | Video AUROC, середнє | **0.8150** | 0.7799 | 0.8147 |
| **FF++ c23 — NeuralTextures** | Video AUROC, середнє | 0.6647 | 0.6522 | **0.6694** |
| **Celeb-DF v2** (кросдатасет) | Frame AUROC | 0.5276 | 0.5382 | **0.5609** |
| **Celeb-DF v2** (кросдатасет) | Video AUROC, середнє | 0.5405 | 0.5458 | **0.6095** |
| **Celeb-DF v2** (кросдатасет) | Video AUROC, максимум | 0.5022 | 0.5542 | **0.5629** |

Жирним виділено найкраще значення в рядку. `physics_9ch` виграє у 5 із 10 порівнянь, `baseline_3ch` — у 3, `physics_6ch` — у 2. Найважливіший підсумок: дев'ятиканальна побудова з частотними мапами дає найкраще узагальнення поза тренувальним датасетом (Celeb-DF v2, Video AUROC середнє 0.6095 проти 0.5405 у базової моделі).

---

## Опис основних файлів та модулів

| Файл / Модуль | Призначення |
|---|---|
| `src/lib.rs` | Корінь Rust-крейту, експорт усіх фаз обчислень |
| `src/energy.rs` | Фаза 4 — енергетичний функціонал `E(z)` |
| `src/pde.rs` | Фаза 5 — розв'язувач Якобі з лінійним пошуком |
| `src/py_bindings.rs` | Прив'язка Rust-ядра до Python через PyO3 |
| `python/forge_detect/pipeline.py` | Точка входу `detect()` для одного зображення |
| `python/forge_detect/backends/` | CPU- (Rust) та CUDA- (PyTorch) реалізації операторів |
| `python/forge_detect/baseline_cnn.py` | Базова (RGB) та фізична (6/9 каналів) EfficientNet-B0 |
| `python/forge_detect/frequency_map.py` | Частотні канали (DCT, FFT) для Фази 3 |
| `python/forge_detect/datasets.py` | Адаптери FF++, Celeb-DF; інтерфейс `ChannelSource` |
| `scripts/extract_faces.py` | Виділення облич MTCNN із кадрів відео |
| `scripts/train_physics_cnn.py` | Навчання n-канальної EfficientNet-B0 |
| `scripts/eval_per_method.py` | Аналіз результатів за методами синтезу FF++ |
| `scripts/eval_celebdf.py` | Перевірка узагальнення на Celeb-DF v2 |
| `scripts/overnight_run.sh` | Послідовний запуск чотирьох важких етапів (~2 год на RTX 3080 Ti) |
| `paper/main.typ` | Повний текст роботи з виведенням математики та результатами |
| `web/` | Веб-демонстратор поверх натренованих ваг |

---

## Як запустити проєкт «з нуля»

### 1. Встановлення інструментів

- Rust toolchain (`rustup`, `cargo` ≥ 1.75)
- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) для керування Python-середовищем
- (опційно) `typst` для перекомпіляції тексту роботи
- (опційно) GPU з CUDA та `nvidia-smi` для повноцінного навчання

### 2. Клонування репозиторію

```bash
git clone https://github.com/siriknikita/deepfake-detection-project.git
cd deepfake-detection-project
```

### 3. Збірка ядра та встановлення залежностей

```bash
make dev-install
```

Команда створює `.venv`, ставить Python-залежності та збирає Rust-розширення з повною оптимізацією.

### 4. Швидка перевірка на одному зображенні

```bash
uv run --python .venv/bin/python forge-detect detect path/to/image.jpg \
    --visualize panel.png
```

Виходом є шестипанельна діагностична схема: вхідне зображення, мапа довіри `W_cnn`, поверхні `z_forged` та `z*`, мапа розривів `R`, тріщини `L`.

### 5. Підготовка датасетів і повне навчання

```bash
# виділити обличчя з кадрів FF++ c23
python scripts/extract_faces.py --data-root /path/to/FaceForensics++ \
    --dataset face-forensics --compression c23 --output-size 256

# попередньо обчислити фізичні мапи
python scripts/cache_physics_maps.py --data-root /path/to/FaceForensics++ \
    --variant heuristic --frames-subdir frames_faces

# натренувати шестиканальну модель
python scripts/train_physics_cnn.py --data-root /path/to/FaceForensics++ \
    --variant heuristic --runs-dir runs/physics_6ch_faces_heuristic \
    --device cuda --use-ff-splits --use-face-crops
```

Повний шлях від нуля до натренованої моделі займає ~5–6 годин на одному RTX 3080 Ti; скрипт `scripts/overnight_run.sh` запускає всі чотири важкі етапи послідовно з єдиним логом.

---

## Інструкція для користувача

1. **Командний рядок** (`forge-detect`):
   - `forge-detect detect <зображення>` — повний прогін на одному зображенні
   - `forge-detect detect <зображення> --visualize panel.png` — зберегти діагностичну схему
   - `forge-detect detect <зображення> --print-features` — вивести 24-вимірний вектор ознак
   - `forge-detect eval --dataset image-folder --data-root <реальні> --fake-dir <фейкові>` — оцінити модель на парі тек

2. **Веб-демонстратор** (директорія `web/`):
   - Завантажити коротке відео обличчя
   - Отримати усереднену оцінку довіри (від 0 до 1) разом із покадровою діагностикою
   - Поточна версія працює зі шкалою без калібрування; рішення «це діпфейк» приймає людина за провізорним порогом 0.5

3. **Перекомпіляція тексту роботи**:
   - `make paper` — згенерувати PDF з `paper/main.typ`
   - `make paper-watch` — режим миттєвого перерахунку під час редагування

---

## Проблеми і рішення

| Проблема | Рішення |
|---|---|
| `torch.cuda.is_available()` повертає `False` | Перевірити `nvidia-smi`; на обчислювальному кластері — звернутися до адміністратора |
| Скрипт обриває роботу через знеструмлення | Запускати під `scripts/continue.sh` у сесії `tmux`; усі довгі етапи відновлювані з контрольної точки |
| FaceForensics++ відмовляє у завантаженні | Підписати EULA та використати персональний скрипт-завантажувач, надісланий авторами датасету |
| Модель видає рішення близько 0.5 | Це очікувано — поточна версія некалібрована, поріг провізорний; докладніше у §12 файлу `paper/main.typ` |
| Збірка Rust-розширення падає | Оновити `rustup update stable`; перевірити, що `python` у `.venv` доступний як `python3` |

---

## Використані джерела

- Rössler A. et al. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019.
- Li Y. et al. *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics.* CVPR 2020.
- Tan M., Le Q. *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML 2019.
- Zhang K. et al. *Joint Face Detection and Alignment Using Multi-task Cascaded Convolutional Networks* (MTCNN). IEEE Signal Processing Letters, 2016.
- Офіційна документація PyTorch, PyO3, `ndarray`, scikit-learn, Typst.

---

## Скриншоти

Веб-демонстратор та діагностичні шестипанельні схеми наведено в §11–§12 повного тексту роботи (`paper/main.typ`).
