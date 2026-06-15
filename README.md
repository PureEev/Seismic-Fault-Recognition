# Seismic Fault Recognition

Исследовательский Python-проект для сегментации разломов в 3D-сейсмических
данных, self-supervised pretraining, super-resolution и воспроизводимой
валидации моделей.

Основная логика находится в `src/seismic_fault_recognition`, а ноутбуки из
`notebooks/` являются тонкими сценариями экспериментов поверх общего package
API и YAML-конфигураций.

## Структура репозитория

```text
configs/                         общие и experiment-specific YAML
docs/                            архитектура, данные и CLI
notebooks/                       17 финальных Jupyter-ноутбуков
scripts/validate_notebooks.py    локальный smoke-runner ноутбуков
src/seismic_fault_recognition/   модели, датасеты, training и validation
tests/                           unit и integration tests
```

## Установка

Требуется Python `3.10+`. Локальная проверка выполнялась на Python `3.12`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train,notebook,dev]"
```

Для preprocessing SEG-Y/HDF5 данных:

```bash
python -m pip install -e ".[preprocess]"
```

На CUDA-машине PyTorch лучше установить заранее командой с официальной
страницы PyTorch для используемой версии CUDA.

## Данные

Данные необходимо разместить локально. Базовая ожидаемая структура:

```text
data/
  thebe/
    raw/
    clean/
    clean_sr/
    sr/
    thebe_train_seis.npz
    thebe_train_fault.npz
    thebe_val_seis.npz
    thebe_val_fault.npz
    thebe_test_seis.npz
    thebe_test_fault.npz
  faultseg3d/
    train/seis/
    train/fault/
    validation/seis/
    validation/fault/
checkpoints/
outputs/
```

Пути для конкретного запуска задаются в `configs/experiments/*.yaml`.
Подробности форматов и структуры находятся в
[docs/data_inventory.md](docs/data_inventory.md).

## Запуск ноутбуков

Запускайте Jupyter из корня репозитория, чтобы ноутбуки нашли `src/` и
`configs/`:

```bash
jupyter lab
```

Рекомендуемый порядок, назначение и входы каждого ноутбука перечислены в
[notebooks/README.md](notebooks/README.md).

## Проверка проекта

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
sfr config validate \
  --base configs/datasphere.yaml \
  --experiments configs/experiments
```

Smoke-runner использует локальные данные и не запускает полное обучение:

```bash
python scripts/validate_notebooks.py \
  --thebe-crops /path/to/Thebe_Clean_Crops_V2 \
  --faultseg-validation /path/to/faultseg3d/validation \
  --swin-checkpoint /path/to/swinunetr_tiny_checkpoint.pth \
  --keep-workdir
```

Методика проверки описана в
[docs/notebook_validation.md](docs/notebook_validation.md).

## CLI

```bash
sfr recipes list
sfr recipes show swinunetr_thebe_finetune_raw
sfr data audit \
  --experiment configs/experiments/01_data_cleaning_and_audit.yaml \
  --output outputs/data_audit/report.json
sfr checkpoint inspect checkpoints/model.pth --json
```

Полный список команд: [docs/cli_reference.md](docs/cli_reference.md).

## Экспериментальный pipeline

1. SimMIM pretraining на сейсмических кубах.
2. Supervised pretraining на FaultSeg3D.
3. Fine-tuning на Thebe.
4. Отдельное обучение super-resolution модели.
5. Валидация сегментации и end-to-end SR + segmentation.

Архитектура training/validation слоя описана в
[docs/training_validation_architecture.md](docs/training_validation_architecture.md),
а финальная Swin-модель в
[docs/swinunetr_tiny.md](docs/swinunetr_tiny.md).

## Ограничения

- Полные датасеты и веса моделей не распространяются вместе с кодом.
- Smoke-проверка подтверждает корректный старт вычислений, но не качество
  сходимости полного обучения.
- VGG perceptual loss может потребовать предварительно доступные веса
  `torchvision`; в стандартном SR-конфиге `use_vgg: false`.
