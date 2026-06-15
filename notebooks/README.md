# Ноутбуки

Ноутбуки являются сценариями запуска. Датасеты, модели, loss-функции,
trainers, метрики и сохранение checkpoint-файлов реализованы в
`src/seismic_fault_recognition`.

## Общие правила

- Запускайте Jupyter из корня репозитория.
- Настройте пути в соответствующем `configs/experiments/*.yaml`.
- Основной пространственный размер для training и validation: `128³`.
- ClearML опционален; локальный запуск возможен с `clearml.enabled: false`.
- Результаты сохраняются в `outputs/`, веса в `checkpoints/`; обе директории
  исключены из Git.
- Ноутбуки хранятся без execution output, чтобы diff оставался читаемым.

## Рекомендуемый порядок

| № | Ноутбук | Назначение |
| ---: | --- | --- |
| 00 | `00_environment_and_data_check.ipynb` | Проверка окружения, configs и локальных путей |
| 01 | `01_data_cleaning_and_audit.ipynb` | Аудит NPZ-данных, форм, dtype и пар |
| 02 | `02_simmim_swinunetr_thebe_pretrain.ipynb` | SimMIM pretraining для `swin_tiny` |
| 03 | `03_simmim_omniseis_thebe_pretrain.ipynb` | SimMIM pretraining для OmniSeis |
| 04 | `04_faultseg3d_swin_tiny_pretrain.ipynb` | FaultSeg3D pretraining для `swin_tiny` |
| 05 | `05_faultseg3d_omniseis_pretrain.ipynb` | FaultSeg3D pretraining для OmniSeis |
| 06 | `06_swinunetr_thebe_finetune_raw.ipynb` | Fine-tuning Swin на raw Thebe |
| 07 | `07_swinunetr_thebe_finetune_clean_sr_cubes.ipynb` | Fine-tuning Swin на clean/SR Thebe |
| 08 | `08_omniseis_thebe_finetune_raw.ipynb` | Fine-tuning OmniSeis на raw Thebe |
| 09 | `09_omniseis_thebe_finetune_clean_cubes.ipynb` | Fine-tuning OmniSeis на clean Thebe |
| 10 | `10_omniseis_thebe_finetune_clean_sr_aug_reg.ipynb` | OmniSeis с SR, augmentation и regularization |
| 11 | `11_sr_training_seisgan.ipynb` | Обучение 3D super-resolution |
| 12 | `12_segmentation_validation.ipynb` | Валидация segmentation checkpoint |
| 13 | `13_end_to_end_sr_segmentation_validation.ipynb` | End-to-end SR + segmentation |
| 14 | `14_swinunetr_tiny.ipynb` | Проверка финальной Swin Tiny на входе `128³` |
| 15 | `15_faultformer_variants.ipynb` | Проверка FaultFormer |
| 16 | `16_3d_visualization.ipynb` | Генерация интерактивной 3D-визуализации |

## Необходимые локальные ресурсы

- Thebe NPZ или директории парных `seis/*.npz` и `fault/*.npz`.
- FaultSeg3D memmap-файлы в парных каталогах `seis/` и `fault/`.
- Segmentation checkpoint для ноутбуков 12 и 13.
- SR checkpoint для ноутбука 13.

Точные ключи путей указаны в YAML каждого эксперимента.

## Smoke-проверка

`scripts/validate_notebooks.py` создаёт изолированную временную копию проекта,
подставляет по одному локальному примеру данных и выполняет все ячейки.
В обучающих ноутбуках финальный длительный цикл временно заменяется одним
реальным шагом оптимизатора; оригинальные `.ipynb` не изменяются.

```bash
python scripts/validate_notebooks.py \
  --thebe-crops /path/to/Thebe_Clean_Crops_V2 \
  --faultseg-validation /path/to/faultseg3d/validation \
  --swin-checkpoint /path/to/swinunetr_tiny_checkpoint.pth
```
