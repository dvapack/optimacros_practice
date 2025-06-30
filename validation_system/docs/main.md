# Описание

В данной библиотеке представлены инструменты для загрузки, валидации и сравнения гиперпараметров.

Работа с данной библиотекой выглядит следующим образом:

Загрузка данных посредством dataloader -> валидация гиперпараметров с помощью validation -> сохранение валидированных гиперпараметров с помощью dataloder -> .... -> сравнение гиперпараметров с помощью model comparision

## Dataloader

Инструмент для манипулирования гиперпараметрами и экземплярами моделей. Поддерживает загрузку гиперпараметром от Optimacros, Python, получение гиперпараметров из обученных экземпляров моделей, сохранение гиперпараметров и обученных экземпляров моделей.

[Пример использования](validation_system/examples/data_manipulation_example.ipynb)

[Исходный код](validation_system/validation_system/dataloader.py)

## Validation

Инструмент для валидирования гиперпараметров, полученных от Optimacros.

[Пример использования](validation_system/examples/validation_example.ipynb)

[Исходный код](validation_system/validation_system/validation.py)

## Model comparision

Инструмент для сравнения гиперпараметров и метрик моделей.

[Пример использования](validation_system/examples/models_comparision_example.ipynb)

[Исходный код](validation_system/validation_system/model_comparision.py)

---
Документация представлена в виде doc-string комментариев в коде. Общие примеры использования представлены в папке /examples
