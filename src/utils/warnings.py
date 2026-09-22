"""Подавление ложных предупреждений numpy на macOS.

На macOS numpy 2.x собран с BLAS-бэкендом Accelerate, который выставляет
флаги плавающей точки (divide by zero, overflow, underflow, invalid) даже при
корректном умножении матриц. Предупреждения появляются на каждом .predict()
линейных моделей и засоряют вывод.

Что это действительно ложная тревога, проверено сравнением с np.einsum:
максимальное расхождение результатов 3.6e-15, все значения конечны.
Поэтому глушим только этот конкретный случай, а не RuntimeWarning вообще.
"""

from __future__ import annotations

import os
import warnings

_MESSAGES = ("divide by zero", "overflow", "underflow", "invalid value")


def silence_blas_warnings() -> None:
    """Глушит ложные RuntimeWarning от matmul на Accelerate BLAS.

    Фильтры warnings живут внутри процесса и не наследуются, а ансамбли и
    кросс-валидация с n_jobs=-1 запускают обучение в дочерних процессах
    joblib. Поэтому дополнительно выставляем PYTHONWARNINGS: эту переменную
    дочерние процессы читают при старте.
    """
    for message in _MESSAGES:
        warnings.filterwarnings(
            "ignore",
            message=f".*{message} encountered in matmul.*",
            category=RuntimeWarning,
        )

    # Формат записи: action:message:category:module:lineno.
    # message сопоставляется с началом текста предупреждения.
    rules = [f"ignore:{message} encountered in matmul:RuntimeWarning" for message in _MESSAGES]
    existing = os.environ.get("PYTHONWARNINGS", "")
    os.environ["PYTHONWARNINGS"] = ",".join([*filter(None, [existing]), *rules])
