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

import warnings


def silence_blas_warnings() -> None:
    """Глушит ложные RuntimeWarning от matmul на Accelerate BLAS."""
    for message in ("divide by zero", "overflow", "underflow", "invalid value"):
        warnings.filterwarnings(
            "ignore",
            message=f".*{message} encountered in matmul.*",
            category=RuntimeWarning,
        )
