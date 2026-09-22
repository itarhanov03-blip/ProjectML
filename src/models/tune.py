"""Подбор гиперпараметров через Optuna.

Почему Optuna, а не GridSearchCV: перебор по сетке тратит одинаковое время
на заведомо плохие и на перспективные комбинации. Optuna строит модель того,
какие области пространства параметров выглядят выигрышно, и сходится к
хорошему решению за заметно меньшее число попыток.

Подбор поддержан для трёх семейств, вышедших в лидеры при сравнении моделей:
двух реализаций градиентного бустинга (XGBoost, LightGBM) и случайного леса
с экстремальной рандомизацией (ExtraTrees). Бустинг и лес устроены
принципиально по-разному, поэтому и пространства поиска у них разные.
"""

from __future__ import annotations

import json
from typing import Any

import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor

from src.config import RUNS_DIR, SEED, TARGET
from src.models.evaluate import compute_metrics
from src.models.train import wrap_log_target
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optuna по умолчанию очень разговорчива.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска для LightGBM.

    Границы выбраны вокруг разумных значений для табличной задачи такого
    размера: слишком глубокие деревья на 63 тысячах строк переобучаются,
    слишком мелкие не ловят взаимодействия признаков.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 16),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 0, 5),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def suggest_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска для XGBoost.

    Параметры те же по смыслу, что у LightGBM, но управление сложностью
    устроено через глубину дерева, а не через число листьев.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-4, 5.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }


def suggest_extra_trees_params(trial: optuna.Trial) -> dict[str, Any]:
    """Пространство поиска для ExtraTrees.

    У случайного леса параметров меньше и они понятнее: глубина, размер
    листа и доля признаков на разбиение. Число деревьев почти всегда
    "чем больше, тем лучше" — ограничено сверху временем обучения.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 10, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def build_tuned_model(model_type: str, params: dict[str, Any], seed: int = SEED):
    """Собирает модель нужного семейства с переданными параметрами."""
    if model_type == "lightgbm":
        return LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)
    if model_type == "xgboost":
        return XGBRegressor(random_state=seed, n_jobs=-1, tree_method="hist", **params)
    if model_type == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=-1, **params)
    raise ValueError(f"Неизвестное семейство моделей: {model_type}")


SUGGESTERS = {
    "lightgbm": suggest_lgbm_params,
    "xgboost": suggest_xgb_params,
    "extra_trees": suggest_extra_trees_params,
}


def tune_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    preprocessor,
    model_type: str = "lightgbm",
    n_trials: int = 30,
    log_target: bool = True,
    seed: int = SEED,
) -> optuna.Study:
    """Ищет гиперпараметры, минимизируя MAE на валидации.

    Оптимизируем именно ту метрику, по которой потом выбираем финальную
    модель. Подбирать по RMSE, а отчитываться по MAE — распространённая
    ошибка: оптимум у них разный.
    """
    suggest = SUGGESTERS[model_type]

    # Препроцессор обучается ОДИН раз, до начала перебора. Он не зависит от
    # гиперпараметров модели, а TF-IDF со сжатием занимает около 15 секунд —
    # пересчитывать его на каждой попытке значит тратить впустую больше
    # времени, чем на само обучение модели.
    # Утечки здесь нет: fit идёт только по train, val проходит через transform.
    logger.info("Готовлю признаки один раз для всех попыток")
    x_train = preprocessor.fit_transform(train, train[TARGET])
    x_val = preprocessor.transform(val)
    y_train = train[TARGET].to_numpy()
    y_val = val[TARGET].to_numpy()
    logger.info("Признаков: %d, обучающих строк: %d", x_train.shape[1], x_train.shape[0])

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        model = build_tuned_model(model_type, params, seed=seed)
        if log_target:
            model = wrap_log_target(model)

        model.fit(x_train, y_train)
        return compute_metrics(y_val, model.predict(x_val))["mae"]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name=f"{model_type}_mae"
    )

    logger.info("Подбор гиперпараметров %s: %d попыток", model_type, n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Лучший MAE: %.1f $", study.best_value)
    logger.info("Лучшие параметры: %s", study.best_params)
    return study


def study_to_frame(study: optuna.Study, top: int = 10) -> pd.DataFrame:
    """Таблица лучших попыток — идёт в отчёт как результат перебора."""
    frame = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    frame = frame.rename(columns={"value": "mae"}).sort_values("mae")
    frame.columns = [c.replace("params_", "") for c in frame.columns]
    return frame.head(top).round(4)


def save_study(study: optuna.Study, name: str = "cp2_lgbm_tuning") -> None:
    """Сохраняет итог подбора в experiments/runs."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "n_trials": len(study.trials),
        "best_mae": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {"number": t.number, "mae": t.value, "params": t.params}
            for t in study.trials
            if t.value is not None
        ],
    }
    path = RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Сохранил результат подбора: %s", path)
