"""Эксперименты CP2: от сравнения моделей до финального прогона на тесте.

Запуск целиком:      python -m src.models.experiments
Запуск по этапам:    python -m src.models.experiments --stage models
                     python -m src.models.experiments --stage features
                     python -m src.models.experiments --stage dimred
                     python -m src.models.experiments --stage tuning
                     python -m src.models.experiments --stage final

Правило, которое соблюдается во всех этапах: тестовая выборка не участвует
ни в одном сравнении. Её видит только финальный прогон, ровно один раз.
"""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.config import MODELS_DIR, PROCESSED_DIR, RUNS_DIR, SEED, TARGET
from src.data.split import check_leakage, split_data
from src.models.dimred import (
    PCA_COMPONENTS_GRID,
    build_pca_pipeline,
    fit_pca,
    plot_dimred_curve,
    plot_explained_variance,
    plot_pca_scatter,
)
from src.models.evaluate import (
    compute_metrics,
    error_analysis,
    format_metrics,
    metrics_table,
    save_run,
)
from src.models.train import (
    add_text_column,
    build_model_zoo,
    build_pipeline,
    build_preprocessor,
    run_experiment,
)
from src.models.tune import build_tuned_model, save_study, study_to_frame, tune_model
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.warnings import silence_blas_warnings

logger = get_logger(__name__)


def load_prepared(sample: int | None = None) -> pd.DataFrame:
    """Читает подготовленный в CP1 датасет и добавляет колонку с текстом."""
    path = PROCESSED_DIR / "dataset.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет {path}. Сначала выполните: python -m src.models.baseline")

    df = pd.read_parquet(path)
    if sample:
        df = df.sample(sample, random_state=SEED)
        logger.info("Работаю на подвыборке: %d строк", len(df))

    return add_text_column(df)


def get_splits(df: pd.DataFrame):
    """Групповой сплит из CP1 плюс проверка на утечку."""
    train, val, test = split_data(df)
    check_leakage(train, test)
    return train, val, test


def stage_models(df, train, val) -> dict[str, dict[str, float]]:
    """Э5. Сравнение моделей на одинаковом наборе признаков."""
    logger.info("=== Э5. Сравнение моделей (признаки: таблица + target encoding) ===")
    preprocessor = build_preprocessor(df, use_target_encoding=True, use_text=False)

    results = {}
    for name, model in build_model_zoo().items():
        results[name] = run_experiment(name, model, preprocessor, train, val)

    save_run("cp2_model_zoo", results, params={"seed": SEED}, notes="Э5: сравнение моделей.")
    return results


def stage_features(df, train, val) -> dict[str, dict[str, float]]:
    """Э6-Э8. Что даёт target encoding, текст и обучение на log(price)."""
    logger.info("=== Э6-Э8. Наборы признаков и преобразование таргета ===")
    results = {}

    def lgbm():
        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            random_state=SEED,
            n_jobs=-1,
            verbose=-1,
        )

    variants = [
        ("без target encoding", {"use_target_encoding": False, "use_text": False}, False),
        ("с target encoding", {"use_target_encoding": True, "use_text": False}, False),
        ("с текстом (TF-IDF+SVD)", {"use_target_encoding": True, "use_text": True}, False),
        ("с текстом + log(price)", {"use_target_encoding": True, "use_text": True}, True),
    ]

    for name, kwargs, log_target in variants:
        preprocessor = build_preprocessor(df, **kwargs)
        results[name] = run_experiment(
            name, lgbm(), preprocessor, train, val, log_target=log_target
        )

    save_run(
        "cp2_feature_sets",
        results,
        params={"model": "lightgbm"},
        notes="Э6-Э8: вклад target encoding, текста и лог-таргета.",
    )
    return results


def stage_finalists(df, train, val) -> dict[str, dict[str, float]]:
    """Э9. Лидеры сравнения — на лучшем наборе признаков.

    Модели в Э5 сравнивались без текста, а текст оказался самой полезной
    группой признаков. Поэтому кандидата на подбор гиперпараметров выбираем
    заново: порядок моделей на другом наборе признаков может измениться.
    """
    logger.info("=== Э9. Финалисты на признаках с текстом и лог-таргетом ===")
    preprocessor = build_preprocessor(df, use_target_encoding=True, use_text=True)

    zoo = build_model_zoo()
    finalists = ["extra_trees", "xgboost", "lightgbm", "random_forest", "voting_lgbm_xgb_rf"]

    results = {}
    for name in finalists:
        results[name] = run_experiment(name, zoo[name], preprocessor, train, val, log_target=True)

    best = min(results, key=lambda k: results[k]["mae"])
    logger.info("Лучший финалист: %s (MAE %.1f $)", best, results[best]["mae"])

    save_run(
        "cp2_finalists",
        results,
        params={"log_target": True, "text": True},
        notes="Э9: топ-модели на лучшем наборе признаков.",
    )
    return results


def stage_dimred(df, train, val) -> dict[str, dict[str, float]]:
    """Э10. Снижение размерности и его влияние на KNN."""
    logger.info("=== Э10. Снижение размерности ===")
    preprocessor = build_preprocessor(df, use_target_encoding=True, use_text=False)

    x_train = preprocessor.fit_transform(train, train[TARGET])
    logger.info("Размерность после препроцессора: %d признаков", x_train.shape[1])

    pca = fit_pca(x_train, n_components=0.95)
    logger.info(
        "Для 95%% дисперсии хватает %d компонент из %d",
        pca.named_steps["pca"].n_components_,
        x_train.shape[1],
    )
    plot_explained_variance(pca)
    plot_pca_scatter(pca.transform(x_train), train[TARGET].to_numpy())

    results = {}
    baseline = run_experiment(
        "knn на всех признаках",
        KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
        build_preprocessor(df, use_target_encoding=True, use_text=False),
        train,
        val,
    )
    results["knn_all_features"] = baseline

    curve = {}
    for n_components in PCA_COMPONENTS_GRID:
        pipeline = build_pca_pipeline(
            build_preprocessor(df, use_target_encoding=True, use_text=False),
            KNeighborsRegressor(n_neighbors=10, n_jobs=-1),
            n_components=n_components,
        )
        pipeline.fit(train, train[TARGET])
        metrics = compute_metrics(val[TARGET].to_numpy(), pipeline.predict(val))
        results[f"knn_pca_{n_components}"] = metrics
        curve[n_components] = metrics["mae"]
        logger.info("knn + PCA(%3d) %s", n_components, format_metrics(metrics))

    plot_dimred_curve(curve, baseline_mae=baseline["mae"])
    save_run("cp2_dimred", results, params={"grid": PCA_COMPONENTS_GRID}, notes="Э10: PCA и KNN.")
    return results


def stage_tuning(df, train, val, n_trials: int = 30):
    """Э11. Подбор гиперпараметров для двух лучших семейств.

    Подбираем и для бустинга, и для случайного леса: они устроены по-разному,
    и заранее неочевидно, кто выиграет после настройки. Побеждает тот,
    у кого ниже MAE на валидации.
    """
    logger.info("=== Э11. Подбор гиперпараметров ===")
    preprocessor = build_preprocessor(df, use_target_encoding=True, use_text=True)

    studies = {}
    for model_type in ("xgboost", "extra_trees"):
        study = tune_model(
            train,
            val,
            preprocessor,
            model_type=model_type,
            n_trials=n_trials,
            log_target=True,
        )
        save_study(study, name=f"cp2_tuning_{model_type}")
        print(f"\nЛучшие попытки — {model_type}:\n", study_to_frame(study, top=5).to_string())
        studies[model_type] = study

    best_type = min(studies, key=lambda k: studies[k].best_value)
    logger.info("Победитель подбора: %s (MAE %.1f $)", best_type, studies[best_type].best_value)
    return best_type, studies[best_type]


# Запасные параметры на случай запуска финала без этапа подбора.
DEFAULT_FINAL_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.03,
    "max_depth": 10,
    "min_child_weight": 13,
    "subsample": 0.65,
    "colsample_bytree": 0.74,
}


def load_best_params(model_type: str) -> tuple[str, dict | None]:
    """Достаёт лучшие параметры из сохранённого результата подбора.

    Нужно, чтобы этап final можно было запускать отдельно от tuning,
    не теряя результат многочасового перебора.
    """
    path = RUNS_DIR / f"cp2_tuning_{model_type}.json"
    if not path.exists():
        logger.warning("Нет %s, беру параметры по умолчанию", path.name)
        return model_type, None

    study = json.loads(path.read_text(encoding="utf-8"))
    logger.info("Беру параметры подбора %s: MAE на валидации %.1f $", model_type, study["best_mae"])
    return model_type, study["best_params"]


def stage_final(df, train, val, test, model_type: str = "xgboost", best_params: dict | None = None):
    """Финал. Обучение на train+val и единственный прогон по тесту."""
    if best_params is None:
        model_type, best_params = load_best_params(model_type)

    logger.info("=== Финальная модель: %s ===", model_type)
    preprocessor = build_preprocessor(df, use_target_encoding=True, use_text=True)

    params = best_params or DEFAULT_FINAL_PARAMS
    model = build_tuned_model(model_type, params)

    # log_target=True — так подбирались гиперпараметры, финальная модель
    # должна обучаться ровно в той же конфигурации.
    pipeline = build_pipeline(preprocessor, model, log_target=True)

    # Финальная модель учится на train+val: валидация свою работу отработала
    # при отборе модели и гиперпараметров, и держать её пустой больше незачем.
    train_full = pd.concat([train, val], ignore_index=True)
    pipeline.fit(train_full, train_full[TARGET])

    predictions = pipeline.predict(test)
    metrics = compute_metrics(test[TARGET].to_numpy(), predictions)
    logger.info("ТЕСТ: %s", format_metrics(metrics))
    print(
        "\nОшибка по ценовым сегментам:\n",
        error_analysis(test[TARGET].to_numpy(), predictions).to_string(),
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "final_model.pkl"
    joblib.dump(pipeline, model_path)
    logger.info("Сохранил финальную модель: %s", model_path)

    save_run(
        "cp2_final",
        {f"final_{model_type}_test": metrics},
        params={"model_type": model_type, **params},
        notes="Финальная модель: обучение на train+val, оценка на тесте (единственный раз).",
    )
    return pipeline, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Эксперименты CP2")
    parser.add_argument(
        "--stage",
        choices=["models", "features", "finalists", "dimred", "tuning", "final", "all"],
        default="all",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="работать на подвыборке (для черновых прогонов)"
    )
    parser.add_argument("--trials", type=int, default=30, help="попыток Optuna")
    args = parser.parse_args()

    silence_blas_warnings()
    set_seed(SEED)

    df = load_prepared(sample=args.sample)
    train, val, test = get_splits(df)

    results: dict[str, dict[str, float]] = {}

    if args.stage in ("models", "all"):
        results |= stage_models(df, train, val)
    if args.stage in ("features", "all"):
        results |= stage_features(df, train, val)
    if args.stage in ("finalists", "all"):
        results |= stage_finalists(df, train, val)
    if args.stage in ("dimred", "all"):
        results |= stage_dimred(df, train, val)

    best_type, study = "xgboost", None
    if args.stage in ("tuning", "all"):
        best_type, study = stage_tuning(df, train, val, n_trials=args.trials)
    if args.stage in ("final", "all"):
        stage_final(
            df,
            train,
            val,
            test,
            model_type=best_type,
            best_params=study.best_params if study else None,
        )

    if results:
        print("\n" + metrics_table(results).to_string())


if __name__ == "__main__":
    main()
