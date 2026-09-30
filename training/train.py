"""Обучение модели фрод-скоринга.

Запуск из корня репозитория:
    python training/train.py [--data fraud_detector/train_data/train.csv]

Результат сохраняется в fraud_detector/models/:
    model.cbm                     — модель CatBoost
    preprocessing_artifacts.json  — статистики препроцессинга, порог и метрики
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(ROOT, 'fraud_detector', 'src'))
from preprocessing import CAT_COLS, build_features, fit_artifacts, save_artifacts  # noqa: E402

RANDOM_STATE = 42
MODELS_DIR = os.path.join(ROOT, 'fraud_detector', 'models')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('train')


def best_f1_threshold(y_true, proba):
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))
    return float(thresholds[best])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=os.path.join(ROOT, 'fraud_detector', 'train_data', 'train.csv'))
    parser.add_argument('--iterations', type=int, default=1500)
    args = parser.parse_args()

    raw = pd.read_csv(args.data)
    logger.info('Loaded %s, fraud rate %.4f', raw.shape, raw['target'].mean())

    train_raw, valid_raw = train_test_split(
        raw, test_size=0.2, stratify=raw['target'], random_state=RANDOM_STATE)

    # Артефакты считаем только на train-части, чтобы валидация была честной
    artifacts = fit_artifacts(train_raw)
    X_train, y_train = build_features(train_raw, artifacts), train_raw['target']
    X_valid, y_valid = build_features(valid_raw, artifacts), valid_raw['target']
    feature_names = list(X_train.columns)

    params = dict(
        iterations=args.iterations,
        learning_rate=0.08,
        depth=6,
        max_ctr_complexity=1,  # без комбинаций категорий: модель остаётся компактной
        eval_metric='PRAUC',
        task_type='CPU',
        thread_count=-1,
        random_seed=RANDOM_STATE,
        verbose=100,
    )
    model = CatBoostClassifier(**params, od_type='Iter', od_wait=150)
    model.fit(Pool(X_train, y_train, cat_features=CAT_COLS),
              eval_set=Pool(X_valid, y_valid, cat_features=CAT_COLS),
              use_best_model=True)

    proba = model.predict_proba(X_valid)[:, 1]
    threshold = best_f1_threshold(y_valid, proba)
    metrics = {
        'roc_auc': float(roc_auc_score(y_valid, proba)),
        'pr_auc': float(average_precision_score(y_valid, proba)),
        'f1': float(f1_score(y_valid, proba >= threshold)),
        'threshold': threshold,
        'best_iteration': int(model.get_best_iteration()),
    }
    logger.info('Validation metrics: %s', metrics)

    # Финальная модель: артефакты и обучение на всех данных с найденным числом итераций
    artifacts = fit_artifacts(raw)
    X_full = build_features(raw, artifacts)
    final_params = {**params, 'iterations': metrics['best_iteration'] + 1, 'eval_metric': 'Logloss'}
    final_model = CatBoostClassifier(**final_params)
    final_model.fit(Pool(X_full, raw['target'], cat_features=CAT_COLS))

    os.makedirs(MODELS_DIR, exist_ok=True)
    final_model.save_model(os.path.join(MODELS_DIR, 'model.cbm'))
    artifacts.update({
        'feature_names': feature_names,
        'cat_features': CAT_COLS,
        'threshold': threshold,
        'validation_metrics': metrics,
    })
    save_artifacts(artifacts, os.path.join(MODELS_DIR, 'preprocessing_artifacts.json'))

    importances = pd.Series(final_model.get_feature_importance(), index=feature_names)
    logger.info('Feature importance:\n%s', importances.sort_values(ascending=False).round(2).to_string())
    logger.info('Saved model and artifacts to %s', MODELS_DIR)


if __name__ == '__main__':
    main()
