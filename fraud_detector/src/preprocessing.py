"""Препроцессинг транзакций.

Один и тот же код используется при обучении (training/train.py) и при инференсе
в сервисе, чтобы признаки гарантированно совпадали. Все статистики, посчитанные
на train, сохраняются в JSON-артефакт, поэтому в рантайме train.csv не нужен.
"""
import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PII_COLS = ['name_1', 'name_2', 'street', 'post_code']
CAT_COLS = ['merch', 'cat_id', 'gender', 'us_state', 'jobs', 'one_city']
FREQ_COLS = ['merch', 'jobs', 'one_city']
NUM_COLS = ['amount', 'population_city', 'lat', 'lon', 'merchant_lat', 'merchant_lon']

RARE = '__rare__'
UNKNOWN = '__unknown__'
MIN_CAT_COUNT = 30
EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def fit_artifacts(train):
    """Считает на train все статистики, нужные для build_features."""
    logger.info('Fitting preprocessing artifacts on %d rows', len(train))
    artifacts = {'known_categories': {}, 'frequencies': {}}

    for col in CAT_COLS:
        counts = train[col].astype(str).value_counts()
        artifacts['known_categories'][col] = sorted(counts[counts >= MIN_CAT_COUNT].index)
        if col in FREQ_COLS:
            artifacts['frequencies'][col] = (counts / len(train)).to_dict()

    artifacts['num_medians'] = {c: float(train[c].median()) for c in NUM_COLS}
    artifacts['cat_amount_median'] = train.groupby('cat_id')['amount'].median().to_dict()
    artifacts['global_amount_median'] = float(train['amount'].median())
    return artifacts


def build_features(df, artifacts):
    """Превращает сырые транзакции (формат test.csv) в матрицу признаков модели."""
    df = df.drop(columns=[c for c in PII_COLS + ['target'] if c in df.columns]).copy()
    for col in NUM_COLS + CAT_COLS + ['transaction_time']:
        if col not in df.columns:
            df[col] = np.nan
    out = pd.DataFrame(index=df.index)

    # Числовые признаки с импутацией медианами train
    for col in NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(artifacts['num_medians'][col])

    # Время транзакции
    ts = pd.to_datetime(df['transaction_time'], errors='coerce')
    out['hour'] = ts.dt.hour.fillna(-1).astype(int)
    out['day_of_week'] = ts.dt.dayofweek.fillna(-1).astype(int)
    out['day_of_month'] = ts.dt.day.fillna(-1).astype(int)
    out['month'] = ts.dt.month.fillna(-1).astype(int)
    out['is_night'] = out['hour'].between(0, 5).astype(int)
    out['is_weekend'] = (out['day_of_week'] >= 5).astype(int)

    # Суммы и население
    out['amount_log'] = np.log1p(df['amount'].clip(lower=0))
    out['population_city_log'] = np.log1p(df['population_city'].clip(lower=0))
    cat_median = (df['cat_id'].astype(str)
                  .map(artifacts['cat_amount_median'])
                  .fillna(artifacts['global_amount_median']))
    out['amount_to_cat_median'] = df['amount'] / cat_median.replace(0, np.nan).fillna(1.0)

    # Расстояние клиент — мерчант
    out['distance_km'] = haversine_km(df['lat'], df['lon'], df['merchant_lat'], df['merchant_lon'])

    # Частотное кодирование
    for col in FREQ_COLS:
        out[f'{col}_freq'] = df[col].astype(str).map(artifacts['frequencies'][col]).fillna(0.0)

    # Категориальные признаки для CatBoost: редкие -> __rare__, отсутствующие -> __unknown__
    for col in CAT_COLS:
        known = set(artifacts['known_categories'][col])
        values = df[col].where(df[col].notna(), UNKNOWN).astype(str)
        out[col] = values.map(lambda v: v if v in known or v == UNKNOWN else RARE)

    return out


def save_artifacts(artifacts, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(artifacts, f, ensure_ascii=False, indent=1)


def load_artifacts(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
