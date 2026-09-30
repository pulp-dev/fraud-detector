"""Скоринг обработанных транзакций моделью CatBoost (inference на CPU)."""
import logging

from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self, model_path, artifacts):
        logger.info('Loading model from %s', model_path)
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.feature_names = artifacts['feature_names']
        self.threshold = float(artifacts['threshold'])
        logger.info('Model loaded, fraud threshold = %.4f', self.threshold)

    def predict(self, features):
        """Возвращает списки скоров и флагов фрода для каждой строки features."""
        scores = self.model.predict_proba(features[self.feature_names])[:, 1]
        flags = (scores >= self.threshold).astype(int)
        return scores.tolist(), flags.tolist()
