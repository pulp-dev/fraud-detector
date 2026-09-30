"""Kafka-сервис скоринга: читает транзакции из топика transactions,
считает скор и флаг фрода и пишет результат в топик scoring."""
import json
import logging
import os
import sys
import time

import pandas as pd
from confluent_kafka import Consumer, KafkaException, Producer
from confluent_kafka.admin import AdminClient

sys.path.append(os.path.abspath('./src'))
from preprocessing import build_features, load_artifacts  # noqa: E402
from scorer import Scorer  # noqa: E402

os.makedirs('/app/logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
TRANSACTIONS_TOPIC = os.getenv('KAFKA_TRANSACTIONS_TOPIC', 'transactions')
SCORING_TOPIC = os.getenv('KAFKA_SCORING_TOPIC', 'scoring')
MODEL_PATH = os.getenv('MODEL_PATH', './models/model.cbm')
ARTIFACTS_PATH = os.getenv('ARTIFACTS_PATH', './models/preprocessing_artifacts.json')


def wait_for_kafka(retries=30, delay=2):
    admin = AdminClient({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})
    for attempt in range(1, retries + 1):
        try:
            topics = admin.list_topics(timeout=5).topics
            if TRANSACTIONS_TOPIC in topics and SCORING_TOPIC in topics:
                logger.info('Kafka is ready, topics found')
                return
            logger.info('Waiting for topics (attempt %d)...', attempt)
        except KafkaException as e:
            logger.info('Kafka is not available yet (attempt %d): %s', attempt, e)
        time.sleep(delay)
    raise RuntimeError('Kafka is not available')


def delivery_report(err, msg):
    if err is not None:
        logger.error('Delivery failed for key %s: %s', msg.key(), err)


class ProcessingService:
    def __init__(self):
        self.artifacts = load_artifacts(ARTIFACTS_PATH)
        self.scorer = Scorer(MODEL_PATH, self.artifacts)

        wait_for_kafka()
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest',
        })
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

    def score_message(self, raw_value):
        data = json.loads(raw_value.decode('utf-8'))
        transaction_id = str(data['transaction_id'])
        features = build_features(pd.DataFrame([data['data']]), self.artifacts)
        scores, flags = self.scorer.predict(features)
        return {'transaction_id': transaction_id, 'score': scores[0], 'fraud_flag': flags[0]}

    def run(self):
        logger.info('Consuming from "%s", producing to "%s"', TRANSACTIONS_TOPIC, SCORING_TOPIC)
        processed = 0
        try:
            while True:
                msg = self.consumer.poll(1.0)
                self.producer.poll(0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error('Kafka error: %s', msg.error())
                    continue
                try:
                    result = self.score_message(msg.value())
                except Exception:
                    logger.exception('Failed to process message at offset %s', msg.offset())
                    continue

                self.producer.produce(
                    SCORING_TOPIC,
                    key=result['transaction_id'],
                    value=json.dumps(result),
                    callback=delivery_report,
                )
                processed += 1
                if processed % 100 == 0:
                    logger.info('Processed %d messages', processed)
        finally:
            self.producer.flush(10)
            self.consumer.close()


if __name__ == '__main__':
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
