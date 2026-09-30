"""Сервис-писатель: читает результаты скоринга из топика scoring
и складывает их в витрину scores в Postgres."""
import json
import logging
import os
import time

import psycopg
from confluent_kafka import Consumer, KafkaException
from confluent_kafka.admin import AdminClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('score_writer')

KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
SCORING_TOPIC = os.getenv('KAFKA_SCORING_TOPIC', 'scoring')
PG_DSN = 'host={} port={} dbname={} user={} password={}'.format(
    os.getenv('POSTGRES_HOST', 'postgres'),
    os.getenv('POSTGRES_PORT', '5432'),
    os.getenv('POSTGRES_DB', 'fraud'),
    os.getenv('POSTGRES_USER', 'fraud'),
    os.getenv('POSTGRES_PASSWORD', 'fraud'),
)
BATCH_SIZE = 100
FLUSH_INTERVAL_SEC = 1.0

INSERT_SQL = """
    INSERT INTO scores (transaction_id, score, fraud_flag)
    VALUES (%s, %s, %s)
    ON CONFLICT (transaction_id) DO NOTHING
"""


def connect_postgres(retries=30, delay=2):
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg.connect(PG_DSN)
            logger.info('Connected to Postgres')
            return conn
        except psycopg.OperationalError as e:
            logger.info('Postgres is not available yet (attempt %d): %s', attempt, e)
            time.sleep(delay)
    raise RuntimeError('Postgres is not available')


def wait_for_topic(retries=30, delay=2):
    admin = AdminClient({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})
    for attempt in range(1, retries + 1):
        try:
            if SCORING_TOPIC in admin.list_topics(timeout=5).topics:
                logger.info('Topic "%s" found', SCORING_TOPIC)
                return
            logger.info('Waiting for topic "%s" (attempt %d)...', SCORING_TOPIC, attempt)
        except KafkaException as e:
            logger.info('Kafka is not available yet (attempt %d): %s', attempt, e)
        time.sleep(delay)
    raise RuntimeError('Kafka is not available')


def parse(raw_value):
    data = json.loads(raw_value.decode('utf-8'))
    return str(data['transaction_id']), float(data['score']), int(data['fraud_flag'])


def main():
    conn = connect_postgres()
    wait_for_topic()
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': 'score-writer',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
    })
    consumer.subscribe([SCORING_TOPIC])
    logger.info('Consuming from "%s"', SCORING_TOPIC)

    batch, last_flush, total = [], time.monotonic(), 0
    try:
        while True:
            msg = consumer.poll(0.5)
            if msg is not None:
                if msg.error():
                    logger.error('Kafka error: %s', msg.error())
                else:
                    try:
                        batch.append(parse(msg.value()))
                    except Exception:
                        logger.exception('Skipping malformed message at offset %s', msg.offset())

            due = time.monotonic() - last_flush >= FLUSH_INTERVAL_SEC
            if batch and (len(batch) >= BATCH_SIZE or due):
                with conn.cursor() as cur:
                    cur.executemany(INSERT_SQL, batch)
                conn.commit()
                # Оффсеты коммитим только после успешной записи в БД
                consumer.commit(asynchronous=False)
                total += len(batch)
                logger.info('Saved %d rows (total %d)', len(batch), total)
                batch = []
            if due or not batch:
                last_flush = time.monotonic()
    finally:
        consumer.close()
        conn.close()


if __name__ == '__main__':
    main()
