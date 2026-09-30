# Real-Time Fraud Detection System

Сервис обнаружения мошеннических транзакций в реальном времени. Транзакции поступают потоком через Kafka, сервис скоринга применяет собственную модель CatBoost (inference на CPU) и публикует скор и флаг фрода в Kafka. Отдельный сервис сохраняет результаты в Postgres, а UI показывает последние фродовые транзакции и распределение скоров.

Данные: соревнование [teta-ml-1-2025](https://www.kaggle.com/competitions/teta-ml-1-2025).

## 🏗️ Архитектура

```
 interface (Streamlit) ──► [transactions] ──► fraud_detector ──► [scoring] ──► score_writer ──► Postgres (scores)
         ▲                                                                                        │
         └────────────── вкладка «Результаты» (SELECT) ◄──────────────────────────────────────────┘
```

| Сервис | Назначение | Порт на хосте |
|---|---|---|
| `interface` | Streamlit UI: отправка CSV в Kafka и просмотр результатов из Postgres | 8501 |
| `fraud_detector` | Kafka consumer/producer: препроцессинг + скоринг моделью | — |
| `score_writer` | Читает топик `scoring` и пишет результаты в Postgres | — |
| `postgres` | Витрина `scores` (transaction_id, score, fraud_flag, created_at) | 5432 |
| `kafka`, `zookeeper` | Брокер сообщений | 9095 (внешний listener) |
| `kafka-setup` | Создаёт топики `transactions` и `scoring` и завершается | — |
| `kafka-ui` | Веб-интерфейс для просмотра топиков | 8080 |

### Сервис скоринга `fraud_detector`
Этапы разнесены по отдельным скриптам:
- `app/app.py` — чтение сообщений из топика `transactions` и публикация `{transaction_id, score, fraud_flag}` в топик `scoring`;
- `src/preprocessing.py` — препроцессинг (тот же код используется при обучении);
- `src/scorer.py` — скоринг обработанного сообщения моделью CatBoost на CPU.

Сервис делает только inference. Модель и статистики препроцессинга лежат в `fraud_detector/models/`, поэтому обучающие данные контейнеру не нужны.

## 🚀 Быстрый старт

### Требования
- Docker 20.10+ и Docker Compose v2
- Свободные порты 8080, 8501, 9095, 5432, 2181

### Запуск
```bash
git clone <URL этого репозитория>
cd mts25_mlops_hw2_real_time_fraud_detection

docker compose up --build -d
docker compose ps
```
Через 30–60 секунд все сервисы должны быть в статусе `Up`, а `kafka-setup` — `Exited (0)`: он только создаёт топики.

- **Streamlit UI**: http://localhost:8501
- **Kafka UI**: http://localhost:8080

### Проверка работы
1. Откройте http://localhost:8501, вкладка **«📤 Отправка транзакций»**.
2. Загрузите файл `samples/test_sample.csv` (200 транзакций в формате `test.csv`) или `test.csv` из соревнования и нажмите **«Отправить …»**.
3. В Kafka UI (http://localhost:8080 → Topics) убедитесь, что сообщения появились в `transactions` и `scoring`.
4. Проверьте витрину в Postgres:
   ```bash
   docker compose exec postgres psql -U fraud -d fraud \
     -c "SELECT count(*) AS total, sum(fraud_flag) AS frauds FROM scores;"
   ```
5. В UI откройте вкладку **«📊 Результаты»** и нажмите **«Посмотреть результаты»**. Появятся:
   - 10 последних транзакций с `fraud_flag = 1` (если такие есть);
   - гистограмма скоров последних 100 транзакций (или всех, если в базе их меньше).

> `samples/test_sample.csv` — 200 строк из train без колонки `target` (20 из них мошеннические), чтобы в результатах гарантированно были фродовые транзакции.

### Логи и остановка
```bash
docker compose logs -f fraud_detector score_writer
docker compose down        # остановить
docker compose down -v     # остановить и удалить данные Postgres
```

## 📨 Форматы сообщений

Топик `transactions` (отправляет UI):
```json
{"transaction_id": "d6b0f7a0-...", "data": {"transaction_time": "2019-12-27 15:21", "merch": "...", "amount": 148.04, "...": "..."}}
```

Топик `scoring` (отправляет `fraud_detector`):
```json
{"transaction_id": "d6b0f7a0-...", "score": 0.0123, "fraud_flag": 0}
```

Витрина `scores` в Postgres (`postgres/init.sql`):
```sql
transaction_id TEXT PRIMARY KEY, score DOUBLE PRECISION, fraud_flag SMALLINT, created_at TIMESTAMPTZ
```
Повторная доставка одного и того же сообщения не создаёт дублей (`ON CONFLICT DO NOTHING`).

## 🤖 Модель

CatBoostClassifier, обучение и inference только на CPU.

Признаки (`fraud_detector/src/preprocessing.py`):
- время: час, день недели, день месяца, месяц, флаги ночи и выходного;
- `log(amount)`, `log(population_city)`, отношение суммы к медиане суммы по категории `cat_id`;
- расстояние клиент — мерчант (haversine, км);
- частотное кодирование `merch`, `jobs`, `one_city`;
- категориальные признаки `merch`, `cat_id`, `gender`, `us_state`, `jobs`, `one_city` обрабатываются нативно в CatBoost; редкие значения объединяются в `__rare__`;
- персональные данные (`name_1`, `name_2`, `street`, `post_code`) не используются.

Порог `fraud_flag` подобран по максимуму F1 на отложенной выборке (20 %, стратифицированно). Затем модель переобучена на всех данных с найденным числом итераций.

| Метрика (holdout 20 %) | Значение |
|---|---|
| ROC-AUC | 0.997 |
| PR-AUC | 0.891 |
| F1 | 0.836 |
| Порог `fraud_flag` | 0.458 |

Для компактности модели (≈1.8 МБ) используется `max_ctr_complexity=1`: без этого CTR-таблицы по комбинациям категорий раздувают модель до ~220 МБ, что не помещается в GitHub. Метрики и порог также сохранены в `fraud_detector/models/preprocessing_artifacts.json` (`validation_metrics`).

### Переобучение (необязательно — готовая модель уже в репозитории)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r training/requirements.txt
# положите train.csv из соревнования в fraud_detector/train_data/train.csv
python training/train.py
docker compose up --build -d fraud_detector
```

## 📁 Структура проекта
```
.
├── docker-compose.yaml
├── fraud_detector/            # сервис скоринга
│   ├── app/app.py             # Kafka consumer/producer
│   ├── src/preprocessing.py   # препроцессинг
│   ├── src/scorer.py          # скоринг моделью
│   ├── models/                # model.cbm + preprocessing_artifacts.json
│   ├── Dockerfile
│   └── requirements.txt
├── score_writer/              # scoring -> Postgres
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── interface/                 # Streamlit UI
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── postgres/init.sql          # создание витрины scores
├── training/                  # обучение модели (вне контейнеров)
│   ├── train.py
│   └── requirements.txt
└── samples/test_sample.csv    # тестовые транзакции
```

## 🛠️ Troubleshooting
- **В `scoring` нет сообщений** — смотрите `docker compose logs fraud_detector`: сервис ждёт появления топиков и пишет ошибки обработки в лог.
- **Во вкладке «Результаты» пусто** — проверьте `docker compose logs score_writer` и запрос `SELECT count(*) FROM scores`.
- **Порт занят** — освободите порт или поменяйте маппинг `ports` в `docker-compose.yaml`.
- **Нужно начать с чистой базы** — `docker compose down -v && docker compose up --build -d`.
