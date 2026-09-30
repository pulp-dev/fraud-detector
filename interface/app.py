import streamlit as st
import pandas as pd
import altair as alt
import psycopg
from kafka import KafkaProducer
import json
import time
import os
import uuid

# Конфигурация Kafka
KAFKA_CONFIG = {
    "bootstrap_servers": os.getenv("KAFKA_BROKERS", "kafka:9092"),
    "topic": os.getenv("KAFKA_TOPIC", "transactions")
}

# Конфигурация Postgres с результатами скоринга
PG_DSN = "host={} port={} dbname={} user={} password={}".format(
    os.getenv("POSTGRES_HOST", "postgres"),
    os.getenv("POSTGRES_PORT", "5432"),
    os.getenv("POSTGRES_DB", "fraud"),
    os.getenv("POSTGRES_USER", "fraud"),
    os.getenv("POSTGRES_PASSWORD", "fraud"),
)

LAST_FRAUD_SQL = """
    SELECT transaction_id, score, fraud_flag, created_at
    FROM scores
    WHERE fraud_flag = 1
    ORDER BY created_at DESC
    LIMIT 10
"""
LAST_SCORES_SQL = "SELECT score FROM scores ORDER BY created_at DESC LIMIT 100"

def load_file(uploaded_file):
    """Загрузка CSV файла в DataFrame"""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {str(e)}")
        return None

def send_to_kafka(df, topic, bootstrap_servers):
    """Отправка данных в Kafka с уникальным ID транзакции"""
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            security_protocol="PLAINTEXT"
        )
        
        # Генерация уникальных ID для всех транзакций
        df['transaction_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        progress_bar = st.progress(0)
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            # Отправляем данные вместе с ID
            producer.send(
                topic, 
                value={
                    "transaction_id": row['transaction_id'],
                    "data": row.drop('transaction_id').to_dict()
                }
            )
            progress_bar.progress((idx + 1) / total_rows)
            time.sleep(0.01)
            
        producer.flush()
     
        return True
    except Exception as e:
        st.error(f"Ошибка отправки данных: {str(e)}")
        return False

def query_df(sql):
    """Выполнение SELECT-запроса к Postgres и возврат результата в DataFrame"""
    with psycopg.connect(PG_DSN, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [c.name for c in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=columns)


def show_results():
    """Раздел с результатами скоринга из Postgres"""
    try:
        fraud_df = query_df(LAST_FRAUD_SQL)
        scores_df = query_df(LAST_SCORES_SQL)
    except Exception as e:
        st.error(f"Ошибка подключения к базе: {str(e)}")
        return

    st.subheader("🚨 10 последних фродовых транзакций")
    if fraud_df.empty:
        st.info("Фродовых транзакций пока нет")
    else:
        st.dataframe(fraud_df, use_container_width=True, hide_index=True)

    st.subheader("📈 Распределение скоров")
    if scores_df.empty:
        st.info("В базе пока нет результатов скоринга")
        return
    st.caption(f"Последние {len(scores_df)} транзакций")
    chart = alt.Chart(scores_df).mark_bar().encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=20, extent=[0, 1]), title="Скор модели"),
        y=alt.Y("count():Q", title="Количество транзакций"),
    )
    st.altair_chart(chart, use_container_width=True)


# Инициализация состояния
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

# Интерфейс
st.title("🛡️ Real-Time Fraud Detection")
upload_tab, results_tab = st.tabs(["📤 Отправка транзакций", "📊 Результаты"])

with upload_tab:

    # Блок загрузки файлов
    uploaded_file = st.file_uploader(
        "Загрузите CSV файл с транзакциями",
        type=["csv"]
    )

    if uploaded_file and uploaded_file.name not in st.session_state.uploaded_files:
        # Добавляем файл в состояние
        st.session_state.uploaded_files[uploaded_file.name] = {
            "status": "Загружен",
            "df": load_file(uploaded_file)
        }
        st.success(f"Файл {uploaded_file.name} успешно загружен!")

    # Список загруженных файлов
    if st.session_state.uploaded_files:
        st.subheader("🗂 Список загруженных файлов")
    
        for file_name, file_data in st.session_state.uploaded_files.items():
            cols = st.columns([4, 2, 2])
        
            with cols[0]:
                st.markdown(f"**Файл:** `{file_name}`")
                st.markdown(f"**Статус:** `{file_data['status']}`")
        
            with cols[2]:
                if st.button(f"Отправить {file_name}", key=f"send_{file_name}"):
                    if file_data["df"] is not None:
                        with st.spinner("Отправка..."):
                            success = send_to_kafka(
                                file_data["df"],
                                KAFKA_CONFIG["topic"],
                                KAFKA_CONFIG["bootstrap_servers"]
                            )
                            if success:
                                st.session_state.uploaded_files[file_name]["status"] = "Отправлен"
                                st.rerun()
                    else:
                        st.error("Файл не содержит данных")

with results_tab:
    if st.button("Посмотреть результаты"):
        show_results()
