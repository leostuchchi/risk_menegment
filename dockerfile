# Используем официальный образ Python как базовый
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей в контейнер
COPY requirements.txt .

# Устанавливаем зависимости проекта
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в рабочую директорию контейнера
COPY . .

# Определяем переменную окружения для Airflow
ENV AIRFLOW_HOME=/app/airflow

# Сначала необходимо инициализировать базу данных Airflow
RUN pip install apache-airflow && airflow db init

# Указываем команду для запуска приложения
CMD ["python", "airflow/dag.py"]




