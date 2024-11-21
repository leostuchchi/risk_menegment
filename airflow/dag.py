
import datetime as dt
#import datetime as datetime
import os
import sys
#import pandas as pd
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Определяем путь к проекту
path = os.path.join(os.path.expanduser('~'), 'scoring_credit_default')

# Устанавливаем переменную окружения
os.environ['PROJECT_PATH'] = path

# Добавим путь к коду проекта в $PATH
sys.path.insert(0, path)


from modules.main import pipeline
from modules.main import predictor



args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}




with DAG(
        dag_id='scoring_credit_default',
        schedule_interval="00 15 * * *",
        default_args=args,
) as dag:

    # BashOperator, выполняющий указанную bash-команду
    first_task = BashOperator(
        task_id='first_task',
        bash_command='echo "Here we start!"',
        dag=dag,
    )


    # запуск моделирования (preparations запускается из pipeline)
    pipeline = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    # запуск predictions
    predict = PythonOperator(
        task_id='predictions',
        python_callable=predictor,
    )

    # BashOperator, выполняющий указанную bash-команду
    last_task = BashOperator(
        task_id='finished',
        bash_command='echo "Congratulations! predictions writen to predictions!"',
        dag=dag,
    )


    first_task >> pipeline >> predict >> last_task
