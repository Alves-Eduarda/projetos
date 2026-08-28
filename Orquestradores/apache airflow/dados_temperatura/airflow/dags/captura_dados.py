from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from call_data.request_data import *

default_args = {
    "onwer": "Eduarda",
    "retries": 2,
    "retry_delay": timedelta(minutes=2)
}


with DAG (
    dag_id= "captura_dados_tempo",
    description= "realiza a captura dos dados de previsão referentes ao tempo na cidade do Recife a cada 3 horas",
    start_date = datetime(2026,8,27),
    schedule='@daily',
    catchup=False
) as dag:
    t1 = PythonOperator(
        task_id="captura_dados_api",
        python_callable= obter_consulta_tempo
    )

t1
