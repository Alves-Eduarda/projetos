# importando as bibliotecas 
from datetime import timedelta,datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from scr.request_data_bc import *

# definindo o default da dag

default_args = {
    "owner": "Eduarda",
    "retries": 2,
    "retry_delay": timedelta(minutes=2)
}

with DAG(
    dag_id = "captura_dados_bc",
    description="Realiza a atualização dos dados da API pública do banco central do Brasil",
    start_date = datetime(2026,9,1),
    schedule='@daily',
    catchup=False,
    default_args=default_args

) as dag:

    t1 = PythonOperator(
        task_id = "export_trim",
        python_callable = export_file,
        op_args = ["trim"]
    )

    t2 = PythonOperator(
        task_id = "export_month",
        python_callable = export_file,
        op_args = ["month"]
    )

    [t1, t2]

