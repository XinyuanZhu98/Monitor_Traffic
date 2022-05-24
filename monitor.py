

from datetime import datetime, timedelta
from airflow import DAG

# Operators
from airflow.operators.bash import BashOperator
# from airflow.operators.python import PythonOperator

AREAS = ["woodlands", "sle", "tpe", "kje", "bke", "cte",
         "pie", "kpe", "aye", "mce", "ecp", "stg"]

default_args = {
    "owner": "xinyuan",
    "depends_on_past": False,
    "start_date": datetime(2022, 5, 22),
    "email": ["example@gmail.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1)
}

dag = DAG(
    dag_id="monitor",
    default_args=default_args,
    schedule_interval="*/3 * * * *",
    catchup=False)

task_collect = BashOperator(
    task_id="collect",
    depends_on_past=False,
    bash_command="python3 ~/airflow/dags/helpers/collect.py",
    dag=dag)

task_detect = BashOperator(
    task_id="detect",
    depends_on_past=False,
    bash_command="python3 ~/airflow/dags/helpers/detect.py",
    dag=dag)

# The execution of task_detect depends on the successful run of task_collect
task_collect >> task_detect
# task_detect.set_upstream(task_collect)
