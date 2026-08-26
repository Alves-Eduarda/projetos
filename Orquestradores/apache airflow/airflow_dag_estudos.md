# Apache Airflow — Guia de configuração de DAGs

# Apache Airflow — Guia de configuração de DAGs

## 1. O que é uma DAG?

No Apache Airflow, uma **DAG (Directed Acyclic Graph)** representa um fluxo de trabalho composto por tarefas que possuem dependências entre si.

Uma DAG normalmente é criada dentro de um arquivo Python localizado na pasta:

```text
dags/
├── minha_dag.py
├── outra_dag.py
└── projeto/
    └── pipeline.py
```

Um exemplo básico:

```python
from airflow import DAG
from datetime import datetime

with DAG(
    dag_id="minha_primeira_dag",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
) as dag:

    ...
```

A classe `DAG` recebe diversos argumentos que controlam o comportamento do pipeline.

---

# 2. Principais argumentos da DAG

Os argumentos mais importantes para estudar são:

```python
with DAG(
    dag_id="minha_dag",
    description="Descrição da DAG",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["estudo", "engenharia-dados"],
) as dag:
```

---

## 3. `dag_id`

### O que é?

É o identificador único da DAG dentro do Airflow.

```python
dag_id="minha_dag"
```

O `dag_id` aparece na interface web do Airflow.

### Regras importantes

O valor deve ser único.

Evite:

```python
dag_id="DAG 1"
```

Prefira:

```python
dag_id="etl_clientes"
```

Uma boa prática é utilizar nomes que indiquem claramente o objetivo do pipeline:

```python
dag_id="etl_vendas_diarias"
dag_id="processamento_clientes"
dag_id="carga_datawarehouse"
```

---

# 4. `description`

Permite adicionar uma descrição para explicar o objetivo da DAG.

```python
description="Pipeline responsável pela atualização diária das vendas"
```

É útil principalmente quando existem muitas DAGs no ambiente.

Exemplo:

```python
with DAG(
    dag_id="etl_vendas",
    description="Extrai vendas, transforma os dados e carrega no Data Warehouse",
):
```

---

# 5. `start_date`

Define a data a partir da qual o Airflow considera que a DAG pode começar a ser executada.

```python
from datetime import datetime

start_date=datetime(2026, 1, 1)
```

Exemplo:

```python
with DAG(
    dag_id="etl_vendas",
    start_date=datetime(2026, 1, 1),
):
```

### Atenção

`start_date` não significa necessariamente:

> "Execute minha DAG imediatamente nessa data."

Ele funciona em conjunto com o agendamento da DAG.

Por exemplo:

```python
start_date=datetime(2026, 1, 1),
schedule="@daily"
```

significa que o Airflow possui um calendário de execução diária começando a partir dessa referência.

---

# 6. `schedule`

Define a frequência de execução da DAG.

Exemplos:

```python
schedule="@daily"
```

Executa diariamente.

```python
schedule="@hourly"
```

Executa a cada hora.

```python
schedule="@weekly"
```

Executa semanalmente.

Também é possível utilizar expressões cron:

```python
schedule="0 6 * * *"
```

Nesse exemplo, a DAG é programada para executar às 06:00.

Outro exemplo:

```python
schedule="0 0 * * *"
```

Executa diariamente à meia-noite.

### Alguns exemplos de cron

| Expressão   | Significado                |
| ----------- | -------------------------- |
| `0 * * * *` | A cada hora                |
| `0 6 * * *` | Todos os dias às 06:00     |
| `0 0 * * *` | Todos os dias à meia-noite |
| `0 0 * * 1` | Toda segunda-feira         |
| `0 0 1 * *` | Primeiro dia de cada mês   |

> Em versões modernas do Airflow, você também pode encontrar `schedule` sendo configurado com objetos de timetable ou outros objetos de agendamento.

---

# 7. `catchup`

Um dos argumentos mais importantes para entender.

```python
catchup=False
```

Controla se o Airflow deve executar períodos anteriores que estavam previstos no calendário da DAG, mas ainda não foram executados.

Exemplo:

```python
with DAG(
    dag_id="etl_vendas",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
):
```

### `catchup=False`

Indica que você normalmente não quer que o Airflow tente criar execuções para todos os períodos passados desde o `start_date`.

Para muitas DAGs de processamento operacional, é uma configuração comum:

```python
catchup=False
```

---

# 8. `max_active_runs`

Define quantas execuções da mesma DAG podem estar ativas simultaneamente.

```python
max_active_runs=1
```

Por exemplo:

```python
with DAG(
    dag_id="etl_vendas",
    max_active_runs=1,
):
```

Imagine que a execução referente ao dia 10 ainda esteja processando quando chega o momento da execução do dia 11.

Com:

```python
max_active_runs=1
```

o Airflow evita que duas execuções da mesma DAG fiquem ativas simultaneamente.

Isso pode ser importante quando:

* existe risco de concorrência;
* o pipeline altera os mesmos arquivos;
* o pipeline escreve na mesma tabela;
* o processamento é pesado;
* a ordem das execuções é importante.

---

# 9. `tags`

Permite categorizar DAGs na interface do Airflow.

```python
tags=["etl", "producao", "vendas"]
```

Exemplo:

```python
with DAG(
    dag_id="etl_vendas",
    tags=["etl", "datawarehouse", "vendas"],
):
```

Isso facilita encontrar DAGs quando o ambiente possui muitos pipelines.

---

# 10. `default_args`

`default_args` permite definir configurações que serão utilizadas como padrão pelas tarefas da DAG.

Exemplo:

```python
from datetime import timedelta

default_args = {
    "owner": "eduarda",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}
```

Depois:

```python
with DAG(
    dag_id="etl_vendas",
    default_args=default_args,
):
```

As configurações podem ser utilizadas pelas tasks.

---

# 11. `owner`

Define quem é o responsável pela tarefa.

Exemplo:

```python
default_args = {
    "owner": "eduarda",
}
```

É uma informação útil para organização e identificação de responsabilidades.

---

# 12. `retries`

Define quantas vezes uma tarefa deve ser tentada novamente caso falhe.

```python
retries=3
```

Se uma task falhar, o Airflow poderá tentar executá-la novamente.

Exemplo:

```python
default_args = {
    "retries": 5,
}
```

Nesse caso, a tarefa pode ser reexecutada após falhas, conforme as demais configurações de retry.

---

# 13. `retry_delay`

Define quanto tempo o Airflow deve esperar antes de tentar novamente uma tarefa que falhou.

```python
from datetime import timedelta

retry_delay=timedelta(minutes=5)
```

Exemplo:

```python
default_args = {
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}
```

Fluxo:

```text
Task inicia
    ↓
Task falha
    ↓
Espera 5 minutos
    ↓
Retry
    ↓
Task falha novamente?
    ↓
Espera 5 minutos
    ↓
Novo retry
```

---

# 14. `retry_exponential_backoff`

Pode ser utilizado para aumentar progressivamente o tempo entre as tentativas.

```python
retry_exponential_backoff=True
```

Por exemplo:

```python
default_args = {
    "retries": 5,
    "retry_delay": timedelta(minutes=1),
    "retry_exponential_backoff": True,
}
```

Em vez de esperar sempre exatamente o mesmo período, o Airflow pode aumentar o intervalo entre as tentativas.

Isso pode ser útil quando o erro provavelmente é temporário, como:

* API indisponível;
* banco temporariamente indisponível;
* serviço externo congestionado;
* problemas temporários de rede.

---

# 15. `execution_timeout`

Define um tempo máximo permitido para execução de uma tarefa.

Exemplo:

```python
from datetime import timedelta

execution_timeout=timedelta(minutes=30)
```

Se uma tarefa ultrapassar esse limite, ela pode ser considerada como falha por timeout.

Exemplo:

```python
from airflow.operators.bash import BashOperator

task = BashOperator(
    task_id="processar_dados",
    bash_command="python processamento.py",
    execution_timeout=timedelta(minutes=30),
)
```

---

# 16. `email_on_failure`

Pode ser utilizado para configurar notificações de falha por e-mail, dependendo da configuração de e-mail do ambiente Airflow.

Exemplo conceitual:

```python
default_args = {
    "email": ["responsavel@empresa.com"],
    "email_on_failure": True,
}
```

Essa configuração só será efetiva se o ambiente do Airflow estiver configurado para envio de e-mails.

---

# 17. `email_on_retry`

Semelhante ao `email_on_failure`, mas relacionado às tentativas de retry.

```python
email_on_retry=True
```

Exemplo:

```python
default_args = {
    "email": ["responsavel@empresa.com"],
    "email_on_retry": True,
}
```

---

# 18. `depends_on_past`

Controla se uma tarefa depende do sucesso da mesma tarefa em uma execução anterior.

```python
depends_on_past=True
```

Imagine:

```text
Execução 01 → Task A
Execução 02 → Task A
Execução 03 → Task A
```

Com `depends_on_past=True`, uma execução pode depender da execução anterior daquela mesma task.

Isso pode ser útil em determinados pipelines sequenciais, mas deve ser utilizado com cuidado.

---

# 19. `dagrun_timeout`

Define um limite de tempo para uma execução inteira da DAG.

Diferente de:

```python
execution_timeout
```

que está relacionado a uma tarefa específica.

Exemplo:

```python
dagrun_timeout=timedelta(hours=2)
```

A ideia é:

```text
DAG
 ├── Task A
 ├── Task B
 └── Task C

limite total da DAG = 2 horas
```

Enquanto:

```python
execution_timeout=timedelta(minutes=30)
```

seria aplicado individualmente a uma task.

---

# 20. `max_active_tasks`

Controla quantas tarefas daquela DAG podem executar simultaneamente.

Exemplo:

```python
max_active_tasks=3
```

Imagine uma DAG com:

```text
Task A
Task B
Task C
Task D
Task E
```

Se todas puderem executar ao mesmo tempo, o limite define quantas podem ficar ativas simultaneamente.

Isso é importante para controlar consumo de:

* CPU;
* memória;
* conexões;
* APIs;
* banco de dados.

---

# 21. `schedule`, `catchup` e `start_date` trabalhando juntos

Esses três argumentos são fundamentais para compreender o comportamento temporal de uma DAG.

Exemplo:

```python
with DAG(
    dag_id="etl_diario",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
):
```

Podemos interpretar:

```text
start_date
    ↓
Define a referência inicial

schedule
    ↓
Define a frequência

catchup
    ↓
Define o comportamento em relação aos períodos anteriores
```

---

# 22. Argumentos de timezone

Em projetos profissionais, timezone merece atenção especial.

Evite depender de horários implícitos quando o pipeline possui requisitos específicos de horário.

O Airflow trabalha internamente com conceitos de data/hora e timezone, e é importante compreender a diferença entre:

```text
UTC
```

e

```text
America/Recife
```

Por exemplo:

```python
import pendulum

start_date = pendulum.datetime(
    2026,
    1,
    1,
    tz="America/Recife",
)
```

Isso pode ser especialmente importante quando o pipeline possui regras de negócio baseadas no horário local.

---

# 23. `params`

Permite disponibilizar parâmetros para uma DAG.

Exemplo:

```python
with DAG(
    dag_id="etl_clientes",
    params={
        "ambiente": "dev",
        "origem": "postgres",
    },
):
```

Os parâmetros podem ser utilizados para tornar o comportamento do pipeline mais flexível.

---

# 24. `doc_md`

Permite adicionar documentação em Markdown diretamente na DAG.

Exemplo:

```python
with DAG(
    dag_id="etl_clientes",
    doc_md="""
    # ETL de clientes

    Esta DAG realiza:

    1. Extração dos clientes
    2. Transformação dos dados
    3. Carga no Data Warehouse
    """,
):
```

Isso é bastante útil em projetos profissionais porque a própria DAG passa a carregar sua documentação.

---

# 25. Argumentos das Tasks

É importante diferenciar:

```python
DAG
```

de:

```python
Task
```

A DAG define o fluxo geral.

As tasks executam as operações.

Por exemplo:

```python
with DAG(
    dag_id="meu_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    task_1 = BashOperator(
        task_id="executar_script",
        bash_command="python script.py",
    )
```

Aqui temos duas camadas:

```text
DAG
 │
 └── Task
```

A DAG possui seus próprios argumentos:

```python
dag_id
start_date
schedule
catchup
tags
```

Enquanto o `BashOperator` possui argumentos próprios:

```python
task_id
bash_command
execution_timeout
retries
```

---

# 26. `task_id`

Toda task deve possuir um identificador dentro da DAG.

```python
task_id="executar_script"
```

Exemplo:

```python
task = BashOperator(
    task_id="processar_clientes",
    bash_command="python processar.py",
)
```

O `task_id` aparece na interface do Airflow.

Procure utilizar nomes descritivos:

```python
task_id="extract_clientes"
task_id="transform_clientes"
task_id="load_clientes"
```

---

# 27. `BashOperator`

Um operador bastante simples para estudar Airflow é o `BashOperator`.

```python
from airflow.operators.bash import BashOperator
```

Exemplo:

```python
task = BashOperator(
    task_id="executar_script",
    bash_command="echo 'Olá Airflow!'",
)
```

Outro exemplo:

```python
task = BashOperator(
    task_id="listar_arquivos",
    bash_command="ls -la /tmp",
)
```

---

# 28. Definindo dependências

Uma das características mais importantes do Airflow é definir a ordem das tarefas.

Podemos utilizar:

```python
task_1 >> task_2
```

Isso significa:

```text
task_1
   ↓
task_2
```

Exemplo:

```python
extract >> transform >> load
```

Representando:

```text
Extract
   ↓
Transform
   ↓
Load
```

---

# 29. Exemplo completo

Um exemplo de DAG para estudo:

```python
from airflow import DAG

from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta

import pendulum


default_args = {
    "owner": "eduarda",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="etl_clientes",
    description="Pipeline de processamento de clientes",
    start_date=pendulum.datetime(
        2026,
        1,
        1,
        tz="America/Recife",
    ),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    max_active_tasks=3,
    tags=["etl", "clientes", "estudo"],
    default_args=default_args,
) as dag:

    extract = BashOperator(
        task_id="extract_clientes",
        bash_command="echo 'Extraindo dados'",
    )

    transform = BashOperator(
        task_id="transform_clientes",
        bash_command="echo 'Transformando dados'",
    )

    load = BashOperator(
        task_id="load_clientes",
        bash_command="echo 'Carregando dados'",
    )

    extract >> transform >> load
```

---

# 30. Como pensar na configuração de uma DAG

Ao criar uma DAG, pense em cinco grupos principais:

## Identificação

```python
dag_id
description
tags
```

Pergunta:

> O que é essa DAG?

---

## Agendamento

```python
start_date
schedule
catchup
```

Pergunta:

> Quando essa DAG deve executar?

---

## Confiabilidade

```python
retries
retry_delay
retry_exponential_backoff
execution_timeout
dagrun_timeout
```

Pergunta:

> O que deve acontecer quando alguma coisa der errado?

---

## Concorrência

```python
max_active_runs
max_active_tasks
```

Pergunta:

> Quantas coisas podem executar simultaneamente?

---

## Organização

```python
default_args
params
doc_md
tags
```

Pergunta:

> Como tornar o pipeline mais organizado, configurável e documentado?

---

# 31. `default_args` x argumentos individuais

Um ponto importante:

Você pode definir configurações diretamente na task:

```python
task = BashOperator(
    task_id="processar",
    retries=5,
)
```

Ou utilizar `default_args`:

```python
default_args = {
    "retries": 5,
}
```

e:

```python
with DAG(
    dag_id="pipeline",
    default_args=default_args,
):
```

A vantagem é evitar repetição.

Imagine 10 tasks:

```text
Task 1 → retries=3
Task 2 → retries=3
Task 3 → retries=3
Task 4 → retries=3
...
```

Em vez disso:

```python
default_args = {
    "retries": 3,
}
```

Você centraliza a configuração.

---

# 32. Boas práticas

Ao criar DAGs, procure:

### 1. Utilizar nomes descritivos

```python
dag_id="etl_vendas_diarias"
```

em vez de:

```python
dag_id="dag1"
```

### 2. Utilizar `catchup=False` quando não precisar processar períodos anteriores

```python
catchup=False
```

### 3. Definir retries para operações sujeitas a falhas temporárias

```python
retries=3
```

### 4. Definir `retry_delay`

```python
retry_delay=timedelta(minutes=5)
```

### 5. Controlar concorrência

```python
max_active_runs=1
```

quando execuções simultâneas puderem causar problemas.

### 6. Utilizar tags

```python
tags=["etl", "financeiro"]
```

### 7. Documentar pipelines importantes

```python
doc_md="..."
```

### 8. Utilizar timezone explicitamente quando o horário local importar

```python
pendulum.datetime(
    2026,
    1,
    1,
    tz="America/Recife",
)
```

---

# 33. Exemplo de configuração para um projeto de Engenharia de Dados

Uma configuração inicial bastante razoável para estudo seria:

```python
from airflow import DAG

from airflow.operators.bash import BashOperator

from datetime import timedelta

import pendulum


default_args = {
    "owner": "data_engineering",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="pipeline_vendas",
    description="Pipeline de dados de vendas",
    start_date=pendulum.datetime(
        2026,
        1,
        1,
        tz="America/Recife",
    ),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    max_active_tasks=3,
    default_args=default_args,
    tags=["data-engineering", "etl", "vendas"],
) as dag:

    extract = BashOperator(
        task_id="extract",
        bash_command="echo 'Extract'",
    )

    transform = BashOperator(
        task_id="transform",
        bash_command="echo 'Transform'",
    )

    load = BashOperator(
        task_id="load",
        bash_command="echo 'Load'",
    )

    extract >> transform >> load
```

---

# 34. Checklist para estudar uma DAG

Ao analisar um arquivo `.py` dentro da pasta `dags/`, procure responder:

* [ ] Qual é o `dag_id`?
* [ ] Qual é o objetivo da DAG?
* [ ] Qual é o `start_date`?
* [ ] Qual é o `schedule`?
* [ ] `catchup` está habilitado?
* [ ] Qual é o timezone?
* [ ] Existem `default_args`?
* [ ] Quantos `retries` existem?
* [ ] Qual é o `retry_delay`?
* [ ] Existe `execution_timeout`?
* [ ] Existe `dagrun_timeout`?
* [ ] Qual é o `max_active_runs`?
* [ ] Qual é o `max_active_tasks`?
* [ ] Existem `tags`?
* [ ] Existem `params`?
* [ ] A DAG possui documentação?
* [ ] Quais operadores estão sendo utilizados?
* [ ] Quais são os `task_id`?
* [ ] Como as tasks estão conectadas?
* [ ] Existem dependências entre tasks?
* [ ] O pipeline pode executar tarefas simultaneamente?
* [ ] O que acontece quando uma task falha?

---

# 35. Resumo mental

Uma forma simples de memorizar a configuração de uma DAG é:

```text
                    DAG
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   IDENTIDADE    AGENDAMENTO   EXECUÇÃO
        │            │            │
     dag_id       schedule      retries
   description   start_date    retry_delay
      tags         catchup     timeout
        │
        └────────────┬────────────┘
                     ↓
                CONCORRÊNCIA
                     │
              max_active_runs
              max_active_tasks
                     │
                     ↓
                   TASKS
                     │
              ┌──────┴──────┐
              ↓             ↓
          Operator      Operator
              │             │
          task_id       task_id
              │             │
              └──────┬──────┘
                     ↓
                DEPENDÊNCIAS
```

A ideia principal é:

> **A DAG define quando, como e sob quais regras o pipeline deve funcionar. As Tasks definem o que efetivamente será executado.**

Para estudar Airflow como Engenharia de Dados, não é necessário memorizar todos os argumentos de uma vez. O mais importante inicialmente é dominar:

```text
DAG
├── dag_id
├── start_date
├── schedule
├── catchup
├── default_args
├── retries
├── retry_delay
├── max_active_runs
├── max_active_tasks
└── tags

TASK
├── task_id
├── operator
├── retries
├── retry_delay
├── execution_timeout
└── dependencies
```

Depois disso, vale avançar para **XCom, Variables, Connections, Sensors, TaskFlow API, pools, branching, trigger rules e parametrização**, que são conceitos fundamentais para construir pipelines Airflow mais próximos de ambientes profissionais.


...
