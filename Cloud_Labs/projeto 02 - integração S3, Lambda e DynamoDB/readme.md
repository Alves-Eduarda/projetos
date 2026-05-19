# Projeto de integração entre arquivo JSON armazenado no S3 com a inserção dos dados no DynamoDB

objetivo: criar um fluxo automático que a partir da adição dos dados no bucket (S3), o lambda será acionado e realizará a inserção destes dados na tabela no dynamoDB

link do dataset utilizado: https://www.kaggle.com/datasets/indiella/store-sales-json

## Informações sobre o dataset

O arquivo -StoreSales.json- contém o histórico de mais de 50.000 vendas e ordens de pedido de uma loja online. Para o projeto vamos particionar o dado em três arquivos para fazer algumas ingestões diferentes e testar a adição dos dados na tabela.

## Estrutura do fluxo

!\[diagrama](diagrama\_projeto\_02.gif)

## Processo de captura dos dados

Utilização da API fornecida gratuitamente pelo Kaggle

* código: captura\_dados.py

Geração dos arquivos de amostra a partir do arquivo principal

* código: generate\_sample.py



## Processo de adição dos dados na tabela (dynamo DB - store\_sales)

* código: code\_insert\_lambda.py

## Resultado

