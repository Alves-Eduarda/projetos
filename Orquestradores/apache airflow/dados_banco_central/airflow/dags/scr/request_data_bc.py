#importando as bibliotecas
import requests
import pandas as pd

def get_data(scr):

    if scr == "trim":

        URL = "https://olinda.bcb.gov.br/olinda/servico/MPV_DadosAbertos/versao/v1/odata/Quantidadeetransacoesdecartoes(trimestre=@trimestre)?@trimestre='20261'&$top=10000&$format=json&$select=trimestre,nomeBandeira,nomeFuncao,produto,modalidade,qtdCartoesEmitidos,qtdCartoesAtivos,qtdTransacoesNacionais,valorTransacoesNacionais,qtdTransacoesInternacionais,valorTransacoesInternacionais"

    else:
        URL = "https://olinda.bcb.gov.br/olinda/servico/MPV_DadosAbertos/versao/v1/odata/MeiosdePagamentosMensalDA(AnoMes=@AnoMes)?@AnoMes='202601'&$top=10000&$format=json&$select=AnoMes,quantidadePix,valorPix,quantidadeTED,valorTED,quantidadeTEC,valorTEC,quantidadeCheque,valorCheque,quantidadeBoleto,valorBoleto,quantidadeDOC,valorDOC"
        
    response = requests.post(URL)
    
    response.raise_for_status()

    dados = response.json()

    return dados

def transform_data(dados):

    if dados == "trim":
        dados_coleta = get_data(dados)
    else: 
        dados_coleta = get_data(dados)

    df = pd.DataFrame(dados_coleta['value'])

    return df

def export_file(scr):

    data = transform_data(dados=scr)

    file = f"tabela_{scr}.csv"

    data.to_csv(f"/opt/airflow/data/{file}")

    print("dados salvos com sucesso")
