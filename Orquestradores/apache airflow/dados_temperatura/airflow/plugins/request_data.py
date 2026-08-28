#importando as bibliotecas
import pandas as pd
import requests

# configuracao da API

# Obtendo as coordenadas da cidade desejada
def obter_coordenadas():
    
    url = "https://api.openweathermap.org/geo/1.0/direct"

    params = {
        "q": "Recife, BR",
        "limit": 1,
        "appid" : "API_KEY"
    }

    response = requests.get(url, params=params,timeout=10)

    response.raise_for_status()

    dados = response.json()

    return dados[0]["lat"], dados[0]["lon"]

# Obtendo os dados do tempo

def obter_consulta_tempo():

    lat,long = obter_coordenadas()

    params = {
        "appid" : "API_KEY",
        "lon" : lat,
        "lat" : long,
        "units" : "metric",
        "lang" : "pt_br"
    }

    URL = f"https://api.openweathermap.org/data/2.5/forecast"

    # realizando o request

    response = requests.get(URL,params=params,timeout=10)
    
    response.raise_for_status()

    dados = response.json()

    return dados

dados_tempo = obter_consulta_tempo()

#capturando os dados
wt_data = dados_tempo.get('list')

# extraindo os valores 
registros = []

for item in wt_data:

    registro = {
        "data_hora": item["dt_txt"],
        "temperatura": item["main"]["temp"],
        "sensacao_termica": item["main"]["feels_like"],
        "temperatura_min": item["main"]["temp_min"],
        "temperatura_max": item["main"]["temp_max"],
        "pressao": item["main"]["pressure"],
        "umidade": item["main"]["humidity"],
        "descricao": item["weather"][0]["description"],
        "condicao": item["weather"][0]["main"],
        "nuvens": item["clouds"]["all"],
        "vento_velocidade": item["wind"]["speed"],
        "vento_direcao": item["wind"]["deg"],
        "visibilidade": item["visibility"],
        "probabilidade_chuva": item["pop"],
        "chuva_3h": item.get("rain", {}).get("3h", 0)
    }

    registros.append(registro)

# criando o dataframe
df = pd.DataFrame(registros)

# salvando o arquivo na pasta de dados (data)
df.to_csv(
    "/opt/airflow/data/previsao_recife.csv",
    index=False
)



