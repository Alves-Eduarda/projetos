# importando a biblioteca
import pandas as pd

# lendo o arquivo original 
df = pd.read_json("./dataset/StoreSales.json")

# criando a separação do arquivo em três partes
tamanho = len(df) // 3

amostras = [
    df.iloc[:tamanho],
    df.iloc[tamanho:2*tamanho],
    df.iloc[2*tamanho:]
]

#print(amostras)

# salvando cada parte em três amostras json
for i, parte in enumerate(amostras, 1):

    parte.to_json(
        f"./dataset/amostra_{i}.json",
        orient="records",
        indent=4
    )

print("Arquivos criados.")

