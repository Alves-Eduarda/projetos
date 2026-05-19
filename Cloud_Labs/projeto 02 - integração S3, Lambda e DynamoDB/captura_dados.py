# repositório que contém o arquivo json que irei utilizar
# https://www.kaggle.com/datasets/indiella/store-sales-json

# importa a biblioteca
import kaggle 

# conecta com a API
kaggle.api.authenticate()

# faz o download do arquivo
kaggle.api.dataset_download_files(
    "indiella/store-sales-json",
    path= "dataset/",
    unzip=True)

