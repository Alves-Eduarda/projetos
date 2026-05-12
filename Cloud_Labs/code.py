
import boto3
import random
from PIL import Image, ImageDraw, ImageFont

BUCKET_OUTPUT = ''

#instanciando o serviço do s3
s3 = boto3.client('s3')

def lambda_handler(event, context):

    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']

    # caminhos temporários no lambda
    input_path = f'/tmp/{file_key.split("/")[-1]}'
    output_path = f'/tmp/edited-{file_key.split("/")[-1]}'

    # download da imagem inserida no bucket de entrada
    s3.download_file(bucket_name, file_key, input_path)

    # acessando a imagem
    image = Image.open(input_path)

    lista_textos = ['lab_aws','lab_teste','cloud_computing','cloud_service']

    # escrita do texto de forma aleatória na imagem
    draw = ImageDraw.Draw(image)
    texto = random.choice(lista_textos)
    draw.text((50, 50), texto, fill="yellow")

    # salvando a imagem no caminho temporário de saída
    image.save(output_path)

    # upload da imagem alterada no bucket de saída
    s3.upload_file(
        output_path,
        BUCKET_OUTPUT,
        f'edited-{file_key}'
    )

    return {
    'statusCode': 200,
    'body': 'Imagem processada com sucesso'
    }