# importando as bibliotecas
import boto3
import json 

# conexão com o S3 e dynamodb
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# tabela dynamodb
table = dynamodb.Table('store_sales')

def lambda_handler(event, context):

    try:

        # captura informações do evento S3
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']

        print(f"Arquivo recebido: {bucket}/{key}")

        # lê arquivo do S3
        response = s3.get_object(
            Bucket=bucket,
            Key=key
        )

        # conteúdo json
        content = response['Body'].read().decode('utf-8')

        dados = json.loads(content)

        # grava registros no dynamodb
        with table.batch_writer() as batch:

            for item in dados:

                registro = {
                        "Row ID": item.get("Row ID"),
                        "Order ID": item.get("Order ID"),
                        "Order Date": item.get("Order Date"),
                        "Ship Date": item.get("Ship Date"),
                        "Ship Mode": item.get("Ship Mode"),
                        "Customer ID": item.get("Customer ID"),
                        "Customer Name": item.get("Customer Name"),
                        "Segment": item.get("Segment"),
                        "City": item.get("City"),
                        "State": item.get("State"),
                        "Country": item.get("Country"),
                        "Postal Code": item.get("Postal Code"),
                        "Market": item.get("Market"),
                        "Region": item.get("Region"),
                        "Product ID": item.get("Product ID"),
                        "Category": item.get("Category"),
                        "Sub-Category": item.get("Sub-Category"),
                        "Product Name": item.get("Product Name"),
                        "Sales": item.get("Sales"),
                        "Quantity": item.get("Quantity"),
                        "Discount": item.get("Discount"),
                        "Profit": item.get("Profit"),
                        "Shipping Cost": item.get("Shipping Cost"),
                        "Order Priority": item.get("Order Priority")
                    }

                batch.put_item(Item=registro)

        return {
            'statusCode': 200,
            'body': json.dumps('Carga realizada com sucesso!')
        }

    except Exception as e:

        print(f"Erro: {str(e)}")

        return {
            'statusCode': 500,
            'body': json.dumps(str(e))
        }