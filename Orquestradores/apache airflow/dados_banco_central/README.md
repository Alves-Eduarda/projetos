# Projeto: Captura de Dados da API do Banco Central

## 🎯 Objetivo
O projeto tem como finalidade **capturar dados da API pública do Banco Central do Brasil** relacionados a:
- Meios de Pagamentos Mensais
- Estoque e Transações de Cartões

Esses dados serão utilizados para **análises exploratórias e estudos de correlação** entre diferentes perfis de consumidores no Brasil.

---

## 📦 Fluxo de Trabalho
1. **Coleta de dados**  
   - Acessar a API pública do Banco Central.  
   - Obter os dados em formato **JSON**.

2. **Transformação e armazenamento**  
   - Converter os dados para **CSV**.  
   - Salvar os arquivos na pasta `data/`.

3. **Uso futuro**  
   - Explorar correlações entre variáveis.  
   - Investigar padrões de consumo e comportamento financeiro.

---

## ⚙️ Arquitetura Utilizada
- **Docker**: para garantir isolamento e portabilidade do ambiente.  
- **WSL/Ubuntu**: integração com o sistema operacional e suporte a ferramentas Linux.  
- **Apache Airflow**: orquestração das tarefas de ETL (Extração, Transformação e Carga).  

---

