import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# Classe para armazenar os dados
class DadosMeteorologicos:
    def __init__(self, data, temperatura_ar):
        self.data = data
        self.temperatura_ar = temperatura_ar


# Função para processar um arquivo CSV e retornar uma lista de objetos DadosMeteorologicos
def process_csv(file_path):
    # Lê o arquivo CSV usando a função read_csv do pandas
    df = pd.read_csv(file_path, delimiter=";", encoding='ISO-8859-1')
    dados_meteorologicos = []  # Lista para armazenar os dados meteorológicos

    # Itera sobre as linhas do dataframe
    for index, row in df.iterrows():
        # Verifica se a coluna 'Data' é nula ou não existe na linha
        if pd.isnull(row['Data']) or 'Data' not in row:
            continue  # Pula para a próxima iteração do loop

        # Verifica se a coluna 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)' é nula
        if pd.isnull(row['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)']):
            continue  # Pula para a próxima iteração do loop

        temperatura_ar = row['TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'].replace(',', '.')

        # Cria um objeto DadosMeteorologicos com os valores da linha atual
        dados = DadosMeteorologicos(
            data=row['Data'],
            temperatura_ar=temperatura_ar,
        )

        # Adiciona o objeto à lista de dados meteorológicos
        dados_meteorologicos.append(dados)

    return dados_meteorologicos  # Retorna a lista de dados meteorológicos


# Função para realizar a previsão dos próximos 5 dias usando regressão linear
def fazer_previsao(dados):
    df = pd.DataFrame({
        'ds': [dado.data for dado in dados],
        'y': [dado.temperatura_ar for dado in dados],
    })

    # Converter a coluna 'ds' para valores numéricos
    df['ds'] = pd.to_datetime(df['ds'])

    # Cria o modelo de regressão linear
    model = LinearRegression()

    # Treina o modelo
    model.fit(df[['ds']], df['y'])

    # Realiza a previsão para os próximos 5 dias
    last_date = df['ds'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq='D')

    # Converte as datas para o formato esperado pelo modelo
    forecast_dates = pd.DataFrame({'ds': forecast_dates})
    forecast_dates['ds'] = forecast_dates['ds'].values.astype(np.float64)

    # Realiza a previsão
    y_pred = model.predict(forecast_dates[['ds']])

    # Cria o dataframe de previsão
    forecast = pd.DataFrame({
        'ds': forecast_dates['ds'],
        'yhat': y_pred
    })

    # Retorna o dataframe de previsão
    return forecast


# Lista de caminhos dos arquivos CSV
caminhos_arquivos_csv = ['INMET_SE_RJ_A625_TRES RIOS_01-01-2023_A_31-03-2023.CSV',
                         'INMET_SE_RJ_A625_TRES RIOS_01-01-2022_A_31-12-2022.CSV',
                         'INMET_SE_RJ_A625_TRES RIOS_01-01-2021_A_31-12-2021.CSV',
                         'INMET_SE_RJ_A625_TRES RIOS_01-01-2020_A_31-12-2020.CSV',
                         'INMET_SE_RJ_A625_TRES RIOS_01-01-2019_A_31-12-2019.CSV']

# Processa os arquivos CSV em paralelo
dados_total = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Mapeia a função process_csv para cada caminho de arquivo CSV e obtém os resultados
    resultados = executor.map(process_csv, caminhos_arquivos_csv)

    # Itera sobre os resultados e adiciona os dados meteorológicos à lista dados_total
    for resultado in resultados:
        dados_total.extend(resultado)

# Realiza a previsão dos próximos 5 dias para o conjunto de dados total
previsao = fazer_previsao(dados_total)

# Exibe as previsões dos próximos 5 dias a partir de 01/04/2023
start_date = datetime(2023, 4, 1)
for index, row in previsao.iterrows():
    date = start_date + timedelta(days=index)
    temperature = round(row['yhat'], 2)
    print(f"Data: {date.strftime('%Y-%m-%d')}, Previsão: {temperature}")
