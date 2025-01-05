import os
import sys
sys.path.append('/home/diegopaes/ml_projects/forecast/src')
from exception import CustomException
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('data/raw', 'data.csv')

class DataFrameSplitter:
    def __init__(self, dataframe, percentual=0.2):
        '''Inicializa o separador de DataFrame.

        Args:
            dataframe: O DataFrame do pandas a ser dividido
            percentual: O percentual das *últimas* linhas a serem separadas (padrão:0.2)        
        '''
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("O argumento 'dataframe' deve ser um DataFrame do pandas")
        if not 0 <= percentual <=1:
            raise ValueError('O percentual deve estar entre 0 e 1.')
        
        self.dataframe = dataframe
        self.percentual = percentual
        self._calcular_indices()
    
    def _calcular_indices(self):
        '''Calcula os indíces para a divisão.'''
        num_linhas = len(self.dataframe)
        self.num_linhas_percentual = int(num_linhas * self.percentual)
        self.indice_inicial = num_linhas - self.num_linhas_percentual

    def split(self):
        """Divide o DataFrame em duas partes: as primeiras (1 - percentual) linhas e as últimas percentual linhas
        
        Returns:
            Um dicionário contendo dois DataFrames: 'primeiras' e 'ultimas'
            Retorna None se o DataFrame for vazio
        """
        if self.dataframe.empty:
            return None
        
        primeiras_linhas = self.dataframe.head(len(self.dataframe) - self.num_linhas_percentual).copy()
        ultimas_linhas = self.dataframe.tail(self.num_linhas_percentual).copy()

        return {'primeiras': primeiras_linhas, 'ultimas': ultimas_linhas}

class DataIngestion:
    def __init__(self):
        self.config_ingestao = DataIngestionConfig()

    def inicializar_dados_ingestao(self):
        logging.info('Método ou ingestão de dados')
        try:
            df=pd.read_csv('/home/diegopaes/ml_projects/forecast/data/raw/vendas.csv')
            logging.info('Ler o dataset com um DataFrame')

            os.makedirs(os.path.dirname(self.config_ingestao.train_data_path), exist_ok=True)

            df.to_csv(self.config_ingestao.raw_data_path, index=False, header=True)

            logging.info('Divisão de treino e teste iniciada')
            
            splitter = DataFrameSplitter(df)
            split_data = splitter.split()
            if split_data:
                train_set = split_data['primeiras']
                test_set = split_data['ultimas']

                train_set.to_csv(self.config_ingestao.train_data_path, index=False, header=True)
                test_set.to_csv(self.config_ingestao.test_data_path, index=False, header=True)
                logging.info("Dados de treino e teste salvos.")
            else:
                logging.warning("DataFrame vazio. Divisão não realizada.")

            logging.info('Ingestão dos dados completa')

            return(
                self.config_ingestao.train_data_path,
                self.config_ingestao.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)