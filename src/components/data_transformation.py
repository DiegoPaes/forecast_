import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from exception import CustomException
from logger import logging
from typing import Optional, Set
import holidays
from utils import save_object

@dataclass
class ConfigTransformacaoDados:
    preprocessor_obj_file_path=os.pathe.join('artifacts', 'preprocessor.pkl')
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    process_data_path: str=os.path.join('data/processed', 'processed.csv')
    final_data_path: str=os.path.join('data/final', 'final.csv')

class ProcessadorForecast:
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa a classe com o DataFrame de Forecast.

        Args:
            df (pd.DataFrame): DataFrame contendo as colunas de data e a outra coluna que se completa com a data.
        """
        self.df = df
        self.feriados = holidays.Brazil()

    def converter_coluna_data(self, coluna: str, formato: str = '%Y-%m-%d') -> pd.Series:
        """
        Converte uma coluna para o tipo datetime, permitindo formatação personalizada.

        Args:
            coluna (str): Nome da coluna a ser convertida.
            formato (str): Formato da data na coluna (default: '%Y-%m-%d').

        Returns:
            pd.Series: Série com as datas convertidas.

        Raises:
            CustomException: Se a coluna não existir no DataFrame.
        """
        try:
            if coluna not in self.df.columns:
                raise ValueError(f"A coluna '{coluna}' não existe no DataFrame.")
            return pd.to_datetime(self.df[coluna], format=formato, errors='coerce')
        except Exception as e:
            raise CustomException(e, sys)

    def verificar_datas_ausentes(self, coluna_data: str, coluna_aux: str) -> int:
        """
        Retorna a quantidade total de combinações de datas e valores únicos ausentes.

        Args:
            coluna_data (str): Nome da coluna contendo as datas a serem analisadas.
            coluna_aux (str): Nome da coluna contendo o objeto da análise (Exemplo: SKU, produto, etc.).

        Returns:
            int: Diferença entre o total esperado e o total real de registros.

        Raises:
            ValueError: Se as colunas não existirem no DataFrame.
        """
        try:
            # Verifica se as colunas existem no DataFrame
            if coluna_data not in self.df.columns or coluna_aux not in self.df.columns:
                raise ValueError(f"Uma das colunas '{coluna_data}' ou '{coluna_aux}' não existe no DataFrame.")

            # Calcula o número de combinações esperadas
            datas_unicas = self.df[coluna_data].nunique()
            valores_unicos = self.df[coluna_aux].nunique()
            total_esperado = datas_unicas * valores_unicos

            # Retorna a diferença entre o total esperado e o total real
            return total_esperado - self.df.shape[0]
        except Exception as e:
            raise CustomException(e, sys)

    def completar_vendas(self, coluna_data: str, coluna_aux: str, lista_colunas: list[str]) -> pd.DataFrame:
        """
        Completa as informações faltantes com todas as combinações de data e uma segunda coluna,
        preenchendo com 0 onde não há informação.

        Args:
            coluna_data (str): Nome da coluna contendo as datas a serem analisadas.
            coluna_aux (str): Nome da coluna contendo o objeto da análise (Exemplo: SKU, produto, etc.).
            lista_colunas (List[str]): Lista com os nomes das colunas do DataFrame resultante.

        Returns:
            pd.DataFrame: DataFrame completo com todas as combinações de datas e valores únicos.

        Raises:
            ValueError: Se as colunas não existirem no DataFrame.
        """
        try:
            # Verifica se as colunas existem no DataFrame
            if coluna_data not in self.df.columns or coluna_aux not in self.df.columns:
                raise ValueError(f"As colunas '{coluna_data}' ou '{coluna_aux}' não existem no DataFrame.")

            # Verifica se há datas ausentes
            if self.verificar_datas_ausentes(coluna_data, coluna_aux) != 0:
                # Cria um MultiIndex com todas as combinações de "coluna_data" e "coluna_aux"
                completar_indice = pd.MultiIndex.from_product(
                    [self.df[coluna_data].unique(), self.df[coluna_aux].unique()],
                    names=[coluna_data, coluna_aux]
                )

                # Reindexa o DataFrame para incluir todas as combinações, preenchendo com 0
                todas_vendas = (
                    self.df.set_index([coluna_data, coluna_aux])
                    .reindex(completar_indice, fill_value=0)
                    .reset_index()
                )

                # Renomeia as colunas conforme a lista fornecida
                todas_vendas.columns = lista_colunas

                return todas_vendas
            else:
                # Se não houver datas ausentes, retorna o DataFrame original
                return self.df
        except Exception as e:
            raise CustomException(e, sys)

    def criar_coluna_dia_semana(self, coluna_data: str) -> pd.DataFrame:
        """
        Retorna uma série de dias da semana com base em uma coluna de data.

        Args:
            coluna (str): Nome da coluna contendo as datas a serem analisadas.

        Returns:
            DataFrame: DataFrame com dias da semana.
        """
        try:
            if coluna_data not in self.df.columns:
                raise ValueError(f"A coluna '{coluna_data}' não existe no DataFrame.")
            
            self.df['dia_semana'] = self.df[coluna_data].dt.day_name()

            return self.df
        except Exception as e:
            raise CustomException(e, sys)
        
    def criar_coluna_mes(self, coluna_data: str) -> pd.DataFrame:
        """
        Retorna uma série de meses com base em uma coluna de data.

        Args:
            coluna (str): Nome da coluna contendo as datas a serem analisadas.

        Returns:
            Series: Série com os meses.
        """
        try:
            if coluna_data not in self.df.columns:
                raise ValueError(f"A coluna '{coluna_data}' não existe no DataFrame.")
            
            self.df['mes'] = self.df[coluna_data].dt.month_name()
            
            return self.df
        except Exception as e:
            raise CustomException(e, sys)
        
    def verificar_feriado(self, data):
        """
        Verifica se uma data é feriado.

        Args:
            data: Data a ser verificada.

        Returns:
            str: "feriado" se for feriado, "dia_normal" caso contrário.
        """
        return "feriado" if data in self.feriados else "dia_normal"
    
    def adicionar_coluna_feriados(self, coluna_data: str) -> pd.DataFrame:
        """
        Adiciona uma coluna ao DataFrame indicando se cada data é feriado ou dia normal.

        Args:
            df (pd.DataFrame): DataFrame contendo a coluna de datas.
            coluna_data (str): Nome da coluna de datas no DataFrame.

        Returns:
            pd.DataFrame: DataFrame com a nova coluna 'tipo_dia'.
        """
        try:
            if coluna_data not in self.df.columns:
                raise ValueError(f"A coluna '{coluna_data}' não existe no DataFrame.")

            # Aplica a função de verificação de feriados à coluna de datas
            self.df['tipo_dia'] = self.df[coluna_data].apply(self.verificar_feriado)
            return self.df
        except Exception as e:
            logging.error(f'Erro ao criar a coluna: {e}')
            raise CustomException(e, sys)
        
    def criar_colunas_df(self, coluna_data: str, coluna_aux: str, lista_colunas: list[str]) -> pd.DataFrame:
        """
        Cria um DataFrame com todas as colunas necessárias.

        Args:
            coluna_data (str): Nome da coluna contendo as datas a serem analisadas.
            coluna_aux (str): Nome da coluna contendo o objeto da análise (Exemplo: SKU, produto, etc.).
            lista_colunas (List[str]): Lista com os nomes das colunas do DataFrame resultante.

        Returns:
            pd.DataFrame: DataFrame com as colunas 'dia_semana', 'mes', 'tipo_dia'.
        """
        try:
            if coluna_data not in self.df.columns or coluna_aux not in self.df.columns:
                raise ValueError(f"Uma das colunas '{coluna_data}' ou '{coluna_aux}' não existe no DataFrame.")
            
            self.df = self.completar_vendas(coluna_data, coluna_aux, ['data_venda', 'item', 'data_venda'])
            self.df = self.criar_coluna_dia_semana(coluna_data)
            self.df = self.criar_coluna_mes(coluna_data)
            self.df = self.adicionar_coluna_feriados(coluna_data)
            return self.df
        except Exception as e:
            raise CustomException(e, sys)
        
class DataFrameSplitter:
    def __init__(self, dataframe: pd.DataFrame, percentual=0.2):
        '''Inicializa o separador de DataFrame.

        Args:
            dataframe: O DataFrame do pandas a ser dividido
            percentual: O percentual das *últimas* linhas a serem separadas (padrão:0.2)        
        '''
        try:
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("O argumento 'dataframe' deve ser um DataFrame do pandas")
            if not 0 <= percentual <=1:
                raise ValueError('O percentual deve estar entre 0 e 1.')
        
            self.dataframe = dataframe
            self.percentual = percentual
            self._calcular_indices()
        except Exception as e:
            raise CustomException(e, sys)
    
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
        try:
            if self.dataframe.empty:
                return None
        
            primeiras_linhas = self.dataframe.head(len(self.dataframe) - self.num_linhas_percentual).copy()
            ultimas_linhas = self.dataframe.tail(self.num_linhas_percentual).copy()

            return {'primeiras': primeiras_linhas, 'ultimas': ultimas_linhas}
        except Exception as e:
            raise CustomException(e, sys)

class TransformacaoDados:
    def __init__(self):
        self.config_transformacao_dados = ConfigTransformacaoDados()

    def iniciar_transformacao_dados(self):
        logging.info('Tratamento de dados iniciado')
        try:
            df = pd.read_csv('/home/diegopaes/ml_projects/forecast/data/raw/vendas.csv')
            logging.info('Ler o dataset com um DataFrame')

            logging.info('Pré-tratamento dos dados')
            preprocessed_df = ProcessadorForecast(df)
            df['data_venda'] = preprocessed_df.converter_coluna_data('data_venda')

            os.makedirs(os.path.dirname(self.config_ingestao.train_data_path), exist_ok=True)

            df.to_csv(self.config_transformacao_dados.process_data_path, index=False, header=True)

            logging.info('Pré-tratamento finalizado')

            logging.info('Divisão de treino e teste iniciada')
            
            splitter = DataFrameSplitter(df)
            split_data = splitter.split()
            if split_data:
                train_set = split_data['primeiras']
                test_set = split_data['ultimas']

                train_set.to_csv(self.config_transformacao_dados.train_data_path, index=False, header=True)
                test_set.to_csv(self.config_transformacao_dados.test_data_path, index=False, header=True)
                logging.info("Dados de treino e teste salvos.")
            else:
                logging.warning("DataFrame vazio. Divisão não realizada.")

            logging.info('Feature Engineering iniciado')

            df_train = pd.read_csv('/home/diegopaes/ml_projects/forecast/artifacts/train.csv')

            pre_processamento = ProcessadorForecast(df_train)
            df_feature_eng = pre_processamento.criar_colunas_df()
            
            df_feature_eng.to_csv(self.config_transformacao_dados.final_data_path, index=False, header=True)

            logging.info('Feature Engineering concluída')

            return(
                self.config_transformacao_dados.process_data_path,
                self.config_transformacao_dados.train_data_path,
                self.config_transformacao_dados.test_data_path,
                self.config_transformacao_dados.final_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)