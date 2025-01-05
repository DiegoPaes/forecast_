import pytest
import pandas as pd
from src import ProcessadorForecast 

def test_converter_coluna_data():
    df = pd.DataFrame({'data': ['2023-01-01', '2023-01-02']})
    processor = ProcessadorForecast(df)
    result = processor.converter_coluna_data('data')
    assert result.dtype == 'datetime64[ns]'

def test_calcular_datas_completas():
    df = pd.DataFrame({'data': pd.to_datetime(['2023-01-01', '2023-01-03'])})
    processor = ProcessadorForecast(df)
    result = processor.calcular_datas_completas('data')
    assert len(result) == 3

def test_verificar_datas_ausentes():
    df = pd.DataFrame({
        'data': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'sku': [1, 2, 1]
    })
    processor = ProcessadorForecast(df)
    result = processor.verificar_datas_ausentes('data', 'sku')
    assert result == 1