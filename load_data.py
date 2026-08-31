"""
Módulo para cargar y limpiar datos.

Funciones:
    load_raw_data: Carga datos desde el directorio raw.
    clean_data: Realiza limpieza básica del dataset.
    save_processed_data: Guarda datos procesados.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(filename: str, data_dir: str = "../data/raw") -> pd.DataFrame:
    """
    Carga un archivo CSV desde el directorio de datos raw.

    Args:
        filename: Nombre del archivo CSV.
        data_dir: Ruta al directorio de datos raw.

    Returns:
        DataFrame con los datos cargados.
    """
    filepath = Path(data_dir) / filename
    df = pd.read_csv(filepath)
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpieza básica del dataset.

    Pasos:
        1. Eliminar duplicados
        2. Manejar valores faltantes
        3. Corregir tipos de datos
        4. Eliminar columnas irrelevantes

    Args:
        df: DataFrame original.

    Returns:
        DataFrame limpio.
    """
    df_clean = df.copy()

    # Eliminar duplicados
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Duplicados eliminados: {initial_rows - len(df_clean)}")

    # Manejar valores faltantes (ejemplo: eliminar filas con >50% faltantes)
    threshold = len(df_clean.columns) * 0.5
    df_clean = df_clean.dropna(thresh=threshold)

    # Eliminar columnas con >80% de valores faltantes
    missing_pct = df_clean.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.8].index.tolist()
    df_clean = df_clean.drop(columns=cols_to_drop)
    print(f"Columnas eliminadas por alta tasa de faltantes: {cols_to_drop}")

    return df_clean


def save_processed_data(df: pd.DataFrame, filename: str, data_dir: str = "../data/processed") -> None:
    """
    Guarda el DataFrame procesado como CSV.

    Args:
        df: DataFrame a guardar.
        filename: Nombre del archivo de salida.
        data_dir: Directorio de destino.
    """
    output_path = Path(data_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Datos guardados en: {output_path}")


if __name__ == "__main__":
    # Ejemplo de uso
    # df = load_raw_data("dataset.csv")
    # df_clean = clean_data(df)
    # save_processed_data(df_clean, "dataset_clean.csv")
    pass
