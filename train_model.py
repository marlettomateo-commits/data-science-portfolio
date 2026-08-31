"""
Módulo para entrenamiento y evaluación de modelos de Machine Learning.

Clases:
    ModelTrainer: Gestiona el entrenamiento, evaluación y guardado de modelos.
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


class ModelTrainer:
    """
    Clase para entrenar y evaluar modelos de ML.

    Attributes:
        model: Instancia del modelo de sklearn.
        model_name: Nombre identificador del modelo.
        metrics: Diccionario con métricas de evaluación.
    """

    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.metrics = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'ModelTrainer':
        """
        Entrena el modelo con los datos proporcionados.

        Args:
            X_train: Features de entrenamiento.
            y_train: Target de entrenamiento.

        Returns:
            self para encadenamiento de métodos.
        """
        print(f"Entrenando {self.model_name}...")
        self.model.fit(X_train, y_train)
        print(f"✅ {self.model_name} entrenado exitosamente.")
        return self

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 task_type: str = 'classification') -> Dict[str, float]:
        """
        Evalúa el modelo en el conjunto de prueba.

        Args:
            X_test: Features de prueba.
            y_test: Target de prueba.
            task_type: 'classification' o 'regression'.

        Returns:
            Diccionario con métricas de rendimiento.
        """
        y_pred = self.model.predict(X_test)

        if task_type == 'classification':
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        else:
            self.metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }

        print(f"📊 Métricas de {self.model_name}:")
        for metric, value in self.metrics.items():
            print(f"   {metric}: {value:.4f}")

        return self.metrics

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5, scoring: str = 'accuracy') -> np.ndarray:
        """
        Realiza validación cruzada.

        Args:
            X: Features.
            y: Target.
            cv: Número de folds.
            scoring: Métrica de evaluación.

        Returns:
            Array con scores de cada fold.
        """
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        print(f"📈 CV Scores ({cv}-fold): {scores}")
        print(f"   Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return scores

    def save_model(self, filepath: str) -> None:
        """
        Guarda el modelo entrenado.

        Args:
            filepath: Ruta donde guardar el modelo.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"💾 Modelo guardado en: {filepath}")

    def save_metrics(self, filepath: str) -> None:
        """
        Guarda las métricas en formato JSON.

        Args:
            filepath: Ruta donde guardar las métricas.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        print(f"📄 Métricas guardadas en: {filepath}")

    @staticmethod
    def load_model(filepath: str):
        """
        Carga un modelo previamente guardado.

        Args:
            filepath: Ruta del modelo guardado.

        Returns:
            Modelo cargado.
        """
        model = joblib.load(filepath)
        print(f"📂 Modelo cargado desde: {filepath}")
        return model


if __name__ == "__main__":
    print("Módulo de modelos cargado correctamente.")
