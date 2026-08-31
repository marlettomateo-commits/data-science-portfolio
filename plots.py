"""
Módulo de visualización para análisis de datos.

Funciones:
    plot_distribution: Grafica distribución de una variable.
    plot_correlation_matrix: Matriz de correlación con heatmap.
    plot_feature_importance: Importancia de características.
    plot_confusion_matrix: Matriz de confusión.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_distribution(df, column, figsize=(10, 6), save_path=None):
    """
    Grafica la distribución de una variable numérica.

    Args:
        df: DataFrame con los datos.
        column: Nombre de la columna a graficar.
        figsize: Tamaño de la figura.
        save_path: Ruta para guardar la imagen (opcional).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histograma
    sns.histplot(data=df, x=column, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title(f'Distribución de {column}')
    axes[0].set_xlabel(column)

    # Boxplot
    sns.boxplot(data=df, y=column, ax=axes[1], color='steelblue')
    axes[1].set_title(f'Boxplot de {column}')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")

    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10), save_path=None):
    """
    Genera un heatmap de la matriz de correlación.

    Args:
        df: DataFrame con variables numéricas.
        figsize: Tamaño de la figura.
        save_path: Ruta para guardar la imagen (opcional).
    """
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5)
    plt.title('Matriz de Correlación', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_feature_importance(model, feature_names, top_n=10, save_path=None):
    """
    Grafica la importancia de características de un modelo.

    Args:
        model: Modelo entrenado con atributo feature_importances_.
        feature_names: Lista de nombres de características.
        top_n: Número de características top a mostrar.
        save_path: Ruta para guardar la imagen (opcional).
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices], align='center', color='forestgreen')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importancia')
    plt.title(f'Top {top_n} Características Más Importantes')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Grafica la matriz de confusión.

    Args:
        y_true: Valores reales.
        y_pred: Valores predichos.
        labels: Etiquetas de las clases.
        save_path: Ruta para guardar la imagen (opcional).
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    print("Módulo de visualización cargado correctamente.")
