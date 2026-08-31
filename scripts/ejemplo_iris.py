#!/usr/bin/env python3
"""
🌸 Ejemplo Completo: Proyecto Iris

Este script demuestra cómo usar TODOS los módulos del proyecto juntos:
  1. Cargar datos con src/data/load_data.py
  2. Visualizar con src/visualization/plots.py
  3. Entrenar modelos con src/models/train_model.py
  4. Guardar resultados

Para ejecutar:
    python ejemplo_iris.py
"""

import sys
from pathlib import Path

# Agregar src/ al path para poder importar los módulos
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Importar nuestros módulos del proyecto
from data.load_data import clean_data, save_processed_data
from visualization.plots import (
    plot_distribution, 
    plot_correlation_matrix,
    plot_feature_importance,
    plot_confusion_matrix
)
from models.train_model import ModelTrainer


def main():
    print("=" * 60)
    print("🚀 PROYECTO IRIS - Ejemplo Completo de Data Science")
    print("=" * 60)

    # ============================================================
    # 1. CARGAR DATOS
    # ============================================================
    print("\n📥 1. CARGANDO DATOS...")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    # Guardar datos raw
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "iris.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)
    print(f"   ✅ Datos guardados en: {raw_path}")
    print(f"   📊 Shape: {df.shape}")
    print(f"   🏷️  Clases: {iris.target_names.tolist()}")

    # ============================================================
    # 2. LIMPIAR DATOS
    # ============================================================
    print("\n🧹 2. LIMPIANDO DATOS...")
    df_clean = clean_data(df)
    save_processed_data(df_clean, "iris_clean.csv")

    # ============================================================
    # 3. VISUALIZACIONES
    # ============================================================
    print("\n📊 3. GENERANDO VISUALIZACIONES...")
    reports_dir = Path(__file__).parent.parent / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Distribución
    plot_distribution(
        df_clean, 
        'sepal length (cm)',
        save_path=str(reports_dir / 'sepal_length_dist.png')
    )

    # Correlación
    plot_correlation_matrix(
        df_clean,
        save_path=str(reports_dir / 'correlation_matrix.png')
    )
    print("   ✅ Gráficos guardados en reports/figures/")

    # ============================================================
    # 4. PREPARAR DATOS PARA MODELADO
    # ============================================================
    print("\n⚙️  4. PREPARANDO DATOS PARA MODELADO...")
    X = df_clean.drop('target', axis=1)
    y = df_clean['target']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # ============================================================
    # 5. ENTRENAR Y COMPARAR MODELOS
    # ============================================================
    print("\n🤖 5. ENTRENANDO MODELOS...")

    models = {
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        print(f"\n   🔹 {name}")
        trainer = ModelTrainer(model, name)

        # Entrenar
        trainer.train(X_train.values, y_train.values)

        # Evaluar
        metrics = trainer.evaluate(X_test.values, y_test.values, task_type='classification')

        # Cross-validation
        trainer.cross_validate(X.values, y.values, cv=5)

        # Guardar el mejor
        if metrics['accuracy'] > best_score:
            best_score = metrics['accuracy']
            best_model = trainer
            best_name = name

    print(f"\n🏆 MEJOR MODELO: {best_name} (Accuracy: {best_score:.4f})")

    # ============================================================
    # 6. GUARDAR RESULTADOS
    # ============================================================
    print("\n💾 6. GUARDANDO RESULTADOS...")

    # Guardar modelo
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model.save_model(str(models_dir / 'best_model.pkl'))

    # Guardar métricas
    reports_dir = Path(__file__).parent.parent / "reports"
    best_model.save_metrics(str(reports_dir / 'metrics.json'))

    # Importancia de características (solo para Random Forest)
    if hasattr(best_model.model, 'feature_importances_'):
        plot_feature_importance(
            best_model.model,
            feature_names,
            save_path=str(reports_dir / 'figures' / 'feature_importance.png')
        )

    # Matriz de confusión
    y_pred = best_model.model.predict(X_test.values)
    plot_confusion_matrix(
        y_test.values, 
        y_pred,
        labels=iris.target_names.tolist(),
        save_path=str(reports_dir / 'figures' / 'confusion_matrix.png')
    )

    print("\n" + "=" * 60)
    print("✅ ¡PROYECTO COMPLETADO!")
    print("=" * 60)
    print("\n📁 Archivos generados:")
    print("   • data/raw/iris.csv")
    print("   • data/processed/iris_clean.csv")
    print("   • models/best_model.pkl")
    print("   • reports/metrics.json")
    print("   • reports/figures/*.png")
    print("\n🎉 Ahora sube todo a GitHub y muestra tu trabajo!")


if __name__ == "__main__":
    main()
