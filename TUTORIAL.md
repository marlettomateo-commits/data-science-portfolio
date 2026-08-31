# 🚀 Guía Práctica: Cómo Usar Este Proyecto

> Tutorial paso a paso para poner en marcha tu portafolio de Ciencia de Datos.

---

## 📥 Paso 1: Descargar el Proyecto

Descarga la carpeta `data-science-portfolio` que se generó. Luego abre una terminal en esa ubicación.

```bash
cd data-science-portfolio
```

---

## 🐍 Paso 2: Crear Entorno Virtual

### Opción A: Con `venv` (recomendado)
```bash
# Crear entorno
python -m venv venv

# Activar (Windows)
venv\Scripts\activate

# Activar (Mac/Linux)
source venv/bin/activate
```

### Opción B: Con `conda`
```bash
conda create -n ds-portfolio python=3.11
conda activate ds-portfolio
```

> 💡 Verás `(venv)` o `(ds-portfolio)` al inicio de tu terminal. Eso significa que está activo.

---

## 📦 Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instala: pandas, numpy, scikit-learn, matplotlib, seaborn, jupyter, xgboost, etc.

### Instalar el proyecto como paquete (opcional pero recomendado)
```bash
pip install -e .
```

Esto permite hacer `from src.data import load_data` desde cualquier lugar.

---

## 🌸 Paso 4: Ejemplo Práctico con Dataset Iris

Vamos a hacer un proyecto completo usando el dataset Iris (incluido en sklearn). Esto te muestra cómo conectan **todos los módulos**.

### 4.1 Cargar y limpiar datos con `src/data/load_data.py`

```python
# scripts/ejemplo_iris.py
from sklearn.datasets import load_iris
import pandas as pd
import sys
sys.path.append('../src')

from data.load_data import clean_data, save_processed_data

# Cargar dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Guardar en data/raw/
df.to_csv('../data/raw/iris.csv', index=False)

# Limpiar datos
df_clean = clean_data(df)

# Guardar procesado
save_processed_data(df_clean, 'iris_clean.csv')
```

### 4.2 Visualizar con `src/visualization/plots.py`

```python
from visualization.plots import plot_distribution, plot_correlation_matrix

# Ver distribución de una variable
plot_distribution(df_clean, 'sepal length (cm)', 
                  save_path='../reports/figures/sepal_dist.png')

# Matriz de correlación
plot_correlation_matrix(df_clean, 
                        save_path='../reports/figures/correlation_matrix.png')
```

### 4.3 Entrenar modelos con `src/models/train_model.py`

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from models.train_model import ModelTrainer

# Preparar datos
X = df_clean.drop('target', axis=1)
y = df_clean['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear y entrenar modelo
rf = RandomForestClassifier(n_estimators=100, random_state=42)
trainer = ModelTrainer(rf, 'Random Forest')

# Entrenar
trainer.train(X_train.values, y_train.values)

# Evaluar
metrics = trainer.evaluate(X_test.values, y_test.values, task_type='classification')

# Validación cruzada
trainer.cross_validate(X.values, y.values, cv=5)

# Guardar modelo y métricas
trainer.save_model('../models/best_model.pkl')
trainer.save_metrics('../reports/metrics.json')
```

### 4.4 Ver importancia de características

```python
from visualization.plots import plot_feature_importance

plot_feature_importance(
    trainer.model, 
    X.columns.tolist(),
    save_path='../reports/figures/feature_importance.png'
)
```

---

## 📓 Paso 5: Usar los Jupyter Notebooks

### Abrir Jupyter
```bash
jupyter lab
# o
jupyter notebook
```

### Flujo típico:
1. **01_eda.ipynb** → Explora tus datos, entiende distribuciones y correlaciones
2. **02_feature_engineering.ipynb** → Crea nuevas variables, normaliza, codifica
3. **03_model_training.ipynb** → Prueba varios modelos, ajusta hiperparámetros
4. **04_model_evaluation.ipynb** → Compara métricas, elige el mejor modelo

> 💡 Los notebooks ya tienen la estructura lista. Solo reemplaza los comentarios con tu código real.

---

## 🧪 Paso 6: Ejecutar Pruebas

```bash
pytest tests/
```

Esto verifica que tus funciones funcionen correctamente.

---

## 📤 Paso 7: Subir a GitHub

### Inicializar repositorio
```bash
git init
git add .
git commit -m "feat: initial project structure with Iris example"
```

### Conectar con GitHub
```bash
git branch -M main
git remote add origin https://github.com/TU_USUARIO/data-science-portfolio.git
git push -u origin main
```

---

## 🗺️ Mapa Mental del Flujo de Trabajo

```
┌─────────────────┐
│  data/raw/      │  ← Tus datos originales (NO subir a GitHub)
│  iris.csv       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  src/data/load_data.py  │  ← Carga + Limpieza
│  clean_data()           │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  data/processed/        │  ← Datos limpios
│  iris_clean.csv         │
└────────┬────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────────────────────┐
│notebooks│  │  src/visualization/      │
│  EDA    │  │  plot_distribution()     │
│  plots  │  │  plot_correlation()      │
└────┬───┘  └───────────┬───────────────┘
     │                  │
     └────────┬─────────┘
              ▼
     ┌──────────────────────────┐
     │  src/models/             │
     │  ModelTrainer            │
     │  .train() → .evaluate()  │
     └───────────┬──────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐   ┌─────────────────┐
│ models/      │   │ reports/        │
│ best_model.  │   │ figures/        │
│     pkl      │   │ metrics.json    │
└──────────────┘   └─────────────────┘
```

---

## 🎯 Checklist para Cada Nuevo Proyecto

- [ ] Copiar esta estructura de carpetas
- [ ] Colocar datos en `data/raw/`
- [ ] Activar entorno virtual
- [ ] Ejecutar EDA en notebook
- [ ] Limpiar datos con `src/data/load_data.py`
- [ ] Ingeniería de features
- [ ] Entrenar modelos con `src/models/train_model.py`
- [ ] Generar visualizaciones con `src/visualization/plots.py`
- [ ] Guardar modelo y métricas
- [ ] Actualizar README con resultados
- [ ] Subir a GitHub

---

## ❓ Preguntas Frecuentes

**¿Por qué no subo los datos a GitHub?**
→ Porque pueden ser muy pesados. El `.gitignore` los ignora automáticamente.

**¿Cómo comparto mi proyecto si no subo los datos?**
→ Incluye un link de descarga en el README o usa `Kaggle Datasets`.

**¿Puedo usar este mismo proyecto para múltiples datasets?**
→ ¡Sí! Solo crea subcarpetas dentro de `data/raw/` o haz branches de Git.

**¿Qué hago si un paquete no se instala?**
→ Prueba: `pip install --upgrade pip` y luego reinstala.

---

¡Listo! Ahora tienes todo para empezar. 🚀
