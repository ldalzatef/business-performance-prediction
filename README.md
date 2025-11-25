# Business Performance Prediction

Estructura de proyecto para un modelo de predicción de negocio:

- `data/` — Aquí almacenarás los datasets (csv/xlsx)
- `models/` — Aquí guardaremos los modelos entrenados (joblib)
- `notebooks/` — Notebooks de experimentación y análisis
- `src/` — Código fuente con lógica de limpieza y predicción (package)
- `app/` — Aplicación Streamlit
- `.gitignore` — Archivos y carpetas a ignorar
- `requirements.txt` — Dependencias del proyecto

Cómo ejecutar la app localmente:

1. Crea y activa un virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Instala dependencias:

```bash
pip install -r requirements.txt
```

3. Corre la app con Streamlit:

```bash
streamlit run app/main.py
```

Notas:

- Guarda tu modelo con joblib en `models/model.joblib` antes de usar la app (o especifica otra ruta en la UI).
- Rellena `src/preprocessing.py` y `src/inference.py` con la lógica de tu dataset.
