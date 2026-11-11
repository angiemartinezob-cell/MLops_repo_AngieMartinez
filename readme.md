# Modelo de riesgo de crédito: predicción de pago a tiempo

Proyecto final de Machine Learning. Se desarrolla un modelo supervisado que estima la probabilidad de que un cliente **no pague a tiempo** su crédito, junto con el flujo MLOps completo: ingeniería de características, entrenamiento y evaluación, despliegue como API en Docker, monitoreo de data drift con un tablero en Streamlit, pruebas automáticas, análisis de calidad en SonarCloud y pipeline en Jenkins.

## 1. Caso de negocio

Una entidad que otorga créditos necesita identificar, al momento de la solicitud, qué clientes tienen mayor riesgo de no pagar a tiempo. Un cliente que entra en mora genera pérdidas por cartera vencida y costos de cobranza, mientras que rechazar a un buen cliente significa perder un negocio.

El modelo no reemplaza la decisión del analista: entrega una **probabilidad de no pago** y una clasificación de riesgo (Alto/Bajo) para priorizar qué solicitudes revisar con más detalle.

- **Variable objetivo:** `Pago_atiempo` (1 = paga a tiempo, 0 = no paga a tiempo).
- **Clase de interés:** 0, que representa solo el **4,75 %** de los registros (511 de 10.763).
- **Datos:** `Base_de_datos.csv`, 10.763 créditos otorgados entre noviembre de 2024 y abril de 2026, con 23 variables del crédito, del cliente y de centrales de riesgo (Datacrédito).

## 2. Estructura del repositorio

```
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb        Carga y validación de la estructura de la base
│   ├── eda_analisis.ipynb        Análisis exploratorio, reglas de validación y decisiones
│   ├── ft_eng.py                 Ingeniería de características (pipeline de scikit-learn)
│   ├── model_evaluacion.py       Entrenamiento, comparación y selección del modelo
│   ├── deploy.py                 API con FastAPI para predicción por lotes
│   └── monitoring.py             Monitoreo de data drift + tablero en Streamlit
├── tests/                        Pruebas unitarias (pytest)
├── .github/workflows/            Pruebas y análisis en SonarCloud (GitHub Actions)
├── Base_de_datos.csv
├── config.json                   Parámetros del proyecto (rutas, umbrales, monitoreo)
├── requirements.txt
├── set_up.bat                    Configuración del entorno en Windows
├── Dockerfile / .dockerignore    Imagen de la API
├── Jenkinsfile                   Pipeline de Jenkins
├── sonar-project.properties      Configuración de SonarCloud
├── .gitignore
└── readme.md
```

Ramas: `developer` (desarrollo), `certification` (validación) y `master` (producción). El trabajo se hace en `developer` y se integra hacia `certification` y luego a `master` mediante merge.

## 3. Cómo ejecutar el proyecto

**Windows (recomendado):** ejecutar `set_up.bat` desde la raíz. Crea el entorno virtual, instala las dependencias, registra el kernel de Jupyter, entrena el modelo y genera el monitoreo.

**Manual:**

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt

python mlops_pipeline/ft_eng.py              # revisa los conjuntos y el preprocesamiento
python mlops_pipeline/model_evaluacion.py    # entrena, compara y guarda el mejor modelo
python mlops_pipeline/monitoring.py          # calcula el drift y genera las alertas
```

Los resultados se guardan en la carpeta `artefactos/`, que no se sube al repositorio porque se regenera con los scripts:

- `artefactos/modelo_final.joblib`: modelo seleccionado, umbral y métricas.
- `artefactos/resultados/`: tablas de evaluación y gráficos comparativos.
- `artefactos/monitoreo/`: tabla de predicciones, métricas de drift por periodo y alertas.

## 4. Hallazgos del análisis exploratorio

El detalle está en `eda_analisis.ipynb`. Los hallazgos principales son:

1. **Fuga de información en `puntaje`.** Tiene una correlación de 0,92 con la variable objetivo y separa perfectamente las clases: el puntaje máximo de quienes no pagan es 62,67 y el mínimo de quienes pagan es 63,81. Una sola regla acierta el 100 %, lo que indica que se calculó con información del resultado. **Se excluye del modelo.**
2. **Bloque de 150 registros atípicos** (índices 10.470 a 10.619): edad de 121 a 123 años, puntaje Datacrédito en 0, sin información de ingresos y con 19 de los 22 salarios mayores a 1.000 millones. Tienen más no pago (7,33 % frente a 4,71 %), por lo que se conservan y se marcan con la variable `registro_atipico`.
3. **Los nulos no son aleatorios.** Cuando falta información de saldos o de Datacrédito el no pago sube de cerca de 4,7 % a 7 %. Se crean indicadores de nulo y en `tendencia_ingresos` los nulos se tratan como la categoría "Sin_informacion".
4. **`tendencia_ingresos` tenía 58 valores numéricos** mezclados con las categorías; se convierten en nulos.
5. **`saldo_mora_codeudor` tiene varianza casi nula** (99,97 % de ceros) y se elimina. En cambio, `saldo_mora` se conserva: los 55 clientes con mora tienen 36,36 % de no pago.
6. **Plazos largos con más riesgo:** 8,19 % de no pago a 24 meses y 13,25 % a 36 meses.
7. **Variables monetarias muy asimétricas** (skewness de hasta 43,8); con log(1 + x) `capital_prestado` y `cuota_pactada` quedan casi simétricas.
8. **Cambios en el tiempo:** el volumen de créditos cae desde junio de 2025 y la tasa de no pago baja a cerca de 2 % a finales de 2025, posiblemente porque los créditos recientes aún no terminan su plazo. Por eso la fecha no se usa como variable del modelo, pero sí en el monitoreo.
9. Las pruebas de Mann-Whitney y Chi-cuadrado muestran diferencias significativas en 13 de 21 variables, pero con mucho solapamiento entre las clases, lo que anticipa una capacidad predictiva moderada sin `puntaje`.

## 5. Ingeniería de características (`ft_eng.py`)

Toda la transformación está dentro de un único `Pipeline` de scikit-learn, por lo que el modelo guardado recibe los **datos crudos** y aplica exactamente la misma lógica en entrenamiento, en la API y en el monitoreo.

| Paso | Detalle |
|---|---|
| Limpieza | Conversión de tipos (acepta coma decimal), `tipo_credito` como categoría, valores inválidos de `tendencia_ingresos` a nulo |
| Variables derivadas | `ratio_cuota_ingreso`, `ratio_deuda_ingreso` (evitan la división por salario 0), `total_creditos_sectores`, `registro_atipico` |
| Variables excluidas | `puntaje` (fuga), `saldo_mora_codeudor` (varianza nula), `fecha_prestamo` |
| Monetarias | Imputación con la mediana → log(1 + x) → RobustScaler |
| Numéricas | Imputación con la mediana → RobustScaler |
| Indicadores de nulo | Uno por cada grupo de nulos coincidentes (`saldo_mora`, `saldo_principal`, `puntaje_datacredito`, ratios) |
| Categóricas | Nulos como "Sin_informacion" → One-Hot; categorías con menos de 20 registros se agrupan y las desconocidas se ignoran |
| División | 80 % entrenamiento / 20 % prueba, estratificada (4,75 % de no pago en ambos) |

## 6. Entrenamiento y evaluación (`model_evaluacion.py`)

Se usan las funciones `build_model` (arma el pipeline features + modelo) y `summarize_classification` (métricas tomando el no pago como clase positiva). Todos los modelos compensan el desbalance con `class_weight="balanced"`.

**Metodología:**

1. Los cuatro modelos se comparan con **validación cruzada estratificada de 5 folds** solo sobre el entrenamiento. La media mide el desempeño y la desviación estándar la consistencia.
2. El **umbral de decisión** de cada modelo se calcula con las predicciones fuera de muestra del entrenamiento (índice de Youden). Con 5 % de no pago, el umbral de 0,5 casi nunca marca a nadie como riesgo.
3. El mejor modelo se elige por **PR-AUC** en validación cruzada, porque es la métrica adecuada cuando la clase de interés es muy pequeña.
4. El conjunto de prueba se usa una sola vez, para la evaluación final.

**Validación cruzada (entrenamiento, 5 folds):**

| Modelo | ROC-AUC | PR-AUC | KS | Balanced acc. | Recall no pago | Recall pago | Tiempo/fold |
|---|---|---|---|---|---|---|---|
| **Regresión Logística** | **0,673 ± 0,017** | **0,135 ± 0,021** | 0,289 | 0,645 | 0,658 | 0,632 | 0,1 s |
| Random Forest | 0,681 ± 0,016 | 0,132 ± 0,024 | 0,268 | 0,634 | 0,550 | 0,717 | 1,7 s |
| HistGradientBoosting | 0,655 ± 0,029 | 0,126 ± 0,019 | 0,241 | 0,620 | 0,575 | 0,666 | 1,2 s |
| Árbol de Decisión | 0,594 ± 0,027 | 0,079 ± 0,010 | 0,140 | 0,570 | 0,550 | 0,590 | 0,1 s |

**Evaluación final en prueba (2.153 registros, 102 de no pago):**

| Modelo | ROC-AUC | Gini | PR-AUC | KS | Balanced acc. | Recall no pago | Precisión no pago |
|---|---|---|---|---|---|---|---|
| **Regresión Logística** | **0,691** | **0,381** | 0,145 | 0,291 | **0,640** | **0,667** | 0,079 |
| Random Forest | 0,685 | 0,369 | 0,184 | 0,280 | 0,610 | 0,510 | 0,080 |
| HistGradientBoosting | 0,650 | 0,299 | 0,198 | 0,256 | 0,618 | 0,588 | 0,077 |
| Árbol de Decisión | 0,644 | 0,288 | 0,108 | 0,221 | 0,607 | 0,647 | 0,069 |

**Modelo seleccionado: Regresión Logística** (umbral 0,48).

- Tiene la mayor PR-AUC en validación cruzada, con una variación entre folds similar a la de Random Forest (consistencia).
- En prueba tiene el mayor ROC-AUC, Gini, KS, balanced accuracy y recall de no pago.
- Es el más escalable: entrena en 0,1 s frente a 1,7 s de Random Forest y predice en 0,006 ms por registro. Además es interpretable, lo cual es importante en decisiones de crédito.
- En prueba, HistGradientBoosting y Random Forest tienen mayor PR-AUC. Sin embargo, el conjunto de prueba solo tiene 102 casos de no pago, y en validación cruzada (más estable) esa ventaja no aparece, por lo que no se cambia la selección con base en el test.

**Interpretación:** con el umbral elegido el modelo detecta 68 de los 102 clientes que no pagan (66,7 %), a cambio de marcar como alto riesgo a cerca del 39 % de los clientes que sí pagan. La precisión de 7,9 % es baja, pero es 1,7 veces la tasa base (4,75 %). El modelo sirve para **priorizar la revisión manual**, no para rechazar solicitudes automáticamente. Si el negocio quiere marcar menos clientes, basta con subir el umbral.

Sin `puntaje` la capacidad predictiva de los datos es moderada (ROC-AUC cercano a 0,68). Este es un resultado honesto: el desempeño casi perfecto que se obtendría con `puntaje` no se repetiría con clientes nuevos.

**Variables más importantes** (importancia por permutación en prueba): `puntaje_datacredito`, `plazo_meses`, `huella_consulta`, `cuota_pactada`, `saldo_principal`, `capital_prestado` y `edad_cliente`.

Los gráficos comparativos (barras de validación cruzada, curvas ROC y Precisión-Recall, matriz de confusión e importancia de variables) se generan en `artefactos/resultados/`.

## 7. Despliegue (`deploy.py` + Docker)

API con FastAPI que carga `modelo_final.joblib` y predice por lotes.

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y disponibilidad del modelo |
| GET | `/modelo` | Nombre, umbral, métricas y fecha de entrenamiento |
| POST | `/predict` | Lote de registros en JSON: `{"registros": [ {...}, {...} ]}` |
| POST | `/predict/csv` | Archivo CSV (separador `;` o `,`) con varios registros |
| GET | `/docs` | Documentación interactiva (Swagger) |

Por cada registro devuelve la probabilidad de no pago, la predicción de `Pago_atiempo`, el nivel de riesgo y **alertas de validación** basadas en las reglas del EDA (edad fuera de 18-100, salario en 0, puntaje fuera de 0-1000, categorías no válidas, campos obligatorios vacíos). Las alertas no bloquean la predicción, pero avisan sobre datos sospechosos.

**Local:**

```bash
uvicorn deploy:app --app-dir mlops_pipeline --host 0.0.0.0 --port 8000
```

**Docker** (la imagen entrena el modelo durante la construcción, para que el `.joblib` quede generado con las mismas versiones de librerías con las que se carga):

```bash
docker build -t mlops-credito-api .
docker run -d -p 8000:8000 --name api-credito mlops-credito-api
```

**Ejemplo de predicción:**

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"registros\": [{\"tipo_credito\": 4, \"capital_prestado\": 1852560, \"plazo_meses\": 12, \"edad_cliente\": 32, \"tipo_laboral\": \"Empleado\", \"salario_cliente\": 3500000, \"cuota_pactada\": 128650, \"puntaje_datacredito\": 795, \"huella_consulta\": 2, \"tendencia_ingresos\": \"Creciente\"}]}"

curl -X POST http://localhost:8000/predict/csv -F "archivo=@Base_de_datos.csv"
```

## 8. Monitoreo (`monitoring.py` + Streamlit)

```bash
python mlops_pipeline/monitoring.py           # trabajo de monitoreo
streamlit run mlops_pipeline/monitoring.py    # tablero
```

**Funcionamiento:**

- Se construye una tabla con los datos y los pronósticos del modelo (`tabla_predicciones.csv`).
- **Referencia:** créditos anteriores a abril de 2025 (5.528 registros). **Periodos actuales:** cada mes desde abril de 2025, con una muestra de hasta 500 registros por mes.
- Métricas por variable: **KS** (numéricas), **Chi-cuadrado** (categóricas), **PSI** y **divergencia de Jensen-Shannon** (todas). Los nulos se cuentan como un bin aparte, así un cambio en la tasa de faltantes también se detecta. Para las pruebas de hipótesis se usa corrección de Bonferroni, porque se prueban muchas variables a la vez.
- También se monitorea el drift de la probabilidad predicha, la tasa real de no pago frente a la predicha y el ROC-AUC de cada periodo.
- **Semáforo por variable:** rojo si PSI ≥ 0,25 o JS ≥ 0,05; amarillo si PSI ≥ 0,10, JS ≥ 0,02 o la prueba es significativa; verde en otro caso. **Semáforo del periodo:** rojo si el PSI de la predicción es ≥ 0,25 o al menos el 20 % de las variables está en rojo. Los meses con menos de 50 registros se marcan como "Muestra insuficiente".
- Se detectan **cambios abruptos** (el PSI sube más de 0,10 de un mes al siguiente) y **tendencias** (el PSI sube tres meses seguidos), y se generan **alertas y recomendaciones automáticas** (reentrenamiento, revisión de variables, revisión de la fuente de datos) en `alertas.json`.

El tablero muestra el semáforo general, indicadores del periodo, la tabla de métricas por variable con colores, barras de riesgo, la comparación de la distribución histórica frente a la actual, la evolución del drift en el tiempo y las recomendaciones. Todos los umbrales se configuran en `config.json`.

**Resultados:** de abril a julio de 2025 la población es estable en general (estado amarillo con pocas variables en rojo). A partir de agosto de 2025 aparece drift en `capital_prestado`, `total_otros_prestamos`, `ratio_deuda_ingreso` y `promedio_ingresos_datacredito`. En enero y febrero de 2026 la probabilidad predicha también se desplaza (PSI de 0,27 a 0,28), por lo que el sistema recomienda **reentrenar el modelo** con datos recientes. Como el modelo se entrenó con una partición aleatoria de toda la base, el ROC-AUC por periodo incluye registros de entrenamiento; en producción se calcularía solo con datos nuevos.

## 9. Pruebas y calidad de código

```bash
pytest --cov=mlops_pipeline --cov-report=term
```

Las pruebas (`tests/`) cubren la estructura del repositorio, la limpieza y las variables derivadas, el pipeline de features, las métricas y la selección del modelo, las métricas de drift (PSI, KS, JS, Chi-cuadrado), el trabajo de monitoreo y la API (endpoints JSON y CSV, validaciones y errores).

**SonarCloud:** en cada push a `developer`, `certification` o `master`, el workflow `.github/workflows/sonarcloud.yml` ejecuta las pruebas con cobertura y envía el análisis a SonarCloud (calidad, seguridad, cobertura y estilo). Configuración necesaria, una sola vez:

1. Entrar a [sonarcloud.io](https://sonarcloud.io) con GitHub e importar el repositorio.
2. En *Administration > Analysis Method*, desactivar *Automatic Analysis*.
3. Generar un token en *My Account > Security* y agregarlo en GitHub como secreto del repositorio con el nombre `SONAR_TOKEN` (*Settings > Secrets and variables > Actions*).
4. Verificar que `sonar.organization` y `sonar.projectKey` en `sonar-project.properties` coincidan con los que muestra SonarCloud.

## 10. Jenkins

El `Jenkinsfile` define el pipeline: clonar el repositorio → validar la estructura de carpetas → preparar el entorno → pruebas unitarias → entrenar y evaluar el modelo → monitoreo → construir la imagen Docker y verificar que la API responda → notificación del resultado (en consola, en un archivo archivado y por correo si hay SMTP configurado).

1. Levantar Jenkins con Docker Compose (imagen `jenkins/jenkins:lts-jdk17`, puerto 8080, montando `/var/run/docker.sock`) y seguir el asistente inicial con *Install suggested plugins*.
2. Dentro del contenedor, si no está Docker: `apt-get update && apt install -y docker.io`.
3. *New Item > Pipeline*. En *Definition* elegir **Pipeline script from SCM**, Git, la URL del repositorio, la rama `*/certification` y *Script Path* `Jenkinsfile`.
4. El trigger `pollSCM` revisa el repositorio cada 5 minutos, así que un merge a `certification` dispara el pipeline automáticamente.

Como el pipeline se ejecuta desde SCM, Jenkins clona el repositorio en el workspace del job y los scripts se ejecutan desde la raíz del proyecto, sin necesidad de retroceder carpetas.

## 11. Limitaciones y próximos pasos

- Validar con el negocio cómo se construye `puntaje`, el origen del bloque de registros con edad mayor a 100 y que `saldo_mora` corresponda al momento de la solicitud.
- La tasa de no pago de los créditos recientes puede estar subestimada porque no han terminado su plazo. Para reentrenar se recomienda usar solo créditos con plazo cumplido.
- Calibrar el umbral con el costo real de un cliente en mora frente al de rechazar un buen cliente.
- Programar el monitoreo de forma periódica (por ejemplo, mensual con Jenkins) y reentrenar cuando el estado general sea rojo.
