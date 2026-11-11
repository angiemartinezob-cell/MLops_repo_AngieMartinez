@echo off
REM Configuracion del entorno del proyecto (Windows)
REM Uso: doble clic o ejecutar "set_up.bat" desde la raiz del proyecto

echo [1/5] Creando entorno virtual...
if not exist venv (
    python -m venv venv
)
if errorlevel 1 (
    echo No se pudo crear el entorno virtual. Verifique que Python este instalado.
    exit /b 1
)

echo [2/5] Activando entorno virtual...
call venv\Scripts\activate.bat

echo [3/5] Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Error instalando dependencias.
    exit /b 1
)

echo [4/5] Registrando el kernel para los notebooks...
python -m ipykernel install --user --name venv --display-name "venv"

echo [5/5] Entrenando el modelo y generando el monitoreo...
python mlops_pipeline\model_evaluacion.py
if errorlevel 1 exit /b 1
python mlops_pipeline\monitoring.py

echo.
echo Entorno listo.
echo   API:      uvicorn deploy:app --app-dir mlops_pipeline --port 8000
echo   Tablero:  streamlit run mlops_pipeline\monitoring.py
echo   Pruebas:  pytest
