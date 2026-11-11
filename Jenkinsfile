// Pipeline de validación y despliegue del modelo de riesgo de crédito.
// Configuración en Jenkins: New Item > Pipeline > "Pipeline script from SCM",
// repositorio de GitHub, rama */certification (o la que se quiera vigilar)
// y Script Path: Jenkinsfile.

pipeline {
    agent any

    // Revisa el repositorio cada 5 minutos: un merge a la rama configurada
    // dispara el pipeline automáticamente. Si Jenkins tiene URL pública se
    // puede usar githubPush() con un webhook de GitHub en lugar de pollSCM.
    triggers {
        pollSCM('H/5 * * * *')
    }

    options {
        timestamps()
        timeout(time: 40, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    environment {
        IMAGEN = 'mlops-credito-api'
        CONTENEDOR_PRUEBA = "api-prueba-${BUILD_NUMBER}"
        // Reemplazar por el correo que debe recibir la notificación
        CORREO_NOTIFICACION = 'angie.martinezob@amigo.edu.co'
    }

    stages {
        stage('Clonar repositorio') {
            steps {
                checkout scm
                sh 'git log -1 --pretty="Último commit: %h - %s (%an)"'
            }
        }

        stage('Validar estructura de carpetas') {
            steps {
                sh '''
                    set -e
                    faltantes=0
                    for archivo in \
                        mlops_pipeline/Cargar_datos.ipynb \
                        mlops_pipeline/eda_analisis.ipynb \
                        mlops_pipeline/ft_eng.py \
                        mlops_pipeline/model_evaluacion.py \
                        mlops_pipeline/deploy.py \
                        mlops_pipeline/monitoring.py \
                        Base_de_datos.csv \
                        requirements.txt \
                        .gitignore \
                        set_up.bat \
                        readme.md
                    do
                        if [ ! -s "$archivo" ]; then
                            echo "FALTA o está vacío: $archivo"
                            faltantes=$((faltantes + 1))
                        else
                            echo "OK: $archivo"
                        fi
                    done
                    if [ "$faltantes" -gt 0 ]; then
                        echo "La estructura del proyecto no cumple: $faltantes archivo(s) con problemas"
                        exit 1
                    fi
                '''
            }
        }

        stage('Preparar entorno Python') {
            steps {
                sh '''
                    set -e
                    if ! command -v python3 >/dev/null 2>&1 || ! python3 -m venv --help >/dev/null 2>&1; then
                        apt-get update && apt-get install -y python3 python3-venv python3-pip
                    fi
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Pruebas unitarias') {
            steps {
                sh '''
                    . venv/bin/activate
                    pytest --junitxml=test-results.xml --cov=mlops_pipeline --cov-report=xml:coverage.xml --cov-report=term
                '''
            }
        }

        stage('Entrenar y evaluar modelo') {
            steps {
                sh '''
                    . venv/bin/activate
                    python mlops_pipeline/ft_eng.py
                    python mlops_pipeline/model_evaluacion.py
                '''
                archiveArtifacts artifacts: 'artefactos/resultados/**', fingerprint: true
            }
        }

        stage('Monitoreo de data drift') {
            steps {
                sh '''
                    . venv/bin/activate
                    python mlops_pipeline/monitoring.py
                '''
                archiveArtifacts artifacts: 'artefactos/monitoreo/*.json, artefactos/monitoreo/resumen_periodos.csv'
            }
        }

        stage('Construir y probar imagen Docker') {
            when {
                expression { sh(script: 'command -v docker', returnStatus: true) == 0 }
            }
            steps {
                sh '''
                    set -e
                    docker build -t ${IMAGEN}:${BUILD_NUMBER} -t ${IMAGEN}:latest .
                    docker run -d --name ${CONTENEDOR_PRUEBA} ${IMAGEN}:${BUILD_NUMBER}
                    for intento in 1 2 3 4 5 6 7 8 9 10; do
                        if docker exec ${CONTENEDOR_PRUEBA} python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"; then
                            echo "La API respondió correctamente"
                            exit 0
                        fi
                        sleep 3
                    done
                    echo "La API no respondió"
                    docker logs ${CONTENEDOR_PRUEBA}
                    exit 1
                '''
            }
        }
    }

    post {
        always {
            junit allowEmptyResults: true, testResults: 'test-results.xml'
            sh 'docker rm -f ${CONTENEDOR_PRUEBA} >/dev/null 2>&1 || true'
        }
        success {
            script { notificar('EXITOSO') }
        }
        failure {
            script { notificar('FALLIDO') }
        }
    }
}

def notificar(String estado) {
    def mensaje = """Pipeline ${env.JOB_NAME} #${env.BUILD_NUMBER}: ${estado}
Rama: ${env.GIT_BRANCH ?: 'N/D'}
Detalle: ${env.BUILD_URL}"""
    echo "NOTIFICACIÓN\n${mensaje}"
    writeFile file: 'notificacion.txt', text: mensaje
    archiveArtifacts artifacts: 'notificacion.txt', allowEmptyArchive: true
    try {
        mail to: env.CORREO_NOTIFICACION,
             subject: "[Jenkins] ${env.JOB_NAME} #${env.BUILD_NUMBER} - ${estado}",
             body: mensaje
    } catch (error) {
        echo "No se envió el correo (configure SMTP en Manage Jenkins > System): ${error}"
    }
}
