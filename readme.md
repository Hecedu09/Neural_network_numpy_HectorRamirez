# Proyecto: Neural_network_numpy_HectorRamirez

Este proyecto implementa una red neuronal simple en Python utilizando `numpy` y `matplotlib`, con funciones de activación como Sigmoide y ReLU. Además, permite entrenar el modelo en un conjunto de datos sintético y visualizar los resultados.

## Estructura del Proyecto
```
- src/
  - numpycode.py    # Implementación de funciones de activación, inicialización de parámetros y entrenamiento
- main.py           # Script principal que ejecuta el entrenamiento del modelo
- README.md         # Documentación del proyecto
- .gitignore        # Archivos a ignorar por Git
- Requirements.txt  # Dependencias necesarias para ejecutar el proyecto
```

## Instalación y Uso
### 1. Clonar el Repositorio
Para obtener el código fuente, clona el repositorio en tu máquina local:
```bash
git clone https://github.com/tu-usuario/Neural_network_numpy_HectorRamirez.git
cd Neural_network_numpy_HectorRamirez
```

### 2. Crear un Entorno Virtual y Instalar Dependencias
Se recomienda el uso de un entorno virtual para evitar conflictos con otras bibliotecas:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r Requirements.txt
```

### 3. Ejecutar el Proyecto
Para entrenar la red neuronal y visualizar los datos, ejecuta el siguiente comando:
```bash
python main.py
```
Esto generará un conjunto de datos sintético, entrenará la red neuronal y mostrará un gráfico con los datos clasificados.

## Descripción de los Archivos
### - `main.py`
   - Punto de entrada del programa.
   - Importa la función `train_model` desde `numpycode.py` y ejecuta el entrenamiento.

### - `src/numpycode.py`
   - Contiene la implementación de:
     - Funciones de activación: **Sigmoide** y **ReLU**.
     - Función de error cuadrático medio (MSE).
     - Inicialización de parámetros de la red neuronal.
     - Propagación hacia adelante y hacia atrás (backpropagation).
     - Algoritmo de entrenamiento de la red neuronal.
     - Generación de un conjunto de datos sintético para pruebas.

## Objetivo y Funcionamiento del Código
Este proyecto tiene como objetivo demostrar el entrenamiento de una red neuronal simple utilizando únicamente `numpy`. 

1. Se genera un conjunto de datos sintético con dos clases mediante `make_gaussian_quantiles`.
2. Se inicializan los pesos y sesgos de una red con tres capas ocultas.
3. Se implementan las funciones de activación Sigmoide y ReLU.
4. La red neuronal se entrena usando propagación hacia adelante y ajuste de parámetros con descenso de gradiente.
5. Se muestra un gráfico de dispersión con los datos clasificados tras el entrenamiento.

## Dependencias del Proyecto
El archivo `Requirements.txt` incluye:
```
numpy
matplotlib
scikit-learn
```
Asegúrate de instalarlas antes de ejecutar el proyecto.
