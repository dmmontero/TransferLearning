# Transfer Learning con TensorFlow Keras (ResNet50, VGG16)

ResNet50 es una arquitectura de red neuronal profunda que ha sido preentrenada en el conjunto de datos ImageNet, que contiene millones de imágenes clasificadas en miles de categorías. Utilizar ResNet50 para transfer learning implica reutilizar esta red preentrenada y ajustarla para una nueva tarea de clasificación de imágenes.

Para el ejercicio de **Transfer Learnign** se entrenó un modelo ResNet50 para el reconocimiento de flores, el modelo
resultante debe copiarse a la carpeta models del projecto, dicho modelo puede descargarlo aquí:

<https://drive.google.com/drive/folders/1g3amN3YfFOOsHX9fFsWsvK-RTdB6WXuE?usp=sharing>

Y se generó un Jupyer Book para la generación del modelo el cual puede descaragase aquí:

<https://colab.research.google.com/gist/dmmontero/13b855c09b966f9d572cfa7648e984fc/kerascv.ipynb>

Adicionalmente hay una copia del book en la carpeta models: **kerascv.ipynb**

## Nota El modelo debe estar presente en la carpeta models para hacer uso de la opción, este no se sube a GitHub por el tamaño, para el ejercicio estamo usado el formato h5

---

## Configuración

A continuación explicaremos cómo conigurar la herramienta:

Requerimientos necesarios para el funcionamiento:

- Instale Anaconda para Windows siguiendo las siguientes instrucciones:
  <https://docs.anaconda.com/anaconda/install/windows/>

- Abra Anaconda Prompt y ejecute las siguientes instrucciones:

1. Crea un ambiente virtual con la versión 3.11.9 de Python\
   _conda create -n ResNet50 python=3.11_

2. Activar el ambiente virtual creado:\
   _conda activate ResNet50_

3. Ir a la carpeta del proyecto:\
   _cd TransFerLearning_

4. Instalar los paquetes definidos para el proyecto:\
   _pip install -r requirements.txt_

5. Ejecutar el proyecto en modo developer, este comando le permite ver la api en la maquina de desarrollo:\
   _fastapi dev .\main.py_

6. Una vez ejecutado el comando podra ver los endpoint de la api con Swagger en <http://localhost:8080/docs> 8080 es el puerto por defecto si deseas mosdifcarlo debe enviar el parametro **--port #puerto**

---

## Uso de la api Gráfica

La api cuenta con 3 Endpoints:

1. **/image_predict (Post)**: Este endpoint permite el reconocimiento de una imágen RGB (JPG, PNG, GIF, TIFF, RAW) usando el modelo **ResNet50**, el endpoint permite la carga de imágenes y retorna una respuesta JSON con las 3 primeras predicciones con al clase y score
   {
   "predictions": [
   {
   "class": "African_elephant",
   "score": 0.789181113243103
   },
   {
   "class": "tusker",
   "score": 0.16567203402519226
   },
   {
   "class": "Indian_elephant",
   "score": 0.04505510255694389
   }
   ]
   }

2. **/image_features (Post)**: Este Endpoint permite el reconocimiento de una imágen RGB (JPG, PNG, GIF, TIFF, RAW) usando el modelo **VGG16**, el endpoint permite la carga de imágenes y retorna una respuesta JSON con las 3 primeras predicciones con al clase y score
   {
   "predictions": [
   {
   "class": "African_elephant",
   "score": 0.789181113243103
   },
   {
   "class": "tusker",
   "score": 0.16567203402519226
   },
   {
   "class": "Indian_elephant",
   "score": 0.04505510255694389
   }
   ]
   }
3. **/flower_predict (Post)**: Este Endpoint muestra un ejemplo de **Transfer Leraning**, para este endpoint se entrenó un modelo **ResNet50** para el reconocimiento de imágenes de flores, espeficicamente (5 clases) **['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']**

---

## Imágen Docker

El proyecto puede se ejecutado desde una Imágen Docker, para ello debe ejecutar los siguientes pasos:

1. **Crear la imágen Docker:**
   Una vez se encuentre en la raíz del proyecto ejecuta el siguiente comando para crear la imágen

   **docker build -t transfer_learning .**

2. **Ejecutar la imagen Docker:**
   Este comando corre la Api en el puerto 8282 sientase en libertad de cambiar si lo desea
   **docker run -d --name transfer_learning_container -p 8282:8282 transfer_learning**

   Tener en cuenta el parámetro **-p** ( de port) el cual nos permite mapear el puerto donde se ejecutara la API

---

## Arquitectura de archivos propuesta

### main.py

Script que contiene el punto de entrada de la aplicación, este archivo contiene la definición de la API en **Fast API**

### source/model_predictor.py

Script que invoca las predicciones en los diferentes modelos y carga el modelo entrenado **(models/RESNET50.h5)** para el reconocimiento de flores.

### models/RESNET50.h5

Archivo binario del modelo de red neuronal convolucional previamente entrenado para el reconocimiento de flores mediante Transfer Leraning usando el modelo ResNet50.h5, este modelo se puede descargar de:

<https://colab.research.google.com/gist/dmmontero/13b855c09b966f9d572cfa7648e984fc/kerascv.ipynb>

o se puede generar usando el Jupyter Book ubicado en **models/KerasCV.ipynb**

### models/KerasCV.ipynb

Jupyter Book que tiene el código para generar el modelo, este se corrió en Google Colab, sientase en libertad de cambair los parámetros de entrenamiento.

---

## Acerca del Modelo

**ResNet-50** es una red neuronal convolucional con 50 capas de profundidad. Puede cargar una versión preentrenada de la red neuronal entrenada con más de un millón de imágenes desde la base de datos de ImageNet. La red neuronal preentrenada puede clasificar imágenes en 1000 categorías de objetos (por ejemplo, teclado, ratón, lápiz y muchos animales). Como resultado, la red neuronal ha aprendido representaciones ricas en características para una amplia gama de imágenes. El tamaño de la entrada de imagen de la red neuronal es de 224 por 224.

**VGG-16** es una red neuronal convolucional con 16 capas de profundidad. Puede cargar una versión preentrenada de la red entrenada en más de un millón de imágenes desde la base de datos de ImageNet. La red preentrenada puede clasificar imágenes en 1000 categorías de objetos (por ejemplo, teclado, ratón, lápiz y muchos animales). Como resultado, la red ha aprendido representaciones ricas en características para una amplia gama de imágenes. El tamaño de la entrada de imagen de la red es de 224 por 224.

**Transfer Learning** o aprendizaje transferido en español, se refiere al conjunto de métodos que permiten transferir conocimientos adquiridos gracias a la resolución de problemas para resolver otros problemas.

**Transfer Learning** ha tenido un gran éxito con el crecimiento del Deep Learning. Frecuentemente, los modelos utilizados en este campo necesitan grandes tiempos de cálculo y muchos recursos. Sin embargo, utilizando como punto de partida modelos pre-entrenados, el Transfer Learning permite desarrollar rápidamente modelos eficaces y resolver problemas complejos de Computer Vision o Natural Language Processing, NLP.

---

## Test Cases

En la carpeta **test** del proyecto se encuentran los casos de prueba creados con **unittest** para ejecutar un caso de prueba específico ejecute el siguiente comando, las clases usadas se cargaron de la carpeta **test/pics** sientase en libertad de usar otras imágenes

**_python -m unittest discover .\test test_model_predictor.py_**

Donde **test_model_predictor.py.py** es el archivo que contiene la prueba, este comando ejecutar todas las pruebas creadas.

---

## Creado por

**Danny Mauricio Montero - <http://github.com/dmmontero>**
