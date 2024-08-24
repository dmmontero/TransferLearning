# Transfer Learning con TensorFlow Keras (ResNet50, VGG16)

ResNet50 es una arquitectura de red neuronal profunda que ha sido preentrenada en el conjunto de datos ImageNet, que contiene millones de imágenes clasificadas en miles de categorías. Utilizar ResNet50 para transfer learning implica reutilizar esta red preentrenada y ajustarla para una nueva tarea de clasificación de imágenes.

<https://medium.com/@t.mostafid/overview-of-vgg16-xception-mobilenet-and-resnet50-neural-networks-c678e0c0ee85>

<https://www.gopichandrakesan.com/day-34-predict-an-image-using-vgg16-pretrained-model/>

Para ejercicio de **Transfer Learnign** se entrenó un modelo ResNet50 para el reconocimiento de flores, el modelo
resultante debe copiarse a la carpeta models del projecto, dicho modelo puede descargarlo aquí:

<https://drive.google.com/drive/folders/1g3amN3YfFOOsHX9fFsWsvK-RTdB6WXuE?usp=sharing>

Y se generó un Jupyer Book para la generación del modelo el cual puede descaragase aquí:

<https://colab.research.google.com/gist/dmmontero/13b855c09b966f9d572cfa7648e984fc/kerascv.ipynb>

Adicionalmente hay una copia del book en la carpeta models: **kerascv.ipynb**

## Nota El modelo debe estar presente en la carpeta models para hacer uso de la opción, este no se sube a GitHub por el tamaño para el ejercicio estamo usado el formato h5

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
   _conda activate neumonia_

3. Ir a la carpeta del proyecto:\
   _cd TransFerLearning_

4. Instalar los paquetes definidos para el proyecto:\
   _pip install -r requirements.txt_

5. Ejecutar el proyecto en modo developer, este comando le permite ver la api en la maquina de desarrollo:\
   _fastapi dev .\main.py_

6. Una vez ejecutado el comando podra ver los endpoint de la api con Swagger en <http://localhost:8080/docs> 8080 es el puerto por defecto sidese mosdifcarlo debe enviar el parametro **--port #puerto**

## Uso de la api Gráfica

La api cuenta con 3 endpoints:

1. /image_predict (Post): Este endpoint permite el reconocimiento de una imagen RGB (JPG, PNG, GIF, TIFF, RAW) usando el modelo **ResNet50**, el enpoint permite la carga de imágenes y retorna una respuesta JSON con las 3 primeras predicciones con al clase y score
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

2. /image_features (Post): Este endpoint permite el reconocimiento de una imagen RGB (JPG, PNG, GIF, TIFF, RAW) usando el modelo **VGG16**, el enpoint permite la carga de imagenes y retorna una respuesta JSON con las 3 primeras predicciones con al clase y score
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
3. /flower_predict (Post): Este endpotin muestra un ejemplo de **Transfer Leraning**, para este ednpoint se entrenó un modelo **ResNet50** para el reconocieminto de imágenes de flores, espeficicamente (5 clases) **['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']**

## Imágen Docker

El proyecto puede se ejecutado desde una Imagen Docker, para ello debe ejecutar los siguientes pasos:

1. Crear la imagen Docker:
   Una vez se encuentre en la raiz del proyecto ejecuta el siguiente comando para crear la imágen

   **docker build -t transfer_learning .**

2. Ejecutar la imagen Docker:
   Este comando corre la Api en el puerto 8282 sientase en libertad de cambiar si lo desea
   **docker run -d --name transfer_learning_container -p 8282:8282 transfer_learning**

   Tener en cuenta el parámetro **-p** ( de port) el cual nos permite mapear el puerto donde se ejecutara la API

---

## Arquitectura de archivos propuesta

Los archivos se encuentra dentro de la carpeta **src** a excepcion del _detector_neumonia.py_
que es el punto de entra de la app y se encuenta en la raíz del proyecto.

## detector_neumonia.py

Contiene el diseño de la interfaz gráfica utilizando Tkinter.

Los botones llaman métodos contenidos en otros scripts.

## integrator.py

Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.
Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM.

## read_img.py

Script que lee la imagen en formato DICOM para visualizarla en la interfaz gráfica. Además, la convierte a arreglo para su preprocesamiento.

## preprocess_img.py

Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:

- resize a 512x512
- conversión a escala de grises
- ecualización del histograma con CLAHE
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor)

## load_model.py

Script que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado **'conv_MLP_84.h5.h5'**.
el cual debe cargarse a la carpeta modelos y se puede descaragar aqui:

**<https://drive.google.com/file/d/18rgX66x7eMHci0bAimoycCe8BQLjBWK_/view?usp=sharing>**

## grad_cam.py

Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.

---

## Acerca del Modelo

**ResNet-50** es una red neuronal convolucional con 50 capas de profundidad. Puede cargar una versión preentrenada de la red neuronal entrenada con más de un millón de imágenes desde la base de datos [1] de ImageNet. La red neuronal preentrenada puede clasificar imágenes en 1000 categorías de objetos (por ejemplo, teclado, ratón, lápiz y muchos animales). Como resultado, la red neuronal ha aprendido representaciones ricas en características para una amplia gama de imágenes. El tamaño de la entrada de imagen de la red neuronal es de 224 por 224.

**VGG-16** es una red neuronal convolucional con 16 capas de profundidad. Puede cargar una versión preentrenada de la red entrenada en más de un millón de imágenes desde la base de datos [1] de ImageNet. La red preentrenada puede clasificar imágenes en 1000 categorías de objetos (por ejemplo, teclado, ratón, lápiz y muchos animales). Como resultado, la red ha aprendido representaciones ricas en características para una amplia gama de imágenes. El tamaño de la entrada de imagen de la red es de 224 por 224.

**Transfer Learning** o aprendizaje transferido en español, se refiere al conjunto de métodos que permiten transferir conocimientos adquiridos gracias a la resolución de problemas para resolver otros problemas.

**Transfer Learning** ha tenido un gran éxito con el crecimiento del Deep Learning. Frecuentemente, los modelos utilizados en este campo necesitan grandes tiempos de cálculo y muchos recursos. Sin embargo, utilizando como punto de partida modelos pre-entrenados, el Transfer Learning permite desarrollar rápidamente modelos eficaces y resolver problemas complejos de Computer Vision o Natural Language Processing, NLP.

---

## Test Cases

En la carpeta **test** del proyecto se encuentra los casos de prueba creados con **unittest** para ejecutar un caso de prueba específico ejecute el siguiente comando:

**_python -m unittest discover .\test test_load_model.py_**

Donde _test_load_model.py_ en el archivo que contiene la prueba este comando ejectar todas las pruebas creadas.

---

## Creado por

**Danny Mauricio Montero - <http://github.com/dmmontero>**
