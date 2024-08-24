import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16, decode_predictions as decode_predictionsVGG
from keras.preprocessing import image
from PIL import Image
import io
from fastapi.responses import JSONResponse


tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)


class ModelPredictor(object):
    """Clase usada para caragar los modelos y realizar predicciones"""

    CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    _modelResNet = None
    _modelVGG16 = None
    _modelResNetTransfer = None

    def __init__(self):
        super(ModelPredictor, self).__init__()

    @classmethod
    def get_model_resnet(cls):
        """
        Gets the ResNet50 model with pre-trained weights from the ImageNet dataset.

        Returns:
            keras.models.Model: The ResNet50 model.
        """
        if cls._modelResNet is None:
            cls._modelResNet = ResNet50(weights="imagenet")
        return cls._modelResNet

    @classmethod
    def get_model_vgg16(cls):
        """
        Gets the ResNet50 model with pre-trained weights from the ImageNet dataset.

        Returns:
            keras.models.Model: The ResNet50 model.
        """
        if cls._modelVGG16 is None:
            cls._modelVGG16 = VGG16(weights="imagenet")
        return cls._modelVGG16

    @classmethod
    def get_model_resnet_transfer(cls):
        """
        Gets the ResNet50 model with pre-trained weights from the ImageNet dataset.

        Returns:
            keras.models.Model: The ResNet50 model.
        """
        if cls._modelResNetTransfer is None:
            cls._modelResNetTransfer = tf.keras.models.load_model(
                "./models/RESNET50.h5"
            )
        return cls._modelResNetTransfer

    @classmethod
    async def predictRestNet(self, file):
        # Leer la imagen subida
        img_array = await self.process_image(file)
        # Hacer la predicción
        preds = self.get_model_resnet().predict(img_array)
        results = decode_predictions(preds, top=3)[0]
        # Formatear los resultados
        predictions = [{"class": res[1], "score": float(res[2])} for res in results]
        return JSONResponse(content={"predictions": predictions})

    @classmethod
    async def predictRestNetTF(self, file):
        # Leer la imagen subida
        img_array = await self.process_image(file)
        # Hacer la predicción
        preds = self.get_model_resnet_transfer().predict(img_array)
        # Formatear los resultados
        image_output_classr = self.CLASS_NAMES[np.argmax(preds)]
        return JSONResponse(content={"prediction": image_output_classr})

    @classmethod
    async def predictVGG16(self, file):
        # Leer la imagen subida
        img_array = await self.process_image(file)
        # extraer features de la imagen
        preds = self.get_model_vgg16().predict(img_array)
        # Let's predict top 5 results
        results = decode_predictionsVGG(preds, top=3)[0]
        predictions = [{"class": res[1], "score": float(res[2])} for res in results]
        return JSONResponse(content={"predictions": predictions})

    @classmethod
    async def process_image(self, file):
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        # Preprocesar la imagen
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
