import numpy as np
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import io
from fastapi.responses import JSONResponse


class ModelPredictor(object):
    """docstring for ClassName."""

    _modelRestNet = None

    def __init__(self, arg):
        super(ModelPredictor, self).__init__()

    @classmethod
    def get_model_restnet(cls):
        """
        Gets the ResNet50 model with pre-trained weights from the ImageNet dataset.

        Returns:
            keras.models.Model: The ResNet50 model.
        """
        if cls._modelRestNet is None:
            cls._model = ResNet50(weights="imagenet")
        return cls._modelRestNet

    @classmethod
    async def predictRestNet(self, file):
        # Leer la imagen subida
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        # Preprocesar la imagen
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        # Hacer la predicción
        preds = self.get_model().predict(img_array)
        results = decode_predictions(preds, top=3)[0]
        # Formatear los resultados
        predictions = [
            {"class": res[1], "confidence": float(res[2])} for res in results
        ]
        return JSONResponse(content={"predictions": predictions})
