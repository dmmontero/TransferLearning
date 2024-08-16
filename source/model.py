import numpy as np
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from PIL import Image
import io
from fastapi.responses import JSONResponse


class Model:
    _model = None

    @classmethod
    def get_model(cls):
        """
        Gets the ResNet50 model with pre-trained weights from the ImageNet dataset.

        Returns:
            keras.models.Model: The ResNet50 model.
        """
        if cls._model is None:
            cls._model = ResNet50(weights="imagenet")
        return cls._model

    @classmethod
    def ClassifyImage(self, image_path):
        img_path = image_path
        img = keras.utils.load_img(img_path, target_size=(224, 224))
        x = keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = self.get_model().predict(x)
        decoded_preds = decode_predictions(preds, top=3)[0]
        predicciones = [
            {"name": name, "clase": clase, "score": score}
            for name, clase, score in decoded_preds
        ]
        return predicciones

    @classmethod
    async def predict(self, file):
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
        predictions = [{"class": res[1], "score": float(res[2])} for res in results]
        return JSONResponse(content={"predictions": predictions})

    def __init__(self):
        pass
