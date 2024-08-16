import numpy as np

# Note that Keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

import warnings

warnings.filterwarnings("ignore")
# Load model
model = ResNet50(weights="imagenet")

img_path = "c:\\Users\\Tatiana\\Desktop\\camaleon.jpg"
img = keras.utils.load_img(img_path, target_size=(224, 224))
x = keras.utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

print("Tipo de la prediccion: ", type(preds))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print("Predicted: ", decode_predictions(preds, top=3)[0])
