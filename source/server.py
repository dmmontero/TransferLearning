import grpc
from concurrent import futures
import image_classification_pb2
import image_classification_pb2_grpc

# import numpy as np
# import keras
# from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import warnings
from model import Model as Model

warnings.filterwarnings("ignore")

# Load model
# model = ResNet50(weights="imagenet")


class ImageClassificationServicer(
    image_classification_pb2_grpc.ImageClassificationServicer
):
    def ClassifyImage(self, request, context):
        img_path = request.image_path
        _model = Model()
        predicciones = _model.ClassifyImage(img_path)
        # img = keras.utils.load_img(img_path, target_size=(224, 224))
        # x = keras.utils.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        # preds = model.predict(x)
        # decoded_preds = decode_predictions(preds, top=3)[0]
        # predicciones = [
        #     {"name": name, "clase": clase, "score": score}
        #     for name, clase, score in decoded_preds
        # ]
        return image_classification_pb2.ImageResponse(predictions=predicciones)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    image_classification_pb2_grpc.add_ImageClassificationServicer_to_server(
        ImageClassificationServicer(), server
    )

    server.add_insecure_port("[::]:50051")
    print("The server is running!")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
