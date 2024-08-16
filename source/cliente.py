import grpc
import image_classification_pb2
import image_classification_pb2_grpc


def run():
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = image_classification_pb2_grpc.ImageClassificationStub(channel)
        response = stub.ClassifyImage(
            image_classification_pb2.ImageRequest(
                image_path="c:\\Users\\Tatiana\\Desktop\\camaleon.jpg"
            )
        )

        print("Predictions: ", response.predictions)


if __name__ == "__main__":
    run()
