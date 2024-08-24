from unittest import IsolatedAsyncioTestCase
from source.model_predictor import ModelPredictor as ModelPredictor
from utils.utils import getFile
from os import path


class TestPreprocessImg(IsolatedAsyncioTestCase):
    """
    Caso de prueba para los modelos
    """

    def setUp(self):
        self.predictor = ModelPredictor()

    async def test_predict_restnet(self):
        file_path = TestPreprocessImg.file_path("pics\elefante.jpg")
        resp = await self.predictor.predictRestNet(getFile(self, file_path))
        print(resp)
        assert resp.status_code == 200

    async def test_predict_vgg16(self):
        file_path = TestPreprocessImg.file_path("pics\camaleon.jpg")
        resp = await self.predictor.predictVGG16(getFile(self, file_path))
        print(resp)
        assert resp.status_code == 200

    async def test_predict_restnet_tf(self):
        file_path = TestPreprocessImg.file_path("pics\8480886751_71d88bfdc0_n.jpg")
        resp = await self.predictor.predictRestNetTF(getFile(self, file_path))
        print(resp)
        assert resp.status_code == 200

    @staticmethod
    def file_path(relative_path):
        dir = path.dirname(path.abspath(__file__))
        split_path = relative_path.split("/")
        new_path = path.join(dir, *split_path)
        return new_path


if __name__ == "__main__":
    pass
