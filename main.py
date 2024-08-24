from fastapi import FastAPI, UploadFile, File
from fastapi.logger import logger as fastapi_logger
from .source.model_predictor import ModelPredictor as ModelPredictor
import logging

app = FastAPI(
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    title="API for Keras CV models",
    description="DESARROLLO DE PROYECTOS DE INTELIGENCIA ARTIFICIAL-HL - 1HL - (2024-24)",
)


logger = logging.getLogger("gunicorn.error")
fastapi_logger.handlers = logger.handlers
fastapi_logger.setLevel(logger.level)


@app.post("/image_predict/")
async def image_predict(uploaded_file: UploadFile = File(...)):
    _model = ModelPredictor()
    return await _model.predictRestNet(uploaded_file)


@app.post("/image_features/")
async def image_features(uploaded_file: UploadFile = File(...)):
    _model = ModelPredictor()
    return await _model.predictVGG16(uploaded_file)


@app.post("/flower_predict/")
async def flower_predict(uploaded_file: UploadFile = File(...)):
    _model = ModelPredictor()
    return await _model.predictRestNetTF(uploaded_file)
