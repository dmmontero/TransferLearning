from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Annotated
from source.model import Model as Model

app = FastAPI(
    swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"},
    title="API for Keras CV models",
    description="DESARROLLO DE PROYECTOS DE INTELIGENCIA ARTIFICIAL-HL - 1HL - (2024-24)",
)


class Clasificacion(BaseModel):
    name: str
    clase: str
    score: float


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: list[str] = []


@app.post("/items/")
async def create_item(item: Item) -> Item:
    return item


@app.get("/items/")
async def read_items() -> list[Item]:
    return [
        Item(name="Portal Gun", price=42.0),
        Item(name="Plumbus", price=32.0),
    ]


@app.get("/clasificar/{path_image}")
async def clasificar_imagen(path_image: str) -> list[Clasificacion]:
    return [
        Clasificacion(name="abcd123", clase="Cat", score=0.90),
        Clasificacion(name="abcd123", clase="Dog", score=0.05),
        Clasificacion(name="abcd124", clase="castor", score=0.05),
    ]


@app.post("/files/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@app.post("/predict/")
async def predict(uploaded_file: UploadFile = File(...)):
    _model = Model()
    return await _model.predict(uploaded_file)


# # Custom Swagger UI HTML template
# def custom_swagger_ui_html(*args, **kwargs):
#     html = get_swagger_ui_html(*args, **kwargs)
#     # Customize the HTML here
#     # For example, adding a custom button
#     custom_button = '<a href="https://example.com/docs" target="_blank" class="my-custom-button">DESARROLLO DE PROYECTOS DE INTELIGENCIA ARTIFICIAL-HL - 1HL - (2024-24)</a>'
#     html = html.replace(
#         "</head>",
#         "<style>.my-custom-button { color: #fff; background-color: #007BFF; padding: 10px 20px; border-radius: 5px; text-decoration: none; }</style></head>",
#     )
#     html = html.replace("</head>", custom_button + "</head>")
#     return Response(content=html, media_type="text/html")


# app.openapi = custom_swagger_ui_html
