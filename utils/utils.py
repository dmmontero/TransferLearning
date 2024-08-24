from fastapi import UploadFile
import io


def getFile(self, file_path):
    """
    Lee un archivo de disco y retorna un objeto UoploadFile con el contenido del archivo.

    Parameters:
        file_path (str): ruta del archivo en disco.

    Returns:
        UploadFile: Objeto UploadFile con el contenido del archivo.
    """
    # Leer el archivo desde el disco
    with open(file_path, "rb") as f:
        file_content = f.read()

    # Crear el objeto UploadFile
    return UploadFile(file=io.BytesIO(file_content))
