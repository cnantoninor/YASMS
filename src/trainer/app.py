from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import zipfile

app = FastAPI()


@app.post("/upload")
async def upload_file(model_data: UploadFile = File(...), model_name: str = Form(...)):
    """
    Uploads a file and returns the filename and model name.

    Args:
        model_data (UploadFile): The file to be uploaded.
        model_name (str): The name of the model.

    Returns:
        dict: A dictionary containing the filename and model name.
    """
    contents = await model_data.read()

    if zipfile.is_zipfile(model_data.file):
        with zipfile.ZipFile(model_data.file, "r") as zip_ref:
            with zip_ref.open("model.csv") as unzipped_file:
                contents = unzipped_file.read()

    print(">>>>>>>>>>>>>>>>>>>>>>>>>> ")
    print(contents)
    return {"filename": model_data.filename, "model_name": model_name}
