from datetime import datetime
import io
import logging
import os
import glob
import traceback
import zipfile
import shutil
from typing import List
from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.responses import JSONResponse, RedirectResponse
import pandas as pd
from app_startup import bootstrap_app
import config
from model_instance import ModelInstance, Models
from prediction_output import PredictionOutput
from utils import check_valid_biz_task_model_pair
from task_manager import tasks_queue
from trainer import TrainingTask

bootstrap_app()

app = FastAPI(title="Y.A.M.S (Yet Another Model Server)", version="0.2")


def request_to_json(request: Request) -> str:
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": str(request.headers),
        "pathParams": request.path_params,
        "queryParams": str(request.query_params),
        "client": str(request.client),
    }


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    ret_code = 400
    request_json = request_to_json(request)

    logging.error(
        "Returning http error code `%s` for request `%s` due to error: `%s`\n%s",
        ret_code,
        request_json,
        str(exc),
        traceback.format_exc(),
    )

    return JSONResponse(
        status_code=ret_code,
        content={
            "error": {
                "code": ret_code,
                "message": "Bad Request: %s" % str(exc),
                "request": request_json,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandeld_exception_handler(request: Request, exc: Exception):
    ret_code = 500
    request_json = request_to_json(request)

    logging.error(
        "Returning http error code `%s` for request `%s` due to error: `%s`\n%s",
        ret_code,
        request_json,
        str(exc),
        traceback.format_exc(),
    )

    return JSONResponse(
        status_code=ret_code,
        content={
            "error": {
                "code": ret_code,
                "message": f"Unexpected exception: {str(exc)}",
                "request": request_json,
            }
        },
    )


@app.get("/", include_in_schema=False)
def read_root():
    return RedirectResponse(url="/docs")


@app.get("/logs", tags=["observability"])
async def get_app_logs():
    """
    Retrieves the content of the application logs file in REVERSED order: the last line is the first one in the list.

        - app.log file used for logging the application logs.
        - uvicorn.log file used for logging the uvicorn logs.
        - uvicorn.err.log file used for logging the uvicorn error logs.

    """

    with open(config.LOG_FILE, encoding="utf8") as f:
        applog = f.read().split("\n")[::-1]
        applog = [line for line in applog if line.strip()]

    if os.path.exists(config.UVICORN_LOG_FILE):
        with open(config.UVICORN_LOG_FILE, encoding="utf8") as f:
            uvicorn_log = f.read().split("\n")[::-1]
            uvicorn_log = [line for line in uvicorn_log if line.strip()]
    else:
        logging.warning("Uvicorn log file not found: %s", config.UVICORN_LOG_FILE)

    if os.path.exists(config.UVICORN_ERR_LOG_FILE):
        with open(config.UVICORN_ERR_LOG_FILE, encoding="utf8") as f:
            uvicorn_err_log = f.read().split("\n")[::-1]
            uvicorn_err_log = [line for line in uvicorn_err_log if line.strip()]
    else:
        logging.warning(
            "Uvicorn error log file not found: %s", config.UVICORN_ERR_LOG_FILE
        )

    return JSONResponse(
        content={
            f"{config.LOG_FILE}": applog,
            f"{config.UVICORN_LOG_FILE}": uvicorn_log,
            f"{config.UVICORN_ERR_LOG_FILE}": uvicorn_err_log,
        },
    )


@app.get("/tasks/queue", tags=["observability"])
async def get_tasks_queue():
    return JSONResponse(content={"tasksQueue": tasks_queue.to_json()})


@app.get("/models/active", tags=["models"])
async def get_active_models():
    """

    # NOT IMPLEMENTED YET

    """
    raise NotImplementedError("This endpoint is not implemented yet.")


@app.get("/models", tags=["models"])
async def get_all_models():
    """
    Retrieves the details of all the models.

    # Returns:
        dict: A dictionary containing the state of all the model instances.
    """
    return JSONResponse(
        content={"models": Models(config.data_path.as_posix()).to_json(verbose=True)}
    )


@app.get("/models/registered_types", tags=["models"])
async def get_registered_types():
    """
    Retrieves the available business tasks and model types registered in the server.

    Returns:
        json: A json containing the available business tasks and model types.

    """
    return JSONResponse(
        content={"validBizTasksModelPair": config.Constants.VALID_BIZ_TASK_MODEL_PAIR},
    )


@app.post("/models/{biz_task}/{mod_type}/{project}/upload_train_data", tags=["models"])
async def upload_train_data(
    biz_task: str,
    mod_type: str,
    project: str,
    train_data: UploadFile = File(...),
    features_fields: List[str] = Form(...),
    target_field: str = Form(...),
):
    """
    Uploads the COMMA (not TAB or other separator) separated training data file to the specified model type and model name directory and submit an asynchrounous training task.

    ## Args:
        - biz_task (str): The business task, e.g. spam_classifier.

        - mod_type (str): The type of the model, e.g. KNN, SVM, etc..

        - project (str): The name of the project.

        - train_data (UploadFile): The CSV file to be uploaded, it can be zipped but must be a COMMA separated file.

        - features_fields (List[str]): the list of the fields in the
            CSV file `train_data` that will be used as features. Existence of fields will be checked.

        - target_field (str): the field in the
            CSV file `train_data` that will be used as target. Existence of the field will be checked.

    ## Returns:
        - dict: A dictionary containing the uploaded train data path.

    ## Raises:
        - If the file is not a CSV or zip file, an error is raised.

        - If the file does not contain the indicated features and target fields, an error is raised.

        - If the fields are in a wrong format in respect or a specific biz_task rules isn't respected, an error is raised.

    ## Specific biz tasks checks for the fields:
        - *spam_classifier*: the *target_field* should have 0 or 1. The features fields should be strings.
            Exception: if the *target_field* name is `Stato Workflow` then the service will automatically convert Y/N/D values:
            {"Y": 1, "D": 0} and remove the rows with `N` value.

    """

    check_valid_biz_task_model_pair(biz_task, mod_type)

    contents = await train_data.read()

    uploaded_data_dir = (
        config.data_path.joinpath(biz_task)
        .joinpath(mod_type)
        .joinpath(project)
        .joinpath(determine_model_instance_name_date_path())
    )

    os.makedirs(uploaded_data_dir, exist_ok=True)

    # Check if the file is a zip file
    if zipfile.is_zipfile(io.BytesIO(contents)):
        # If it's a zip file, extract it
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            zip_ref.extractall(path=uploaded_data_dir)
    else:
        # If it's not a zip file, write it directly
        with open(os.path.join(uploaded_data_dir, train_data.filename), "wb") as f:
            f.write(contents)

    __write_features_and_target_fields(uploaded_data_dir, features_fields, target_field)
    __check_csv_file(uploaded_data_dir, features_fields, target_field)
    __clean_train_data_dir_if_needed(uploaded_data_dir.parent)
    model_instance = ModelInstance(uploaded_data_dir.as_posix())
    tasks_queue.submit(TrainingTask(model_instance))

    logging.info(
        """Successfully uploaded train data and submitted train task for ModelInstance: `%s`""",
        model_instance,
    )

    return JSONResponse(
        content={
            "modelInstance": model_instance.to_json(),
            "path": uploaded_data_dir.relative_to(config.root_path).as_posix(),
        }
    )


def __write_features_and_target_fields(directory, features_fields, target_field):
    with open(
        os.path.join(directory, config.Constants.FEATURES_FIELDS_FILE),
        "w",
        encoding="utf8",
    ) as f:
        f.write("\n".join(features_fields))

    with open(
        os.path.join(directory, config.Constants.TARGET_FIELD_FILE),
        "w",
        encoding="utf8",
    ) as f:
        f.write(target_field)


def __check_csv_file(directory, features_fields, target_field):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if len(csv_files) == 0:
        raise ValueError(f"No CSV file found in the train data dir: {directory}.")

    df = pd.read_csv(csv_files[0], sep=",")

    if df.columns.str.contains("\t").any():
        shutil.rmtree(directory)
        raise ValueError(
            f"The file {csv_files[0]} is tab separated. It should be comma separated. Parsed dataframe shape is: {df.shape}, parsed columns are: {df.columns}."
        )

    missing_columns = []
    for column in features_fields:
        if column not in df.columns:
            missing_columns.append(column)

    if target_field not in df.columns:
        missing_columns.append(target_field)

    if missing_columns:
        shutil.rmtree(directory)
        raise ValueError(
            f"The following columns are missing in the {csv_files[0]}: {missing_columns}, parsed dataframe shape is: {df.shape}, parsed columns are: {df.columns}."
        )

    if len(df) < 50:
        shutil.rmtree(directory)
        raise ValueError(
            f"The {csv_files[0]} must have at least 50 rows, parsed dataframe shape is: {df.shape}, parsed columns are: {df.columns}."
        )


def __clean_train_data_dir_if_needed(directory: str) -> None:
    # Get a list of all subdirectories
    subdirectories = [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]

    # If there are more than 10 subdirectories
    if len(subdirectories) > 10:
        # Sort the subdirectories alphabetically
        subdirectories.sort()

        # Remove the ones that are on top of the sorted list until only 10
        # remain
        for subdirectory in subdirectories[:-10]:
            shutil.rmtree(subdirectory)


def determine_model_instance_name_date_path() -> str:
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H-%M-%S-%f")
    return date_time_str


@app.post(
    "/models/{biz_task}/{mod_type}/{project}/predict",
    tags=["models"],
    response_model=PredictionOutput,
)
async def predict(
    biz_task: str,
    mod_type: str,
    project: str,
    features: List[str] = Form(...),
):
    """
    Perform a prediction based on the given parameters.

    Args:
        biz_task (str): The business task for the prediction.
        mod_type (str): The type of model to use for the prediction.
        project (str): The project to use for the prediction.
        features (List[str], optional): The list of features to use for the prediction. Defaults to Form(...).

    Returns:
        The prediction result.

    Raises:
        Any exceptions that may occur during the prediction process.

    # NOT IMPLEMENTED YET

    """

    pass
