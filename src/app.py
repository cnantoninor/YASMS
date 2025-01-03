from datetime import datetime
import io
import json
import logging
import os
import glob
import traceback
import zipfile
import shutil
from typing import List
from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
import pandas as pd
from app_startup import bootstrap_app
import config
from model_instance import ModelInstance, models
from prediction_model import PredictionInput, PredictionOutput
from utils import check_valid_biz_task_model_pair
from task_manager import tasks_queue
from trainer import TrainingTask


bootstrap_app()

app = FastAPI(title="Y.A.M.S (Yet Another Model Server)", version="0.9.1")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Log the error
    logging.error("Validation error: %s - Request: %s", exc, request)
    # Return a JSON response with the details of the validation error
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "body": exc.body,
        },
    )


async def request_to_json(request: Request) -> str:
    data = {
        "method": request.method,
        "url": str(request.url),
        "headers": str(request.headers),
        "pathParams": request.path_params,
        "queryParams": str(request.query_params),
        "client": str(request.client),
    }

    if (
        request.method == "POST"
        and "content-type" in request.headers
        and request.headers.get("content-type")
        .lower()
        .startswith("multipart/form-data")
    ):
        try:
            form_data = await request.form()
            form_data_dict = {}
            for key, value in form_data.items():
                if is_json_serializable(value):
                    form_data_dict[key] = value
            data["formData"] = form_data_dict
        except Exception as e:
            logging.error(
                "An error occurred while trying to get `form_data`: %s", str(e)
            )

    return data


def is_json_serializable(field):
    try:
        json.dumps(field)
        return True
    except TypeError:
        return False


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    ret_code = 400
    request_json = await request_to_json(request)

    logging.error(
        "BAD REQUEST `%s` for request `%s` due to error: `%s`\n%s",
        ret_code,
        request_json,
        str(exc),
        traceback.format_exc(),
    )
    delete_dir_if_upload_training_data_failed(request)

    return JSONResponse(
        status_code=ret_code,
        content={
            "error": {
                "code": ret_code,
                "message": f"Bad Request: {exc.__class__.__name__} - {str(exc)}",
                "request": request_json,
            }
        },
    )


@app.exception_handler(json.JSONDecodeError)
@app.exception_handler(Exception)
async def unhandeld_exception_handler(request: Request, exc: Exception):
    ret_code = 500
    request_json = await request_to_json(request)

    logging.error(
        "Returning http error code `%s` for request `%s` due to error: `%s`\n%s",
        ret_code,
        request_json,
        str(exc),
        traceback.format_exc(),
    )
    delete_dir_if_upload_training_data_failed(request)

    return JSONResponse(
        status_code=ret_code,
        content={
            "error": {
                "code": ret_code,
                "message": f"Unexpected exception: {exc.__class__.__name__} - {str(exc)}",
                "request": request_json,
            }
        },
    )


def delete_dir_if_upload_training_data_failed(request: Request) -> None:
    if hasattr(request.state, "uploaded_data_dir"):
        directory = request.state.uploaded_data_dir
        if os.path.exists(directory):
            shutil.rmtree(directory)


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

    with open(config.LOG_FILE, encoding="utf-8") as f:
        applog = f.read().split("\n")[::-1]
        applog = [line for line in applog if line.strip()]

    if os.path.exists(config.UVICORN_LOG_FILE):
        with open(config.UVICORN_LOG_FILE, encoding="utf-8") as f:
            uvicorn_log = f.read().split("\n")[::-1]
            uvicorn_log = [line for line in uvicorn_log if line.strip()]
    else:
        logging.warning("Uvicorn log file not found: %s", config.UVICORN_LOG_FILE)
        logging.warning("Uvicorn log file not found: %s", config.UVICORN_LOG_FILE)

    if os.path.exists(config.UVICORN_ERR_LOG_FILE):
        with open(config.UVICORN_ERR_LOG_FILE, encoding="utf-8") as f:
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

    Retrieves the details of all the active model instances for each model type.
    An active model is the newest model instance for each model type.

    """
    return JSONResponse(content={"activeModels": models.get_active_models()})


@app.get("/models", tags=["models"])
async def get_models(verbose: bool = True, regex_filter: str = None):
    """
    Retrieves the details of all the models eventually filtered by the `regex_filter`.
    E.g.: spam_classifier/* will return all the models for the `spam_classifier` business task.

    # Parameters:
        - verbose (bool, optional): A flag to indicate whether to include the model details in the response.
            Defaults to True.
            If False, only the model instance names are included in the response.

        - regex_filter (str, optional): An optional regular expression filter to apply on the model instance names.

    # Returns:
        dict: A dictionary containing the state of all the model instances
            that match the eventual filter and the eventual filter applied.
    """
    return JSONResponse(
        content={
            "regex_filter": regex_filter,
            "models": models.to_json(
                verbose=verbose,
                regex=regex_filter,
            ),
        }
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
    request: Request,  # Add this line to define the "request" object
    biz_task: str,
    mod_type: str,
    project: str,
    train_data: UploadFile = File(...),
    features_fields: List[str] = Form(...),
    target_field: str = Form(...),
):
    """
    Uploads the COMMA (not TAB or other separator) separated training data file to the specified model type and model
    name directory and submit an asynchrounous training task.

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

        - If the fields are in a wrong format in respect or a specific biz_task rules isn't respected,
            an error is raised.

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

    request.state.uploaded_data_dir = uploaded_data_dir

    # Check if the file is a zip file
    if zipfile.is_zipfile(io.BytesIO(contents)):
        # If it's a zip file, extract it
        with zipfile.ZipFile(io.BytesIO(contents), "r") as zip_ref:
            zip_ref.extractall(path=uploaded_data_dir)
    else:
        filename = os.path.basename(train_data.filename)
        # If it's not a zip file, write it directly
        with open(os.path.join(uploaded_data_dir, filename), "wb") as f:
            f.write(contents)

    _write_features_and_target_fields(uploaded_data_dir, features_fields, target_field)
    _check_csv_file(uploaded_data_dir, features_fields, target_field)
    model_instance = ModelInstance(uploaded_data_dir.as_posix())
    tasks_queue.submit(TrainingTask(model_instance))
    _clean_train_data_dir_if_needed(uploaded_data_dir.parent)

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


def _write_features_and_target_fields(directory, features_fields, target_field):
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


def _check_csv_file(directory, features_fields, target_field):
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    if len(csv_files) == 0:
        raise ValueError(f"No CSV file found in the train data dir: {directory}.")

    file_name = csv_files[0]
    df = pd.read_csv(file_name, sep=",")

    if df.columns.str.contains("\t").any():
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
        raise ValueError(
            f"The following columns are missing in the {csv_files[0]}: {missing_columns}, parsed dataframe shape is: {df.shape}, parsed columns are: {df.columns}."
        )

    if len(df) < 50:
        raise ValueError(
            f"The {csv_files[0]} must have at least 50 rows, parsed dataframe shape is: {df.shape}, parsed columns are: {df.columns}."
        )

    model_data_file_name = os.path.join(directory, config.Constants.MODEL_DATA_FILE)
    os.rename(file_name, model_data_file_name)


def _clean_train_data_dir_if_needed(directory: str) -> None:
    # Get a list of all subdirectories
    subdirectories = [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ]

    # If there are more than 5 subdirectories
    if len(subdirectories) > 5:
        # Sort the subdirectories alphabetically
        subdirectories.sort()

        # Remove the ones that are on top of the sorted list until only 5
        # remain
        for subdirectory in subdirectories[:-5]:
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
    prediction_input: PredictionInput,
) -> PredictionOutput:
    """
    Perform a prediction based on the given parameters.

    Args:
        biz_task (str): The business task for the prediction.
        mod_type (str): The type of model to use for the prediction.
        project (str): The project to use for the prediction.
        features (List[str]): The list of features to use for the prediction.

    Returns:
        PredictionOutput: The prediction output object. This object contains the timestamp of the prediction, the model ID, and the predictions.
            Predictions are a list of features and their corresponding predicted values.
    """
    check_valid_biz_task_model_pair(biz_task, mod_type)

    model_type_id = f"{biz_task}/{mod_type}/{project}"

    model_instance = models.get_active_model_for_type(model_type_id)

    return model_instance.predict(prediction_input)


@app.get("/isalive", tags=["observability"])
async def isalive():
    """
    Check if the application is alive.

    Returns:
        dict: A dictionary indicating whether the application is alive.
            The dictionary has a single key-value pair: "alive" -> True.
    """
    return {"alive": True}
