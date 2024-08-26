import functools
import threading
import time
import uuid

from fastapi import Request
import supervisely as sly

import json
import numpy as np

import model_functions

import sly_globals as g
import sly_functions as f



requests_statuses = {}


def warn_on_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        value = None
        try:
            value = func(*args, **kwargs)
        except Exception as e:
            sly.logger.warn(f'{e}', exc_info=True)
        return value

    return wrapper


def _inference(data_to_process, request_uuid=None):
    indexes = []
    embeddings = []
    for batch in sly.batched(data_to_process, batch_size=g.batch_size):
        f.cache_images(batch)
        f.crop_images(batch)
        filtered_batch = [row for row in batch if row['cached_image'] is not None]
        images_to_process = np.asarray([np.asarray(row['cached_image']) for row in filtered_batch])
        indexes.extend([row['index'] for row in filtered_batch])
        batch_embeddings = f.batch_inference(images_to_process)
        embeddings.extend(batch_embeddings)

    output_data = json.dumps(str([{'index': index,
                                   'embedding': list(embedding)} for index, embedding in zip(indexes, embeddings)]))

    if request_uuid is not None:
        requests_statuses[request_uuid]["data"] = output_data
        requests_statuses[request_uuid]["status"] = "done"
    return output_data

@g.my_server.post("/inference")
@warn_on_exception
@sly.timeit
def inference(request: Request):
    sly.logger.debug("Inference request", extra={'request': request})
    state = request.state.state
    data_to_process = list(state['input_data'])

    return {"data": _inference(data_to_process)}


@g.my_server.post("/inference_async")
def inference_async(request: Request):
    inference_request_uuid = uuid.uuid5(
        namespace=uuid.NAMESPACE_URL, name=f"{time.time()}"
    ).hex
    state = request.state.state
    data_to_process = list(state['input_data'])
    requests_statuses[inference_request_uuid] = {
        "status": "processing",
        "data": None,
    }
    threading.Thread(target=_inference, args=(data_to_process, inference_request_uuid)).start()
    return {
        "inference_request_uuid": inference_request_uuid,
        "message": "Inference request has been accepted",
    }


@g.my_server.post("/get_inference_status")
def get_inference_request_status(request: Request):
    request_uuid = request.state.state["inference_request_uuid"]
    if request_uuid in requests_statuses:
        return {"status":requests_statuses[request_uuid]["status"]}
    else:
        return {"status": "not found"}


@g.my_server.post("/get_inference_result")
def get_inference_result(request: Request):
    request_uuid = request.state.state["inference_request_uuid"]
    if request_uuid in requests_statuses:
        if requests_statuses[request_uuid]["status"] == "done":
            request = requests_statuses.pop(request_uuid)
            return {"status": request["status"], "data": request["data"]}
        else:
            return {"status": "processing", "data": None}
    else:
        return {"status": "not found", "data": None}


@g.my_server.post("/get_info")
@warn_on_exception
@sly.timeit
def get_info():
    if g.selected_weights_type == 'pretrained':
        output_data = {'weightsType': g.selected_weights_type}
        output_data.update(g.model_info)

    else:
        output_data = {
            'weightsType': g.selected_weights_type,
            'Model': g.remote_weights_path.split('/')[-1]
        }

    output_data["support_async_inference"] = True

    return output_data

@g.my_server.post("/get_session_info")
def get_session_info():
    return get_info()


@g.my_server.post("/is_deployed")
def is_deployed():
    return {
        "deployed": True,
        "description:": "Model is ready to receive requests",
    }


def main():
    sly.logger.info("Script arguments", extra={

        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    model_functions.initialize_network()
    sly.logger.info("Downloading model and config...")
    f.download_model_and_config()
    sly.logger.info("Model downloaded, downloading weights...")
    model_functions.load_weights(g.local_weights_path)
    sly.logger.info("Weights downloaded.")

    sly.logger.info("🟩 Model has been successfully deployed")
    sly.logger.debug("Script arguments", extra={
        "Remote weights": g.remote_weights_path,
        "Local weights": g.local_weights_path,
        "device": g.device
    })


main()
app = g.my_app
