import functools
import os

import supervisely as sly

import json
import numpy as np

import model_functions

import sly_globals as g
import sly_functions as f


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


@g.my_app.callback("inference")
@warn_on_exception
@sly.timeit
def inference(api: sly.Api, task_id, context, state, app_logger):
    data_to_process = list(state['input_data'])

    indexes = []
    embeddings = []
    # for batch in sly.batched(data_to_process, batch_size=g.batch_size):
    #     f.cache_images(batch)
    #     f.crop_images(batch)
    #     filtered_batch = [row for row in batch if row['cached_image'] is not None]
    #     images_to_process = np.asarray([np.asarray(row['cached_image']) for row in filtered_batch])
    #     indexes.extend([row['index'] for row in filtered_batch])
    #     batch_embeddings = f.batch_inference(images_to_process)
    #     embeddings.extend(batch_embeddings)
    embeddings = np.random.rand(len(data_to_process), 512*64)
    indexes = [i for i in range(len(data_to_process))]

    output_data = json.dumps(str([{'index': index,
                                   'embedding': list(embedding)} for index, embedding in zip(indexes, embeddings)]))

    sly.logger.info(f"embeddings shape: {embeddings.shape}", extra={"shape": embeddings.shape, "data_len": len(data_to_process)})
    
    request_id = context["request_id"]
    save_to_team_files = state.get("save_to_team_files", False)
    if save_to_team_files:
        tmp_path = f'/tmp/output_data.json'
        remote_path = f"{g.remote_embeddings_dir}tmp_output_data.json"
        with open(tmp_path, 'w') as file:
            file.write(output_data)
        api.file.upload(g.team_id, tmp_path, remote_path)
        os.remove(tmp_path)
        output_data = {"team_files_path": remote_path}

    g.my_app.send_response(request_id, data=output_data)


@g.my_app.callback("get_info")
@warn_on_exception
@sly.timeit
def get_info(api: sly.Api, task_id, context, state, app_logger):
    if g.selected_weights_type == 'pretrained':
        output_data = {'weightsType': g.selected_weights_type}
        output_data.update(g.model_info)

    else:
        output_data = {
            'weightsType': g.selected_weights_type,
            'Model': g.remote_weights_path.split('/')[-1]
        }

    output_data = json.dumps(str(output_data))

    request_id = context["request_id"]
    g.my_app.send_response(request_id, data=output_data)


def main():
    sly.logger.info("Script arguments", extra={

        "modal.state.slyFile": g.remote_weights_path,
        "device": g.device
    })

    model_functions.initialize_network()
    f.download_model_and_config()
    model_functions.load_weights(g.local_weights_path)

    sly.logger.info("🟩 Model has been successfully deployed")
    sly.logger.debug("Script arguments", extra={
        "Remote weights": g.remote_weights_path,
        "Local weights": g.local_weights_path,
        "device": g.device
    })

    g.my_app.run()


if __name__ == "__main__":
    sly.main_wrapper("main", main)
