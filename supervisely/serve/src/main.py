import functools

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


@g.my_server.post("/inference")
@warn_on_exception
@sly.timeit
def inference(request):
    state = request.state
    data_to_process = list(state['input_data'])

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

    return output_data


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

    output_data = json.dumps(str(output_data))

    return output_data


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


main()
app = g.my_app
